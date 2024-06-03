# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu
import argparse
import codecs
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from eval.evaluate_lm import R2D2GenFastEvaluator
from eval.grammar_induction import (TreeFormat, convert_to_bracket,
                                    convert_to_ptb)
from model.model_factory import create_model
from model.weighted_sum_func import WeightedSumFunc
from reader.data_collator import TextCollator
from reader.dataset_text import TextDataset
from utils.beam_searcher import R2D2GenFastBeamSearcher
from utils.misc import (align_spans, convert_token, get_sentence_from_words,
                        gpt_token)
from utils.model_loader import get_max_epoch_step, load_checkpoint
from utils.tree_utils import get_tree_from_merge_trajectory


def print_grad_hook(grad):
    if grad is not None:
        print('Gradient received: ', grad.abs().sum().item())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _scalar(val):
    if val is not None:
        if isinstance(val, torch.Tensor):
            return val.item()
        return val
    return 0

class LinearProgressScheduler:
    def __init__(self, start, end, proportion, total_steps):
        # e.g. proportion = 0.8
        # then val will go from start to end at previous 80% steps and keep end in the last 20% steps
        self._start = start
        self._end = end
        self._total_steps = total_steps * proportion

    def update(self, current_step):
        r = min(1.0, current_step / self._total_steps)
        return self._start * (1 - r) + self._end * r


class InsidePrinter(object):
    def __init__(self, 
                 modeltype, 
                 model,
                 dataloader,
                 tokenizer,
                 device, 
                 index):
        self.model = model
        self.model_type = modeltype
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self._sep_word = ' '
        self.device = device
        self.index = index
    
    def dooutput(self, data_mode, epoch_num):
        self.model.eval()
        # TODO: check if this is right!!!
        if args.model_type in ['r2d2-gen', 'r2d2-gen-fast', 'r2d2-gen-fast-struct', 'r2d2-gen-fast-ext']:
            self.model.enable_gpt = False
        
        data_iterator = tqdm(self.dataloader, desc="Iteration")
        with codecs.open(os.path.join(args.output_dir, f'wsj_'+data_mode+f'_pred_{self.index}_inside_ptb.txt'), mode='w', encoding='utf-8') as f_out1, \
            codecs.open(os.path.join(args.output_dir, f'wsj_'+data_mode+f'_pred_{self.index}_inside_bracket.txt'), mode='w', encoding='utf-8') as f_out2:
            with torch.no_grad():
                for _, inputs in enumerate(data_iterator):
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)

                    result = self.model(**inputs)  
                    assert result.splits is not None
                    
                    for sent_id in range(inputs["masks"].shape[0]):
                        seq_len = inputs["masks"][sent_id].sum()
                        input_ids = inputs['input_ids'][sent_id, :seq_len].cpu().data.numpy()
                        split_points = [_ for _ in reversed(result.splits[sent_id, :seq_len - 1].cpu().data.numpy())]
                        root = get_tree_from_merge_trajectory(split_points, seq_len)

                        tokens = ''.join([convert_token(item) for item in self.tokenizer.convert_ids_to_tokens(input_ids)]).split()
                        # print("tokens:", tokens)
                        # gold: 
                        # ['Skipper', "'s", 'Inc.', 'Bellevue', 'Wash.', 'said', 'it', 'signed', 'a', 'definitive', 'merger', 'agreement', 'for', 'a', 'National', 'Pizza', 'Corp.', 'unit', 'to', 'acquire', 'the', '90.6', '%', 'of', 'Skipper', "'s", 'Inc.', 'it', 'does', "n't", 'own', 'for', '11.50', 'a', 'share', 'or', 'about', '28.1', 'million']
                        sentence, spans = get_sentence_from_words(tokens, self._sep_word)
                        # print("sentence: ", sentence)
                        # print("spans: ", spans)
                        # exit()
                        outputs = self.tokenizer.encode_plus(sentence,
                                                              add_special_tokens=False,
                                                              return_offsets_mapping=True)
                        offset_mapping = outputs['offset_mapping']
                        word_starts, word_ends = align_spans(spans, offset_mapping)
                        atom_spans = []
                        indices_mapping = [0] * len(outputs['input_ids'])
                        for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
                            if ed > st:
                                atom_spans.append([st, ed])
                            for idx in range(st, ed + 1):
                                indices_mapping[idx] = pos

                        # print("tokens: ", tokens)
                        # print(f"root: {root}")
                        output1 = convert_to_ptb(root, tokens, atom_spans, indices_mapping)
                        if output1.startswith('(T-1'):
                            output1 = f'(NT-1 {output1})'
                        output2 = convert_to_bracket(root, tokens, atom_spans, indices_mapping)
                        print(output1, file=f_out1)
                        print(output2, file=f_out2)
        
        if args.model_type in ['r2d2-gen', 'r2d2-gen-fast', 'r2d2-gen-fast-struct', 'r2d2-gen-fast-ext']:
            self.model.enable_gpt = True


class GenerativePrinter(object):
    def __init__(self, 
                 modeltype,
                 model,  
                 beam_searcher, 
                 dataloader,
                 tokenizer,
                 device,
                 index):
        self._beam_searcher = beam_searcher
        self.model_type = modeltype
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self._sep_word = ' '
        self.device = device
        self.index = index

    def dooutput(self, data_mode, epoch_num):
        self.model.eval()
        
        data_iterator = tqdm(self.dataloader, desc="Iteration")
        with codecs.open(os.path.join(args.output_dir, f'wsj_'+data_mode+f'_pred_{self.index}_generative_ptb.txt'), mode='w', encoding='utf-8') as f_out1, \
            codecs.open(os.path.join(args.output_dir, f'wsj_'+data_mode+f'_pred_{self.index}_generative_bracket.txt'), mode='w', encoding='utf-8') as f_out2:
            with torch.no_grad():
                for _, inputs in enumerate(data_iterator):
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)
                    
                    # TODO: updata target_ids and target masks
                    states = self._beam_searcher.beam_search(target_ids=inputs["chunk_input_ids"], 
                                                             target_masks=(inputs["chunk_masks"]>0).long(),
                                                             atom_spans=inputs["atom_spans"]) 
                    
                    for sent_id in range(inputs["masks"].shape[0]):
                        seq_len = inputs["masks"][sent_id].sum()
                        input_ids = inputs['input_ids'][sent_id, :seq_len].cpu().data.numpy()
                        root = states[sent_id][0].stack_top

                        tokens = ''.join([convert_token(item) for item in self.tokenizer.convert_ids_to_tokens(input_ids)]).split()
                        # gold: 
                        # ['Skipper', "'s", 'Inc.', 'Bellevue', 'Wash.', 'said', 'it', 'signed', 'a', 'definitive', 'merger', 'agreement', 'for', 'a', 'National', 'Pizza', 'Corp.', 'unit', 'to', 'acquire', 'the', '90.6', '%', 'of', 'Skipper', "'s", 'Inc.', 'it', 'does', "n't", 'own', 'for', '11.50', 'a', 'share', 'or', 'about', '28.1', 'million']
                        sentence, spans = get_sentence_from_words(tokens, self._sep_word)
                        # logger.info(f'sentence: {sentence}')
                        outputs = self.tokenizer.encode_plus(sentence,
                                                              add_special_tokens=False,
                                                              return_offsets_mapping=True)
                        offset_mapping = outputs['offset_mapping']
                        word_starts, word_ends = align_spans(spans, offset_mapping)
                        atom_spans = []
                        indices_mapping = [0] * len(outputs['input_ids'])
                        for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
                            if ed > st:
                                atom_spans.append([st, ed])
                            for idx in range(st, ed + 1):
                                indices_mapping[idx] = pos

                        # print("tokens: ", tokens)
                        # print(f"root: {root}")
                        output1 = convert_to_ptb(root, tokens, atom_spans, indices_mapping)
                        if output1.startswith('(T-1'):
                            output1 = f'(NT-1 {output1})'
                        output2 = convert_to_bracket(root, tokens, atom_spans, indices_mapping)
                        print(output1, file=f_out1)
                        print(output2, file=f_out2)


class GenerativeNosyncPrinter(object):
    def __init__(self, 
                 modeltype,
                 model,  
                 beam_searcher, 
                 dataloader,
                 tokenizer,
                 device,
                 index):
        self._beam_searcher = beam_searcher
        self.model_type = modeltype
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self._sep_word = ' '
        self.device = device
        self.index = index

    def dooutput(self, data_mode, epoch_num):
        self.model.eval()
        
        data_iterator = tqdm(self.dataloader, desc="Iteration")
        with codecs.open(os.path.join(args.output_dir, f'wsj_'+data_mode+f'_pred_{self.index}_generativenosync_ptb.txt'), mode='w', encoding='utf-8') as f_out1, \
            codecs.open(os.path.join(args.output_dir, f'wsj_'+data_mode+f'_pred_{self.index}_generativenosync_bracket.txt'), mode='w', encoding='utf-8') as f_out2:
            with torch.no_grad():
                for _, inputs in enumerate(data_iterator):
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)
                    
                    # TODO: updata target_ids and target masks
                    # states = self._beam_searcher.beam_search(target_ids=inputs["chunk_input_ids"], 
                    #                                          target_masks=(inputs["chunk_masks"]>0).long(),
                    #                                          atom_spans=inputs["atom_spans"]) 
                    _, states = self._beam_searcher.beam_search(inputs["chunk_input_ids"], atom_spans=inputs["atom_spans"])
                    
                    for sent_id in range(inputs["masks"].shape[0]):
                        assert sent_id == 0
                        seq_len = inputs["masks"][sent_id].sum()
                        input_ids = inputs['input_ids'][sent_id, :seq_len].cpu().data.numpy()
                        root = states[sent_id][0].stack_top

                        tokens = ''.join([convert_token(item) for item in self.tokenizer.convert_ids_to_tokens(input_ids)]).split()
                        # gold: 
                        # ['Skipper', "'s", 'Inc.', 'Bellevue', 'Wash.', 'said', 'it', 'signed', 'a', 'definitive', 'merger', 'agreement', 'for', 'a', 'National', 'Pizza', 'Corp.', 'unit', 'to', 'acquire', 'the', '90.6', '%', 'of', 'Skipper', "'s", 'Inc.', 'it', 'does', "n't", 'own', 'for', '11.50', 'a', 'share', 'or', 'about', '28.1', 'million']
                        sentence, spans = get_sentence_from_words(tokens, self._sep_word)
                        # logger.info(f'sentence: {sentence}')
                        outputs = self.tokenizer.encode_plus(sentence,
                                                              add_special_tokens=False,
                                                              return_offsets_mapping=True)
                        offset_mapping = outputs['offset_mapping']
                        word_starts, word_ends = align_spans(spans, offset_mapping)
                        atom_spans = []
                        indices_mapping = [0] * len(outputs['input_ids'])
                        for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
                            if ed > st:
                                atom_spans.append([st, ed])
                            for idx in range(st, ed + 1):
                                indices_mapping[idx] = pos

                        # print("tokens: ", tokens)
                        # print(f"root: {root}")
                        output1 = convert_to_ptb(root, tokens, atom_spans, indices_mapping)
                        if output1.startswith('(T-1'):
                            output1 = f'(NT-1 {output1})'
                        output2 = convert_to_bracket(root, tokens, atom_spans, indices_mapping)
                        print(output1, file=f_out1)
                        print(output2, file=f_out2)


class ParserPrinter(object):
    def __init__(self, 
                 modeltype, 
                 model,
                 dataloader,
                 tokenizer,
                 device):
        self.model = model
        self.model_type = modeltype
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self._sep_word = ' '
        self.device = device
    
    def dooutput(self, data_mode, epoch_num):
        self.model.eval()
        parser = self.model.r2d2.parser
        parser.to(device)
        parser.eval()
        
        data_iterator = tqdm(self.dataloader, desc="Iteration")
        with codecs.open(os.path.join(args.output_dir, f'wsj_'+data_mode+f'_pred_{epoch_num}_parser_ptb.txt'), mode='w', encoding='utf-8') as f_out1, \
            codecs.open(os.path.join(args.output_dir, f'wsj_'+data_mode+f'_pred_{epoch_num}_parser_bracket.txt'), mode='w', encoding='utf-8') as f_out2:
            with torch.no_grad():
                for _, inputs in enumerate(data_iterator):
                    for k, v in inputs.items():
                        if v is not None and isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)

                    # print("chunk_input_ids: ", inputs["chunk_input_ids"])
                    # print("chunk_masks: ", inputs["chunk_masks"])
                    # print("atom_spans: ", inputs["atom_spans"])
                    # exit()
                    r2d2_input_ids = torch.where(inputs["chunk_input_ids"] == -100, 0, inputs["chunk_input_ids"])
                    s_indices, _ = parser(r2d2_input_ids, inputs["chunk_masks"], atom_spans=inputs["atom_spans"], noise_coeff=0.0)
                    
                    for sent_id in range(inputs["masks"].shape[0]):
                        seq_len = inputs["masks"][sent_id].sum()
                        input_ids = inputs['input_ids'][sent_id, :seq_len].cpu().data.numpy()
                        split_points = [_ for _ in s_indices[sent_id, :seq_len - 1].cpu().data.numpy()]
                        root = get_tree_from_merge_trajectory(split_points, seq_len)

                        tokens = ''.join([convert_token(item) for item in self.tokenizer.convert_ids_to_tokens(input_ids)]).split()
                        # gold: 
                        # ['Skipper', "'s", 'Inc.', 'Bellevue', 'Wash.', 'said', 'it', 'signed', 'a', 'definitive', 'merger', 'agreement', 'for', 'a', 'National', 'Pizza', 'Corp.', 'unit', 'to', 'acquire', 'the', '90.6', '%', 'of', 'Skipper', "'s", 'Inc.', 'it', 'does', "n't", 'own', 'for', '11.50', 'a', 'share', 'or', 'about', '28.1', 'million']
                        sentence, spans = get_sentence_from_words(tokens, self._sep_word)
                        outputs = self.tokenizer.encode_plus(sentence,
                                                              add_special_tokens=False,
                                                              return_offsets_mapping=True)
                        offset_mapping = outputs['offset_mapping']
                        word_starts, word_ends = align_spans(spans, offset_mapping)
                        atom_spans = []
                        indices_mapping = [0] * len(outputs['input_ids'])
                        for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
                            if ed > st:
                                atom_spans.append([st, ed])
                            for idx in range(st, ed + 1):
                                indices_mapping[idx] = pos

                        # print("tokens: ", tokens)
                        # print(f"root: {root}")
                        output1 = convert_to_ptb(root, tokens, atom_spans, indices_mapping)
                        if output1.startswith('(T-1'):
                            output1 = f'(NT-1 {output1})'
                        output2 = convert_to_bracket(root, tokens, atom_spans, indices_mapping)
                        print(output1, file=f_out1)
                        print(output2, file=f_out2)


class Trainer(object):
    def __init__(self, 
                 model,
                 collator,
                 tokenizer,
                 device,
                 logger,
                 is_master=True,
                 num_workers = 0,
                 lr=5e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.is_master = is_master
        self.logger = logger
        self.collator = collator
        self.num_workers = num_workers

        self.device = device
        self.lr = lr

    def train(self, 
              data_loader: DataLoader, 
              optimizer, 
              scheduler, 
              scaler,
              output_dir,
              valid_printer = None,
              test_printer = None, 
              amp_dtype=torch.float16,
              coeff_scheduler=None,
              temp_scheduler=None,
              log_steps=100, save_steps=100, epochs=1, 
              max_norm=1.0, max_recover_step=-1,
              accumulation_steps=1):

        train_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = len(data_loader) * epochs
        self.model.train()

        for epoch in train_iterator:
            epoch_iterator = tqdm(data_loader, desc="Iteration")
            for step, inputs in enumerate(epoch_iterator):
                break
                curr_step = step + epoch * len(data_loader)
                if curr_step <= max_recover_step:
                    continue
                max_recover_step = -1

                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)

                coeff = 1.0 if coeff_scheduler is None else coeff_scheduler.update(curr_step)
                temperature = 1.0 if temp_scheduler is None else temp_scheduler.update(curr_step)
                with self.model.no_sync():
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        result = self.model(**inputs, coeff=coeff, temperature=temperature)
                    if result.struct_loss is not None:
                        WeightedSumFunc.a_ij_require_grad = True
                        scaler.scale(result.struct_loss / accumulation_steps).backward(retain_graph=True)
                    WeightedSumFunc.a_ij_require_grad = False
                    scaler.scale(result.non_struct_loss / accumulation_steps).backward()
                
                if (curr_step + 1) % accumulation_steps == 0:
                    # for p in model.parameters():
                    for name, p in self.model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                            p.grad /= torch.distributed.get_world_size()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                if (curr_step + 1) % accumulation_steps == 0:
                    optimizer.zero_grad()

                if curr_step % log_steps == 0 and curr_step > 0:
                    self.logger.info(f'progress:{curr_step}/{total_step} coeff: {coeff} temperature: {temperature} loss: {_scalar(result.non_struct_loss + result.struct_loss)} gpt loss: {_scalar(result.gpt_loss)} ' + \
                        f'inside_outside loss: {_scalar(result.inside_outside_loss)} parser loss: {_scalar(result.parser_loss)} ' + \
                        f'action loss: {_scalar(result.action_loss)}')
                    with torch.no_grad():
                        # output generated binary tree
                        if self.is_master and result.splits is not None:
                            # output binary trees for different iteration epochs
                            # sent_id = np.random.randint(inputs['input_ids'].shape[0])
                            sent_id = 0
                            seq_len = inputs["masks"][sent_id].sum()
                            input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][sent_id].cpu().data.numpy())
                            self.logger.info(f"input sentence: {input_tokens}")
                            tokens = [gpt_token(t) for t in input_tokens]
                            split_points = [_ for _ in reversed(result.splits[sent_id, :seq_len - 1].cpu().data.numpy())]
                            merged_tree = get_tree_from_merge_trajectory(split_points, seq_len, tokens)
                            self.logger.info(f"parsed tree : {merged_tree}")
            
            if self.is_master:
                if valid_printer is not None:
                    valid_printer.dooutput("valid", epoch)
                else:
                    while True:
                        try:
                            torch.save(self.model.state_dict(), os.path.join(output_dir, f'model{epoch}.bin'))
                            break
                        except:
                            time.sleep(5)
                if test_printer is not None:
                    test_printer.dooutput("test", epoch)
                self.model.train()


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Arguments for grammar indcution finetune')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--batch_size', default=8, type=int, help='training batch size')
    cmd.add_argument('--eval_batch_size', default=32, type=int, help='evaluating batch size')
    cmd.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    cmd.add_argument('--parser_lr', default=1e-3, type=float, help='learning rate')
    cmd.add_argument('--r2d2_config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--gpt_config_path', required=True, type=str, help='config for gpt')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--ext_vocab_path', required=False, default=None, type=str, help='external vocab path')
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--valid_corpus_path', required=False, default=None, type=str, help='path to the validation corpus')
    cmd.add_argument('--test_corpus_path', required=False, default=None, type=str, help='path to the test corpus')
    cmd.add_argument('--accumulation_steps', type=int, default=1)
    cmd.add_argument('--model_type', choices=['r2d2-gen', 'gpt', 'llama', 'r2d2', 'r2d2-gen-fast', 'r2d2-fast', 'r2d2-gen-fast-struct', 'r2d2-gen-fast-ext'], default='r2d2-gen')
    cmd.add_argument('--eval_mode', choices=['generative', 'generativenosync', 'inside', 'parser'], default='inside')
    cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--checkpoint_dir', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--pretrain_dir', default=None, type=str)
    cmd.add_argument('--coeff_start', type=float, default=1.0)
    cmd.add_argument('--coeff_end', type=float, default=1.0)
    cmd.add_argument('--coeff_proportion', type=float, default=0.8)
    cmd.add_argument('--temperature_start', type=float, default=1.0)
    cmd.add_argument('--temperature_end', type=float, default=1.0)
    cmd.add_argument('--temperature_proportion', type=float, default=0.8)
    cmd.add_argument('--pool_size', type=int, default=1)
    cmd.add_argument('--seed', type=int, default=3407)
    cmd.add_argument('--fix_embedding', action='store_true')
    cmd.add_argument('--disable_group', action='store_true')
    cmd.add_argument('--warm_up', type=float, default=0.01)
    cmd.add_argument('--log_steps', default=100, type=int)
    cmd.add_argument('--gradient_checkpoint', action='store_true')
    cmd.add_argument('--compile', action='store_true')
    cmd.add_argument('--save_steps', default=500, type=int)
    cmd.add_argument('--cache_dir', required=False, default=None, type=str)
    cmd.add_argument('--epochs', default=10, type=int)
    cmd.add_argument('--beam_size', default=20, type=int)
    cmd.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm")
    cmd.add_argument('--index', default=0, type=int)

    args = cmd.parse_args(sys.argv[1:])
    torch.set_printoptions(profile='full')

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = -1
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)  # for multi-process in a single machine with multiple GPUs.
        global_rank = local_rank
        while True:
            try:
                logging.info('init process group')
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                if torch.distributed.is_initialized():
                    break
            except ValueError:
                time.sleep(5)
            except:
                logging.error('Exit with unknown error')
                exit(-1)
        device = torch.device('cuda')
    else:
        global_rank = -1
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')

    is_master = local_rank == -1 or global_rank == 0
    if not os.path.exists(args.output_dir) and is_master:
        os.mkdir(args.output_dir)
    if is_master:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.output_dir, 'training_log.txt'), mode='a', encoding="utf-8")
        logger.addHandler(fh)
    else:
        logger = logging

    logger.info(f'initialize model on {global_rank}')

    model = create_model(args.model_type, args.r2d2_config_path, args.gpt_config_path, args.fix_embedding, args.gradient_checkpoint)

    max_epoch = -1
    max_step = -1
    
    if args.pretrain_dir is not None:
        model.from_pretrain(args.pretrain_dir, strict=True)
        logger.info("load from pretrain dir successfully")
    if args.checkpoint_dir is not None:
        max_epoch, max_step = get_max_epoch_step(args.checkpoint_dir, 'model*_*.bin')
        print(f'detect max_epoch: {max_epoch}, max_step:{max_step}')
        if max_epoch >= 0:
            logger.info(f'load from checkpoint, turn: {max_epoch}_{max_step}')
            if args.model_type == 'gpt':
                if is_master:
                    state_dicts = torch.load(os.path.join(args.checkpoint_dir, f'model{max_epoch}_{max_step}.bin'), map_location=lambda a, b: a)
                    out_dict = {}
                    for key, val in state_dicts.items():
                        new_key = key.replace('module.gpt.', '')
                        out_dict[new_key] = val

                    torch.save(out_dict, os.path.join(args.checkpoint_dir, f'pytorch_model.bin'))
                torch.distributed.barrier()
                model.from_pretrain(args.checkpoint_dir)
            else:
                model.from_pretrain(os.path.join(args.checkpoint_dir, f'model{max_epoch}_{max_step}.bin'))
            # TODO: add loading from checkpoint for the parser
    
    logger.info(f'move model to gpu:{global_rank}')
    model.to(device=device)

    # named_par_list = list(model.named_parameters())
    # unused_parser_indices = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 107 108 109"
    # unused_parser_indices = [int(t) for t in unused_parser_indices.split()]
    # for idx in unused_parser_indices:
    #     print(named_par_list[idx][0])

    set_seed(args.seed)

    logger.info(f'start loading dataset on {global_rank}')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    dataset = TextDataset(args.corpus_path)
    
    collator = TextCollator(tokenizer, lambda x: (x.split(' '), ' '), external_vocab_path=args.ext_vocab_path)
    collator_fn = collator.collate_fn

    parser_params = []
    model_params = []
    for name, params in model.named_parameters():
        if name.find('.parser.') > 0:
            parser_params.append(params)
        else:
            model_params.append(params)

    if global_rank == -1:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset),
                                collate_fn=collator_fn, num_workers=args.pool_size)
        n_gpu = 1
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        # TODO: seperate learning rate
        optimizer = AdamW([{"params": model_params},
                           {"params": parser_params, "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps // args.accumulation_steps,
                                                    num_training_steps=t_total // args.accumulation_steps)
    elif global_rank >= 0:
        n_gpu = 1
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset, shuffle=False),
                                collate_fn=collator_fn, num_workers=args.pool_size)
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW([{"params": model_params},
                           {"params": parser_params, "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
       
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps // args.accumulation_steps,
                                                    num_training_steps=t_total // args.accumulation_steps)
        ddpmodel = DDP(model)
        
    coeff_scheduler = LinearProgressScheduler(args.coeff_start, args.coeff_end, args.coeff_proportion, t_total)
    temp_scheduler = LinearProgressScheduler(args.temperature_start, args.temperature_end, args.temperature_proportion, t_total)
    scaler = torch.cuda.amp.GradScaler()
    
    if max_epoch >= 0:
        modules = [optimizer, scheduler, scaler]
        files = [f'optimizer{max_epoch}_{max_step}.pt', f'scheduler{max_epoch}_{max_step}.pt', \
                f'scaler{max_epoch}_{max_step}.pt']
        load_checkpoint(modules, files, args.checkpoint_dir)

    if is_master and args.valid_corpus_path is not None and args.test_corpus_path is not None:
        valid_dataset = TextDataset(args.valid_corpus_path)
        test_dataset = TextDataset(args.test_corpus_path)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, sampler=SequentialSampler(valid_dataset),
                                collate_fn=collator_fn, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, sampler=SequentialSampler(test_dataset),
                                collate_fn=collator_fn, num_workers=0)
        if args.eval_mode == "inside":
            valid_printer = InsidePrinter(modeltype=args.model_type, model=model, dataloader=valid_dataloader, tokenizer=tokenizer, device=device, index=args.index)
            test_printer = InsidePrinter(modeltype=args.model_type, model=model, dataloader=test_dataloader, tokenizer=tokenizer, device=device, index=args.index)
        elif args.eval_mode == "generative":
            gptconfig = AutoConfig.from_pretrained(args.gpt_config_path)
            beam_searcher = R2D2GenFastBeamSearcher(model, gptconfig, device, beam_size=args.beam_size)
            valid_printer = GenerativePrinter(modeltype=args.model_type, model=model, beam_searcher=beam_searcher, dataloader=valid_dataloader, tokenizer=tokenizer, device=device, index=args.index)
            test_printer = GenerativePrinter(modeltype=args.model_type, model=model, beam_searcher=beam_searcher, dataloader=test_dataloader, tokenizer=tokenizer, device=device, index=args.index)
        elif args.eval_mode == "generativenosync":
            gptconfig = AutoConfig.from_pretrained(args.gpt_config_path)
            beam_searcher = R2D2GenFastEvaluator(model, gptconfig, device, beam_size=args.beam_size)
            valid_printer = GenerativeNosyncPrinter(modeltype=args.model_type, model=model, beam_searcher=beam_searcher, dataloader=valid_dataloader, tokenizer=tokenizer, device=device, index=args.index)
            test_printer = GenerativeNosyncPrinter(modeltype=args.model_type, model=model, beam_searcher=beam_searcher, dataloader=test_dataloader, tokenizer=tokenizer, device=device, index=args.index)
        elif args.eval_mode == "parser":
            valid_printer = ParserPrinter(modeltype=args.model_type, model=model, dataloader=valid_dataloader, tokenizer=tokenizer, device=device)
            test_printer = ParserPrinter(modeltype=args.model_type, model=model, dataloader=test_dataloader, tokenizer=tokenizer, device=device)
    else:
        valid_printer = None
        test_printer = None
    
    trainer = Trainer(ddpmodel, collator, device=device, tokenizer=tokenizer, logger=logger,
                      is_master=is_master, num_workers=args.pool_size)

    amp_dtype=torch.float16
    if torch.cuda.is_bf16_supported():
        amp_dtype=torch.bfloat16
    
    logger.info(f"start training on {global_rank}")
    trainer.train(dataloader, optimizer, scheduler, scaler,
                  args.output_dir,
                  valid_printer=valid_printer, test_printer=test_printer, 
                  amp_dtype=amp_dtype,
                  coeff_scheduler=coeff_scheduler,
                  temp_scheduler=temp_scheduler,
                  log_steps=args.log_steps, save_steps=args.save_steps,
                  epochs=args.epochs,
                  max_norm=args.max_grad_norm, max_recover_step=max_step,
                  accumulation_steps=args.accumulation_steps)
