# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
import argparse
import logging
import os
import random
import sys
import time

import evaluate
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (AdamW, AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from eval.xsum_evaluator import XSumEvaluator
from model.model_factory import create_model, xsum_create_model
from model.weighted_sum_func import WeightedSumFunc
from reader.data_collator import DefaultCollator, XSumCollator
from reader.dataset_xsum import XSumDataset
from utils.generator_factory import create_generator
from utils.misc import gpt_token
from utils.model_loader import get_max_epoch_step, load_checkpoint
from utils.tree_utils import get_tree_from_merge_trajectory
from utils.vocab_builder import load_dict, load_span_tokenizer


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
              eval_data_loader: DataLoader, 
              optimizer, 
              scheduler, 
              scaler,
              evaluator,
              output_dir,
              amp_dtype=torch.float16,
              coeff_scheduler=None,
              temp_scheduler=None,
              log_steps=100, save_steps=100, epochs=1, 
              max_norm=1.0, max_recover_step=-1,
              accumulation_steps=1):
        
        train_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = len(data_loader) * epochs
        best_eval_score = {}
        self.model.train()

        for epoch in train_iterator:
            epoch_iterator = tqdm(data_loader, desc="Iteration")
            for step, inputs in enumerate(epoch_iterator):
                # break
                # if step == 0:
                #     continue
                # if step >= 10:
                #     break
                curr_step = step + epoch * len(data_loader)
                if curr_step <= max_recover_step:
                    continue
                max_recover_step = -1

                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                # if step % 50 == 0:
                #     print("input: ", inputs)

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
                if curr_step % save_steps == 0 and curr_step > 0:
                    try: 
                        torch.save(self.model.state_dict(),
                                os.path.join(output_dir, f"model{epoch}_{step}.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer{epoch}_{step}.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler{epoch}_{step}.pt"))

                        if scaler is not None:
                            torch.save(scaler.state_dict(), os.path.join(output_dir, f'scaler{epoch}_{step}.pt'))
                    except:
                        pass
            
            if self.is_master and epoch >= args.epochs - 5:
                while True:
                    try:
                        torch.save(self.model.state_dict(), os.path.join(output_dir, f'model{epoch}.bin'))
                        break
                    except:
                        time.sleep(5)

            # if self.is_master and evaluator is not None and epoch >= args.epochs - 5:
            #     self.model.eval()
            #     score = evaluator.eval(eval_data_loader)
            #     self.model.train()
            #     self.logger.info(f"epoch{epoch}, eval metric: {score}")
            #     for item in score.keys():
            #         if item not in best_eval_score.keys() or score[item] > best_eval_score[item]:
            #             best_eval_score[item] = score[item]
            #             # torch.save(self.model.state_dict(),
            #             #     os.path.join(output_dir, f"curr_{item}_bestmodel_{score[item]}.bin"))
            #             # torch.save(optimizer.state_dict(), os.path.join(output_dir, f"curr_{item}_bestoptimizer_{score[item]}.pt"))
            #             # torch.save(scheduler.state_dict(), os.path.join(output_dir, f"curr_{item}_bestscheduler_{score[item]}.pt"))
            #             # if scaler is not None:
            #             #     torch.save(scaler.state_dict(), os.path.join(output_dir, f'curr_{item}_bestscaler_{score[item]}.pt'))
        self.logger.info(f"best score{best_eval_score}")

if __name__ == '__main__':
    cmd = argparse.ArgumentParser("Arguments for summary trainer")
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--batch_size', default=8, type=int, help='training batch size')
    cmd.add_argument('--eval_batch_size', default=80, type=int, help='evaluating batch size')
    cmd.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    cmd.add_argument('--parser_lr', default=1e-3, type=float, help='learning rate')
    cmd.add_argument('--r2d2_config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--gpt_config_path', required=True, type=str, help='config for gpt')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--ext_vocab_path', required=False, default=None, type=str, help='external vocab path')
    cmd.add_argument('--summary_dir', required=True, type=str, help="directory of the summary task data")
    cmd.add_argument('--accumulation_steps', type=int, default=1)
    cmd.add_argument('--model_type', choices=['r2d2-gen', 'gpt', 'llama', 'r2d2', 'r2d2-gen-fast', 'r2d2-fast', 'r2d2-gen-fast-struct', 'r2d2-gen-fast-ext'], default='r2d2-gen')
    cmd.add_argument('--word_sync', action='store_true')
    cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--checkpoint_dir', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--pretrain_dir', default=None, type=str)
    cmd.add_argument('--coeff_start', type=float, default=1.0)
    cmd.add_argument('--coeff_end', type=float, default=0.0)
    cmd.add_argument('--coeff_proportion', type=float, default=0.5)
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
    cmd.add_argument('--save_steps', default=2000, type=int)
    cmd.add_argument('--cache_dir', required=False, default=None, type=str)
    cmd.add_argument('--eval_perepoch', action='store_true')
    cmd.add_argument('--epochs', default=2, type=int)
    cmd.add_argument('--beam_size', default=2, type=int)
    cmd.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm")
    cmd.add_argument('--document_threshold', default=900, type=int, help="truncate length of document")
    cmd.add_argument('--summary_threshold', default=100, type=int, help="truncate length of document and summary")

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

    model = xsum_create_model(args.model_type, args.r2d2_config_path, args.gpt_config_path, args.fix_embedding, args.gradient_checkpoint)

    max_epoch = -1
    max_step = -1
    
    if args.pretrain_dir is not None:
        model.from_pretrain(args.pretrain_dir, strict=True)
        logger.info("load from pretrain dir successfully")
    if args.checkpoint_dir is not None:
        max_epoch, max_step = get_max_epoch_step(args.output_dir, 'model*_*.bin')
        print(f'detect max_epoch: {max_epoch}, max_step:{max_step}')
        if max_epoch >= 0:
            logger.info(f'load from checkpoint, turn: {max_epoch}_{max_step}')
            model.from_pretrain(os.path.join(args.output_dir, f'model{max_epoch}_{max_step}.bin'))
            # TODO: add loading from checkpoint for the parser
    
    logger.info(f'move model to gpu:{global_rank}')
    model.to(device=device)

    # named_par_list = list(model.named_parameters())
    # unused_parser_indices = "118"
    # unused_parser_indices = [int(t) for t in unused_parser_indices.split()]
    # for idx in unused_parser_indices:
    #     logger.info(f"unused_parameter{named_par_list[idx][0]}")
    #     # print(named_par_list[idx][0])

    set_seed(args.seed)
    
    logger.info(f'start loading dataset on {global_rank}')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    gpt_config = AutoConfig.from_pretrained(args.gpt_config_path)
    eos_id = gpt_config.eos_token_id

    dataset = XSumDataset(data_dir=args.summary_dir, mode="train", tokenizer=tokenizer, eos_id=eos_id, document_threshold=args.document_threshold, summary_threshold=args.summary_threshold)
    logger.info(f'finished loading dataset on {global_rank}')
    
    collator = XSumCollator("train", not args.disable_group, \
                               external_vocab_path=args.ext_vocab_path, padding=-1)
    collator_fn = collator.generative_r2d2_xsum_collate_fn

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
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset, shuffle=True),
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

    if is_master and args.eval_perepoch:
        eval_dataloader = None
        evaluator = None
        eval_dataset = XSumDataset(data_dir=args.summary_dir, mode="test", tokenizer=tokenizer, eos_id=eos_id, document_threshold=args.document_threshold, summary_threshold=args.summary_threshold)
        eval_collator_fn = XSumCollator("test", not args.disable_group, \
                               external_vocab_path=args.ext_vocab_path, padding=-1).generative_r2d2_xsum_collate_fn
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, sampler=SequentialSampler(eval_dataset),
                                collate_fn=eval_collator_fn, num_workers=0)
        metric = evaluate.load('rouge')
        ext_vocab = None
        span_tokenizer = None
        if args.ext_vocab_path:
            ext_vocab = load_dict(args.ext_vocab_path)
            span_tokenizer = load_span_tokenizer(args.ext_vocab_path)
        eval_model = model.model
        generator = create_generator(args.model_type, eval_model, device, gpt_config, beam_size=args.beam_size, sampling=True, word_sync=args.word_sync)
        evaluator = XSumEvaluator(args.model_type, metric, generator, tokenizer, device, word_sync=args.word_sync)
    else:
        eval_dataloader = None
        evaluator = None

    trainer = Trainer(ddpmodel, collator, device=device, tokenizer=tokenizer, logger=logger,
                      is_master=is_master, num_workers=args.pool_size)
    
    amp_dtype=torch.float16
    if torch.cuda.is_bf16_supported():
        amp_dtype=torch.bfloat16

    logger.info(f"start training on {global_rank}")
    trainer.train(dataloader, eval_dataloader, optimizer, scheduler, scaler, evaluator, 
                  args.output_dir, 
                  amp_dtype=amp_dtype,
                  coeff_scheduler=coeff_scheduler, 
                  temp_scheduler=temp_scheduler,
                  log_steps=args.log_steps, save_steps=args.save_steps, 
                  epochs=args.epochs, 
                  max_norm=args.max_grad_norm, max_recover_step=max_step, 
                  accumulation_steps=args.accumulation_steps)