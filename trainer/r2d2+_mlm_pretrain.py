# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu


import math
import random
import torch
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import sys
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import numpy as np
import os
import logging
from reader.memory_line_reader import BatchSelfRegressionLineDataset, HugeBatchInsideOutsideNSPDataset, HugeBatchSelfRegressionLineDataset
from reader.data_collator import DefaultCollator, MLMDataCollator
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.model_loader import get_max_epoch_step, load_checkpoint, load_model
from utils.tree_utils import get_tree_from_merge_trajectory
import time


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MyDistributedSampler(DistributedSampler):
    def refresh_total_size(self):
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

class DifferentiableTreeTrainer(object):
    def __init__(self, 
                 model,
                 parser,
                 tokenizer,
                 device,
                 logger,
                 is_master=True,
                 lr=5e-5):
        self.model = model
        self.parser = parser
        self.tokenizer = tokenizer
        self.is_master = is_master
        self.logger = logger

        self.device = device
        self.lr = lr

    def train(self, 
              data_loader: DataLoader, 
              optimizer, 
              scheduler, 
              scaler,
              parser_optimizer,
              parser_scheduler,
              parser_scaler,
              output_dir,
              epochs, log_steps=100, save_steps=100, max_norm=1.0, 
              coeff_decline=0.1,
              max_recover_epoch=-1, max_recover_step=-1):
        train_iterator = trange(
            0, int(epochs), desc="Epoch"
        )
        total_step = len(data_loader)
        for epoch in train_iterator:
            data_loader.dataset.shuffle()
            if epoch < max_recover_epoch:
                continue
            
            if isinstance(data_loader, DataLoader) and isinstance(data_loader.sampler, MyDistributedSampler):
                data_loader.sampler.set_epoch(epoch)
                data_loader.sampler.refresh_total_size()

            epoch_iterator = tqdm(data_loader, desc="Iteration")
            self.model.train()
            self.parser.train()

            for step, (inputs, parser_inputs) in enumerate(epoch_iterator):
                if step <= max_recover_step:
                    continue
                max_recover_step = -1
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device, non_blocking=True)
                for k, v in parser_inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        parser_inputs[k] = v.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        s_indices = parser(**parser_inputs, noise_coeff=1.0 - coeff_decline * epoch)
                    results = self.model(**inputs, merge_trajectory=s_indices, 
                                         recover_tree=True)

                loss = results['loss']
                trees = results['trees']

                total_loss = loss if isinstance(loss, torch.Tensor) else sum(loss)
                
                try:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                except RuntimeError as e:
                    self.logger.error(e)
                finally:
                    optimizer.zero_grad()
                    
                target = trees[-1]
                with torch.cuda.amp.autocast():
                    parser_loss = parser(**parser_inputs, **target)
                try:
                    parser_scaler.scale(parser_loss).backward()
                    parser_scaler.unscale_(parser_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parser.parameters(), max_norm)
                    parser_scaler.step(parser_optimizer)
                    parser_scaler.update()
                    parser_scheduler.step()
                except RuntimeError as e:
                    self.logger.error(f"parser backward error! {e}")
                finally:
                    parser_optimizer.zero_grad()

                if step % log_steps == 0 and step > 0:
                    if isinstance(loss, torch.Tensor):
                        loss_expr = loss.item()
                    else:
                        loss_expr = ','.join([f'{l.item()}' for l in loss])
                    self.logger.info(f'progress:{step}/{total_step} loss: {loss_expr}, parser loss: {parser_loss.item()}')
                    with torch.no_grad():
                        self.model.eval()
                        self.parser.eval()
                        with torch.cuda.amp.autocast():
                            s_indices = parser(**parser_inputs, noise_coeff=0)
                            results = self.model(**inputs, merge_trajectory=s_indices, recover_tree=True)
                        self.model.train()
                        self.parser.train()
                        # output generated binary tree
                        if self.is_master:
                            # output binary trees for different iteration epochs
                            seq_len = inputs["masks"][0].sum()
                            input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().data.numpy())
                            self.logger.info(f"input sentence: {input_tokens}")
                            tokens = self.tokenizer.convert_ids_to_tokens(parser_inputs['input_ids'][0].cpu().data.numpy())
                            # for iter_i, trees_dict in enumerate(results['trees']):
                            trees_dict = results['trees'][-1]
                            split_points = [_ for _ in reversed(
                                trees_dict['split_points'][0, 0, :].cpu().data.numpy()[:seq_len])]
                            merged_tree = get_tree_from_merge_trajectory(split_points, seq_len, tokens)
                            self.logger.info(f"parsed tree : {merged_tree}")
                if step % save_steps == 0 and step > 0:
                    torch.save(self.model.state_dict(),
                           os.path.join(output_dir, f"model{epoch}_{step}.bin"))
                    torch.save(self.parser.state_dict(),
                           os.path.join(output_dir, f"parser{epoch}_{step}.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer{epoch}_{step}.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler{epoch}_{step}.pt"))
                    
                    torch.save(parser_optimizer.state_dict(), os.path.join(output_dir, f"parser_optimizer{epoch}_{step}.pt"))
                    torch.save(parser_scheduler.state_dict(), os.path.join(output_dir, f"parser_scheduler{epoch}_{step}.pt"))
                    if scaler is not None:
                        torch.save(scaler.state_dict(), os.path.join(output_dir, f'scaler{epoch}_{step}.pt'))
                        torch.save(parser_scaler.state_dict(), os.path.join(output_dir, f'parser_scaler{epoch}_{step}.pt'))
            max_recover_step = -1
            if self.is_master:
                torch.save(self.model.state_dict(),
                           os.path.join(output_dir, f"model{epoch}_{step}.bin"))
                torch.save(self.parser.state_dict(),
                           os.path.join(output_dir, f"parser{epoch}_{step}.bin"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer{epoch}_{step}.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler{epoch}_{step}.pt"))
                torch.save(parser_optimizer.state_dict(), os.path.join(output_dir, f"parser_optimizer{epoch}_{step}.pt"))
                torch.save(parser_scheduler.state_dict(), os.path.join(output_dir, f"parser_scheduler{epoch}_{step}.pt"))
                if scaler is not None:
                    torch.save(scaler.state_dict(), os.path.join(output_dir, f'scaler{epoch}_{step}.pt'))
                    torch.save(parser_scaler.state_dict(), os.path.join(output_dir, f'parser_scaler{epoch}_{step}.pt'))
        if self.is_master:
            torch.save(self.model.state_dict(), os.path.join(output_dir, f'model.bin'))
            torch.save(self.parser.state_dict(), os.path.join(output_dir, f'parser.bin'))

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--batch_size', default=8, type=int, help='training batch size')
    cmd.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    cmd.add_argument('--parser_lr', default=1e-3, type=float, help='learning rate')
    cmd.add_argument('--config_path', required=True, type=str,
                     help='bert model config')
    cmd.add_argument('--vocab_path', required=True, type=str,
                     help='vocab path')
    cmd.add_argument('--input_type', default='txt', type=str, choices=['txt', 'ids', 'bin'])
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--max_batch_len', default=512, type=int)
    cmd.add_argument('--min_len', default=2, type=int)
    cmd.add_argument('--max_len', default=999, type=int)
    cmd.add_argument('--max_line', default=-1, type=int)
    cmd.add_argument('--model_type', default='cio', type=str, choices=['cio', 'io'])  # cio short for contextualized inside outside, io short for inside outside
    cmd.add_argument('--backend', default='cuda', type=str, choices=['cuda', 'py'])
    cmd.add_argument('--noshare', default=False, action='store_true', 
                     help='whether to share parameters for inside and outside')
    cmd.add_argument('--ascending', default=False, action='store_true')
    cmd.add_argument('--nsp', action='store_true', default=False)
    cmd.add_argument('--coeff_decline', default=0.1, type=float)
    cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--checkpoint_dir', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--pretrain_dir', default=None, type=str)
    cmd.add_argument('--seed', type=int, default=404)
    cmd.add_argument('--epochs', default=10, type=int, help='training epochs')
    cmd.add_argument('--warm_up', type=float, default=0.01)
    cmd.add_argument('--log_steps', default=100, type=int)
    cmd.add_argument('--save_steps', default=500, type=int)
    cmd.add_argument('--cache_dir', required=False, default=None, type=str)

    args = cmd.parse_args(sys.argv[1:])

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

    config = AutoConfig.from_pretrained(args.config_path)
    logger.info(f'initialize model on {global_rank}')

    if args.model_type == 'cio':
        logger.info('model type: cio')
        # from model.fast_r2d2_iter_attn_share import FastR2D2Plus
        # from model.fast_r2d2_attn_share import FastR2D2Plus
        if args.noshare:
            from model.fast_r2d2_iter_attn_mlp import FastR2D2Plus
        else:
            from model.fast_r2d2_iter_attn_share_mlp import FastR2D2Plus
        model = FastR2D2Plus(config)
    elif args.model_type == 'io':
        logger.info('model type: io')
        if args.backend == 'cuda':
            from model.fast_r2d2_insideoutside import FastR2D2Plus
        elif args.backend == 'py':
            from model.fast_r2d2_io_span_attn import FastR2D2Plus
        model = FastR2D2Plus(config)
    from model.topdown_parser import LSTMParser
    parser = LSTMParser(config)

    max_epoch = -1
    max_step = -1
    
    if args.pretrain_dir is not None:
        model.from_pretrain(os.path.join(args.pretrain_dir, f'model.bin'))
        load_model(parser, os.path.join(args.pretrain_dir, f'parser.bin'))
        logger.info("load from pretrain dir successfully")
    elif args.checkpoint_dir is not None:
        max_epoch, max_step = get_max_epoch_step(args.output_dir, 'model*_*.bin')
        print(f'detect max_epoch: {max_epoch}, max_step:{max_step}')
        if max_epoch >= 0:
            logger.info(f'load from checkpoint, turn: {max_epoch}_{max_step}')
            model.from_pretrain(os.path.join(args.output_dir, f'model{max_epoch}_{max_step}.bin'))
            # TODO: add loading from checkpoint for the parser
            load_model(parser, os.path.join(args.output_dir, f'parser{max_epoch}_{max_step}.bin'))
    
    logger.info(f'move model to gpu:{global_rank}')
    parser.to(device)
    model.to(device)
    set_seed(args.seed)

    logger.info(f'start loading dataset on {global_rank}')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_path, config=config)

    if args.input_type in ['txt', 'ids']:
        if args.nsp:
            raise Exception('NSP is not supported on input types : txt, ids')
        dataset = BatchSelfRegressionLineDataset(args.corpus_path, tokenizer, config=config,
                    batch_max_len=args.max_batch_len, min_len=args.min_len,
                    max_len=args.max_len, batch_size=args.batch_size, 
                    max_line=args.max_line, input_type=args.input_type,
                    cache_dir=args.cache_dir, descending=not args.ascending)
    else:
        if args.nsp:
            dataset = HugeBatchInsideOutsideNSPDataset(args.corpus_path, tokenizer, config=config,
                        batch_max_len=args.max_batch_len, min_len=args.min_len,
                        max_len=args.max_len, batch_size=args.batch_size, 
                        max_line=args.max_line, cache_dir=args.cache_dir,
                        descending=not args.ascending)
        else:
            dataset = HugeBatchSelfRegressionLineDataset(args.corpus_path, tokenizer, config=config,
                        batch_max_len=args.max_batch_len, min_len=args.min_len,
                        max_len=args.max_len, batch_size=args.batch_size, 
                        max_line=args.max_line, cache_dir=args.cache_dir,
                        descending=not args.ascending)
    
    if args.model_type == 'cio':
        if args.input_type == 'bin':
            if args.nsp:
                collator_fn = MLMDataCollator(tokenizer).fastr2d2_cio_nsp_mlm_collator
            else:
                collator_fn = MLMDataCollator(tokenizer).fastr2d2_cio_mlm_collator
        else:
            collator_fn = MLMDataCollator(tokenizer).fastr2d2_cio_mlm_atom_span_collator
    elif args.model_type == 'io':
        if args.nsp:
            print('fast r2d2 nsp mlm')
            collator_fn = DefaultCollator(tokenizer, False).default_fastr2d2_nsp_mlm_collator
        else:
            print('fast r2d2 mlm')
            collator_fn = DefaultCollator(tokenizer, False).default_fastr2d2_mlm_collator

    if global_rank == -1:
        dataloader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                                collate_fn=collator_fn)
        n_gpu = 1
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=t_total)
        parser_optimizer = AdamW(parser.parameters(), lr=args.lr,
                                 correct_bias=False)
        parser_scheduler = get_linear_schedule_with_warmup(parser_optimizer, num_warmup_steps=warm_up_steps,
                                                           num_training_steps=t_total)
    elif global_rank >= 0:
        n_gpu = 1
        dataloader = DataLoader(dataset, batch_size=1, sampler=MyDistributedSampler(dataset, shuffle=False),
                                collate_fn=collator_fn, drop_last=True)
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          correct_bias=False)
        parser_optimizer = AdamW(parser.parameters(), lr=args.parser_lr,
                                 correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=t_total)
        parser_scheduler = get_linear_schedule_with_warmup(parser_optimizer, num_warmup_steps=warm_up_steps,
                                                           num_training_steps=t_total)
        model = DDP(model, find_unused_parameters=True)
        parser = DDP(parser)
    
    scaler = torch.cuda.amp.GradScaler()
    parser_scaler = torch.cuda.amp.GradScaler()
    
    if max_epoch >= 0:
        modules = [optimizer, scheduler, scaler, parser_optimizer, parser_scheduler, parser_scaler]
        files = [f'optimizer{max_epoch}_{max_step}.pt', f'scheduler{max_epoch}_{max_step}.pt', \
                f'scaler{max_epoch}_{max_step}.pt', f'parser_optimizer{max_epoch}_{max_step}.pt',
                f'parser_scheduler{max_epoch}_{max_step}.pt', f'parser_scaler{max_epoch}_{max_step}.pt']
        load_checkpoint(modules, files, args.checkpoint_dir)
    
    trainer = DifferentiableTreeTrainer(model, parser, device=device, tokenizer=tokenizer, logger=logger,
                                        is_master=is_master)

    trainer.train(dataloader, optimizer, scheduler, scaler, 
                  parser_optimizer, parser_scheduler, parser_scaler,
                  args.output_dir, args.epochs,
                  log_steps=args.log_steps, save_steps=args.save_steps,
                  max_recover_epoch=max_epoch, max_recover_step=max_step,
                  coeff_decline=args.coeff_decline)