# coding=utf-8
# Copyright (c) 2021 Ant Group

import datetime
import random
import torch
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import sys
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import numpy as np
import os
import logging
from model.topdown_parser import LSTMParser
from model.r2d2_cuda import R2D2Cuda
from reader.memory_line_reader import BatchSelfRegressionLineDataset, HugeBatchSelfRegressionLineDataset
import time
from utils.model_loader import get_max_epoch_step, load_checkpoint, load_model
from utils.tree_utils import get_token_tree, get_tree_from_merge_trajectory
import shutil
from torch.nn.parallel import DistributedDataParallel as DDP


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer(object):
    def __init__(self,
                 model,
                 parser,
                 is_master,
                 tokenizer,
                 device,
                 logger,
                 distributed=False,
                 scaler=None,
                 n_gpu=1):
        self.model = model
        self.parser = parser
        self.tokenizer = tokenizer
        self.is_master = is_master
        self.logger = logger

        self.scaler = scaler

        self.distributed = distributed

        self.device = device
        self.n_gpu = n_gpu


    def train(
            self,
            data_loader: DataLoader,
            optimizer,
            scheduler,
            output_dir,
            log_step,
            epochs,
            num_samples=10,
            max_grad_norm=1.0,
            max_recover_epoch=-1,
            max_recover_step=-1,
            save_steps=10000
    ):
        train_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = len(data_loader)
        self.model.train()
        for epoch in train_iterator:
            if epoch < max_recover_epoch:
                continue
            # data_loader.dataset.shuffle()
            if isinstance(data_loader, DataLoader) and isinstance(
                    data_loader.sampler, DistributedSampler):
                data_loader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(data_loader, desc="Iteration")

            for step, inputs in enumerate(epoch_iterator):
                if step <= max_recover_step:
                    continue
                max_recover_step = -1
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                merge_trajectories = self.parser(**inputs)
                with torch.cuda.amp.autocast():
                    results = self.model(**inputs, merge_trajectories=merge_trajectories, sample_trees=num_samples)
                loss = results['loss']

                sampled_trees = results['sampled_trees']
                parser_loss = self.parser(**inputs, split_masks=sampled_trees['split_masks'],
                                          split_points=sampled_trees['split_points'])

                if self.n_gpu > 1:
                    total_loss = (loss + parser_loss).mean()
                else:
                    total_loss = loss + parser_loss

                self.scaler.scale(total_loss).backward()

                try:
                    if max_grad_norm > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                        max_grad_norm, error_if_nonfinite=True)
                        torch.nn.utils.clip_grad_norm_(self.parser.parameters(),
                                                        max_grad_norm, error_if_nonfinite=True)
                    self.scaler.step(optimizer)
                finally:
                    scheduler.step()
                    optimizer.zero_grad()
                    self.scaler.update()

                if step % log_step == 0 and step > 0:
                    with torch.no_grad():
                        self.model.eval()
                        self.parser.eval()
                        merge_trajectories = self.parser(**inputs)
                        results = self.model(**inputs, merge_trajectories=merge_trajectories.clone(),
                                             sample_trees=num_samples, recover_tree=True)
                        sampled_trees = results['sampled_trees']
                        self.model.train()
                        self.parser.train()
                        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                        if self.is_master:
                            seq_len = inputs["attention_mask"][0].sum()
                            merge_trajectories = merge_trajectories.to('cpu').data.numpy()
                            _, tree_str = get_tree_from_merge_trajectory(merge_trajectories[0], seq_len, tokens)
                            self.logger.info(
                                f"parsed tree: {tree_str}"
                            )
                            tables = results['tables']
                            self.logger.info(
                                f"best cky tree: {get_token_tree(tables[0].root.best_node, tokens)}"
                            )
                            seq_len = len(inputs["input_ids"][0])
                            self.logger.info(
                                f"progress:{step}/{total_step} input_len: {seq_len}, loss: {loss.item()}, "
                                f"parser loss: {parser_loss.item()}"
                            )
                if step % save_steps == 0 and step > 0:
                    torch.save(self.model.state_dict(),
                           os.path.join(output_dir, f"model{epoch}_{step}.bin"))
                    torch.save(self.parser.state_dict(), os.path.join(output_dir, f"parser{epoch}_{step}.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer{epoch}_{step}.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler{epoch}_{step}.pt"))
                    if self.scaler is not None:
                        torch.save(self.scaler.state_dict(), os.path.join(output_dir, f'scaler{epoch}_{step}.pt'))
            if self.is_master:
                torch.save(self.model.state_dict(),
                           os.path.join(output_dir, f"model{epoch}.bin"))
                torch.save(self.parser.state_dict(), os.path.join(output_dir, f"parser{epoch}.bin"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler.pt"))
                if self.scaler is not None:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, f'scaler.pt'))

        if self.is_master:
            torch.save(self.model.state_dict(),
                       os.path.join(output_dir, f"model.bin"))
            torch.save(self.parser.state_dict(),
                       os.path.join(output_dir, f"parser.bin"))


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("Arguments to pretrain R2D2")
    cmd.add_argument("--batch_size",
                     default=8,
                     type=int,
                     help="training batch size")
    cmd.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before "
             "performing a backward/update pass.",
    )
    cmd.add_argument("--max_grad_norm",
                     default=1.0,
                     type=float,
                     help="Max gradient norm.")
    cmd.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    cmd.add_argument("--parser_lr", default=1e-2, type=float, help="learning rate")
    cmd.add_argument("--config_path",
                     required=True,
                     type=str,
                     help="bert model config")
    cmd.add_argument("--vocab_dir",
                     required=True,
                     type=str,
                     help="Directory to the vocabulary")
    cmd.add_argument("--input_type",
                     default="txt",
                     type=str,
                     choices=["txt", "ids"])
    cmd.add_argument("--corpus_path",
                     required=True,
                     type=str,
                     help="path to the training corpus")
    cmd.add_argument("--max_batch_len", default=512, type=int)
    cmd.add_argument("--min_len", default=2, type=int)
    cmd.add_argument("--max_line", default=-1, type=int)
    cmd.add_argument("--output_dir", required=True, type=str, help="save dir")
    cmd.add_argument("--seperator", type=str, default=None)
    cmd.add_argument("--local_rank",
                     default=-1,
                     type=int,
                     help="multi gpu training")
    cmd.add_argument("--pretrain_dir", default=None, type=str)
    cmd.add_argument("--epochs", default=10, type=int, help="training epochs")
    cmd.add_argument("--constraints", default='none', choices=['none', 'mat'], type=str)
    cmd.add_argument("--warm_up", type=float, default=0.1)
    cmd.add_argument("--log_step", default=100, type=int)
    cmd.add_argument("--num_samples", default=100, type=int)
    cmd.add_argument("--random_sample", action='store_true', default=False)
    cmd.add_argument("--transformer_parser", action='store_true', default=False)
    cmd.add_argument("--seed", default=404, type=int)
    cmd.add_argument("--huge_mode", default=False, action='store_true')

    args = cmd.parse_args(sys.argv[1:])

    if args.local_rank >= 0:
        torch.cuda.set_device(
            args.local_rank
        )  # for multi-process in a single machine with multiple GPUs.
        global_rank = args.local_rank
        while True:
            try:
                torch.distributed.init_process_group(backend="nccl",
                                                     init_method="env://",
                                                     timeout=datetime.timedelta(seconds=7200))
                if torch.distributed.is_initialized():
                    global_rank = torch.distributed.get_rank()
                    break
            except ValueError:
                time.sleep(5)
            except Exception as e:
                logging.error(e)
                logging.error("Exit with unknown error")
                exit(-1)
        device = torch.device("cuda")
    else:
        global_rank = -1
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")

    is_master = args.local_rank == -1 or global_rank == 0
    config = AutoConfig.from_pretrained(args.config_path)
    print(f"initialize model on {global_rank}")
    set_seed(args.seed)

    model = R2D2Cuda(config)
    if args.transformer_parser:
        parser = TransformerParser(config)
    else:
        parser = LSTMParser(config)

    max_epoch, max_step = get_max_epoch_step(args.output_dir, 'model*_*.bin')
    if args.pretrain_dir is not None:
        if os.path.exists(os.path.join(args.pretrain_dir, f'model.bin')):
            model.from_pretrain(os.path.join(args.pretrain_dir, f'model.bin'))
        else:
            logging.warn('no model.bin in pretrain dir')
        if os.path.exists(os.path.join(args.pretrain_dir, f'parser.bin')):
            load_model(parser, os.path.join(args.pretrain_dir, f'parser.bin'))
        else:
            logging.warn('no parser.bin in pretrain dir')
    elif max_epoch >= 0:
        print(f"load from checkpoint, turn: {max_epoch}_{max_step}")
        model.from_pretrain(os.path.join(args.output_dir, f'model{max_epoch}_{max_step}.bin'))
        load_model(parser, os.path.join(args.output_dir, f'parser{max_epoch}_{max_step}.bin'))

    print(f"move model to gpu:{global_rank}")
    model.to(device)
    parser.to(device)

    print(f"start loading dataset on {global_rank}")
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    data_batch_size = 1

    if args.huge_mode:
        dataset = HugeBatchSelfRegressionLineDataset(
            args.corpus_path,
            tokenizer,
            batch_max_len=args.max_batch_len,
            min_len=args.min_len,
            batch_size=args.batch_size,
            max_line=args.max_line,
            input_type=args.input_type,
            random=args.random_sample,
            seperator=args.seperator
        )
    else:
        dataset = BatchSelfRegressionLineDataset(
            args.corpus_path,
            tokenizer,
            batch_max_len=args.max_batch_len,
            min_len=args.min_len,
            batch_size=args.batch_size,
            max_line=args.max_line,
            input_type=args.input_type,
            random=args.random_sample,
            seperator=args.seperator
        )

    data_batch_size = 1  # dynamic batch_size

    if global_rank == -1:
        if args.random_sample:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=data_batch_size,
                                sampler=sampler,
                                collate_fn=dataset.collate_batch)
        print(f'data total len {len(dataloader)}')
        n_gpu = 1
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = max(10, args.warm_up * t_total)
        optimizer = AdamW([{"params": model.parameters()},
                           {"params": parser.parameters(), "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
    elif global_rank >= 0:
        n_gpu = 1
        print(f"initialize ddp on {global_rank}")
        dataloader = DataLoader(
            dataset,
            batch_size=data_batch_size,
            sampler=DistributedSampler(dataset, shuffle=args.random_sample),
            collate_fn=dataset.collate_batch,
            drop_last=True,
        )
        print(f'data total len {len(dataloader)}')
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = max(10, args.warm_up * t_total)
        model = DDP(model)
        parser = DDP(parser)
        # optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
        optimizer = AdamW([{"params": model.parameters()},
                           {"params": parser.parameters(), "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)


    scaler = torch.cuda.amp.GradScaler()
    if is_master:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            shutil.copyfile(args.config_path,
                            os.path.join(args.output_dir, 'config.json'))
            shutil.copyfile(os.path.join(args.vocab_dir, 'vocab.txt'),
                            os.path.join(args.output_dir, 'vocab.txt'))
        except RuntimeError:
            pass
    if is_master:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.output_dir, "training_log.txt"),
                                 mode="a", encoding="utf-8")
        logger.addHandler(fh)
    else:
        logger = logging

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warm_up_steps,
                                                num_training_steps=t_total)

    if max_epoch >= 0:
        try:
            modules = [optimizer, scheduler, scaler]
            files = [f'optimizer{max_epoch}_{max_step}.pt', f'scheduler{max_epoch}_{max_step}.pt', \
                    f'scaler{max_epoch}_{max_step}.pt']
            load_checkpoint(modules, files, args.output_dir)
        except:
            logging.warning('load optimizer error')
            pass
    
    trainer = Trainer(
        model,
        parser,
        device=device,
        tokenizer=tokenizer,
        logger=logger,
        is_master=is_master,
        n_gpu=n_gpu,
        scaler=scaler,
        distributed=args.local_rank >= 0,
    )

    trainer.train(
        dataloader,
        optimizer,
        scheduler,
        log_step=args.log_step,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        num_samples=args.num_samples,
        max_recover_epoch=max_epoch,
        max_recover_step=max_step
    )
