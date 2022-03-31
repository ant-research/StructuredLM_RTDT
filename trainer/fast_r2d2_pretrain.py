# coding=utf-8
# Copyright (c) 2021 Ant Group

from lib2to3.pgen2 import token
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
from model.topdown_parser import TopdownParser
from model.r2d2_cuda import R2D2Cuda
from reader.memory_line_reader import BatchSelfRegressionLineDataset, TreeBankDataset
import time
from utils.model_loader import get_max_epoch, load_checkpoint, load_model
from utils.tree_utils import get_token_tree, get_tree_from_merge_trajectory
import shutil


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
                 convert_ids_to_tokens,
                 device,
                 logger,
                 apex_enable=False,
                 n_gpu=1):
        self.model = model
        self.parser = parser
        self.is_master = is_master
        self.logger = logger
        self.convert_ids_to_tokens = convert_ids_to_tokens

        self.device = device
        self.n_gpu = n_gpu
        self.apex_enable = apex_enable

    def train(
            self,
            data_loader: DataLoader,
            optimizer,
            parser_optimizer,
            scheduler,
            parser_scheduler,
            output_dir,
            log_step,
            epochs,
            num_samples=10,
            max_grad_norm=1.0,
            max_recover_epoch=-1
    ):
        train_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = len(data_loader)
        self.model.train()
        for epoch in train_iterator:
            if epoch <= max_recover_epoch:
                continue
            data_loader.dataset.shuffle()
            if isinstance(data_loader, DataLoader) and isinstance(
                    data_loader.sampler, DistributedSampler):
                data_loader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(data_loader, desc="Iterat ion")

            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                merge_trajectories = self.parser(**inputs)
                results = self.model(**inputs, merge_trajectories=merge_trajectories, sample_trees=num_samples)
                loss = results['loss']
                if self.n_gpu > 1:
                    loss = loss.mean()

                if self.apex_enable:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                try:
                    if max_grad_norm > 0:
                        if self.apex_enable:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer), max_grad_norm, error_if_nonfinite=True)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           max_grad_norm, error_if_nonfinite=True)

                    optimizer.step()
                    scheduler.step()
                except RuntimeError as e:
                    self.logger.error(e)
                finally:
                    self.model.zero_grad()

                sampled_trees = results['sampled_trees']
                parser_loss = self.parser(**inputs, split_masks=sampled_trees['split_masks'],
                                          split_points=sampled_trees['split_points'])
                if self.apex_enable:
                    with amp.scale_loss(parser_loss, parser_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    parser_loss.backward()

                try:
                    if max_grad_norm > 0:
                        if self.apex_enable:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(parser_optimizer), max_grad_norm, error_if_nonfinite=True)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.parser.parameters(),
                                                           max_grad_norm, error_if_nonfinite=True)
                    parser_optimizer.step()
                    parser_scheduler.step()
                except RuntimeError as e:
                    self.logger.error(e)
                finally:
                    self.parser.zero_grad()

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
                        tokens = self.convert_ids_to_tokens(inputs["input_ids"][0])
                        if self.is_master:
                            # sampled_splits = sampled_trees['split_points'].to('cpu').data.numpy()  # (B, K, L - 1)
                            seq_len = inputs["attention_mask"][0].sum()
                            # for splits in sampled_splits[0]:
                            #     reorged_splits = list(reversed(splits[:seq_len - 1]))
                            #     self.logger.info(
                            #         f"sampled tree: {get_tree_from_merge_trajectory(reorged_splits, seq_len, tokens)}"
                            #     )
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
            if self.is_master:
                torch.save(self.model.state_dict(),
                           os.path.join(output_dir, f"model{epoch}.bin"))
                torch.save(self.parser.state_dict(), os.path.join(output_dir, f"parser{epoch}.bin"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler.pt"))
                torch.save(parser_optimizer.state_dict(), os.path.join(output_dir, f"parser_optimizer.pt"))
                torch.save(parser_scheduler.state_dict(), os.path.join(output_dir, f"parser_scheduler.pt"))
                if self.apex_enable:
                    torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
        if self.is_master:
            torch.save(self.model.state_dict(),
                       os.path.join(output_dir, f"model.bin"))
            torch.save(self.parser.state_dict(),
                       os.path.join(output_dir, f"parser.bin"))
            torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler.pt"))
            torch.save(parser_optimizer.state_dict(), os.path.join(output_dir, f"parser_optimizer.pt"))
            torch.save(parser_scheduler.state_dict(), os.path.join(output_dir, f"parser_scheduler.pt"))
            if self.apex_enable:
                torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))


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
                     choices=["txt", "ids", "treebank"])
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
    cmd.add_argument("--apex_mode", default='O1', type=str)

    args = cmd.parse_args(sys.argv[1:])

    if args.local_rank >= 0:
        torch.cuda.set_device(
            args.local_rank
        )  # for multi-process in a single machine with multiple GPUs.
        global_rank = args.local_rank
        while True:
            try:
                torch.distributed.init_process_group(backend="nccl",
                                                     init_method="env://")
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

    config = AutoConfig.from_pretrained(args.config_path)
    print(f"initialize model on {global_rank}")
    set_seed(404)

    model = R2D2Cuda(config)
    parser = TopdownParser(config)

    max_epoch = get_max_epoch(args.output_dir, 'model*.bin')
    if args.pretrain_dir is not None:
        model.from_pretrain(os.path.join(args.pretrain_dir, f'model.bin'))
        load_model(parser, os.path.join(args.pretrain_dir, f'parser.bin'))
    elif max_epoch >= 0:
        print(f"load from checkpoint, turn: {max_epoch}")
        model.from_pretrain(os.path.join(args.output_dir, f'model{max_epoch}.bin'))
        load_model(parser, os.path.join(args.output_dir, f'parser{max_epoch}.bin'))

    print(f"move model to gpu:{global_rank}")
    model.to(device)
    parser.to(device)

    if args.input_type != "treebank":
        tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
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
    else:
        dataset = TreeBankDataset(
            args.corpus_path, 
            os.path.join(args.vocab_dir, 'vocab.txt'),
            batch_max_len=args.max_batch_len,
            batch_size=args.batch_size,
            random=args.random_sample)

    data_batch_size = 1  # dynamic batch_size

    apex_enable = False
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
        optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
        parser_optimizer = AdamW(parser.parameters(), lr=args.parser_lr, correct_bias=False)
        try:
            from apex import amp
            from apex.parallel import DistributedDataParallel

            print(f"enable apex successful")
            [model, parser], [optimizer, parser_optimizer] = amp.initialize([model, parser],
                                                                            [optimizer, parser_optimizer],
                                                                            opt_level=args.apex_mode)
            apex_enable = True
        except Exception as e:
            logging.error(e)
            pass

    elif global_rank >= 0:
        n_gpu = 1
        print(f"initialize apex on {global_rank}")
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
        optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
        parser_optimizer = AdamW(parser.parameters(), lr=args.parser_lr, correct_bias=False)
        try:
            from apex import amp
            from apex.parallel import DistributedDataParallel

            print(f"enable apex successful on {global_rank}")
            t_total = int(len(dataloader) * args.epochs)
            [model, parser], [optimizer, parser_optimizer] = amp.initialize([model, parser],
                                                                            [optimizer, parser_optimizer],
                                                                            opt_level=args.apex_mode)
            model = DistributedDataParallel(model, delay_allreduce=True)
            parser = DistributedDataParallel(parser, delay_allreduce=True)
            apex_enable = True
        except Exception as e:
            logging.error(e)
            logging.error("import apex failed")
            sys.exit(-1)
    is_master = args.local_rank == -1 or global_rank == 0
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
        fh = logging.FileHandler(os.path.join(args.output_dir,
                                              "training_log.txt"),
                                 mode="a",
                                 encoding="utf-8")
        logger.addHandler(fh)
    else:
        logger = logging

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warm_up_steps,
                                                num_training_steps=t_total)
    parser_scheduler = get_linear_schedule_with_warmup(parser_optimizer,
                                                       num_warmup_steps=warm_up_steps,
                                                       num_training_steps=t_total)

    if max_epoch >= 0:
        modules = [optimizer, parser_optimizer, scheduler, parser_scheduler]
        files = ['optimizer.pt', 'parser_optimizer.pt', 'scheduler.pt', 'parser_scheduler.pt']
        if apex_enable:
            modules.append(amp)
            files.append('amp.pt')
        load_checkpoint(modules, files, args.output_dir)
    trainer = Trainer(
        model,
        parser,
        device=device,
        convert_ids_to_tokens=dataset.convert_ids_to_tokens,
        logger=logger,
        is_master=is_master,
        n_gpu=n_gpu,
        apex_enable=apex_enable,
    )

    trainer.train(
        dataloader,
        optimizer,
        parser_optimizer,
        scheduler,
        parser_scheduler,
        log_step=args.log_step,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        num_samples=args.num_samples,
        max_recover_epoch=max_epoch
    )
