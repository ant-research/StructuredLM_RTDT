# coding=utf-8
# Copyright (c) 2021 Ant Group

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
from reader.memory_line_reader import BatchSelfRegressionLineDataset
import time
from utils.statistic_tools import count_context_information


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DifferentiableTreeTrainer(object):
    def __init__(self, model,
                 is_master,
                 tokenizer,
                 device,
                 logger,
                 lr=5e-5,
                 apex_enable=False,
                 n_gpu=1,
                 do_statistic=False):
        self.model = model
        self.tokenizer = tokenizer
        self.is_master = is_master
        self.logger = logger

        self.device = device
        self.lr = lr
        self.n_gpu = n_gpu
        self.apex_enable = apex_enable
        self.do_statistic = do_statistic

    def train(self, data_loader: DataLoader, optimizer, scheduler, output_dir,
              log_step, save_step, epochs):
        train_iterator = trange(
            0, int(epochs), desc="Epoch"
        )
        total_step = len(data_loader)
        for epoch in train_iterator:
            sentence_processed = 0
            full_rate_sum = 0
            coverage_sum = 0
            if isinstance(data_loader, DataLoader) and isinstance(data_loader.sampler, DistributedSampler):
                data_loader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(data_loader, desc="Iteration")

            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                loss, tables = self.model(**inputs)
                if self.n_gpu > 1:
                    loss = loss.mean()
                if self.apex_enable:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                if step % log_step == 0 and step > 0:
                    with torch.no_grad():
                        if self.do_statistic:
                            for t in tables:
                                full_rate, coverage_rate = count_context_information(t)
                                full_rate_sum += full_rate
                                coverage_sum += coverage_rate
                                sentence_processed += 1
                            self.logger.info(f'coverage: {coverage_sum / sentence_processed}, '
                                             f'full_rate: {full_rate_sum / sentence_processed}')
                        self.model.eval()
                        loss, tables = self.model(**inputs)
                        self.model.train()
                        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().data.numpy())
                        if self.is_master:
                            self.logger.info(f'tree: {tables[0].get_token_tree(tokens)}')
                            self.logger.info(f'merge log: {tables[0].get_merge_log(tokens)}')
                            self.logger.info(f'progress:{step}/{total_step} loss: {loss.item()}')
            if self.is_master:
                torch.save(self.model.state_dict(), os.path.join(output_dir, f'model{epoch}.bin'))
        if self.is_master:
            torch.save(self.model.state_dict(), os.path.join(output_dir, f'model.bin'))


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--batch_size', default=8, type=int, help='training batch size')
    cmd.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    cmd.add_argument('--config_path', required=True, type=str,
                     help='bert model config')
    cmd.add_argument('--vocab_path', required=True, type=str,
                     help='vocab path')
    cmd.add_argument('--input_type', default='txt', type=str, choices=['txt', 'ids'])
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--max_batch_len', default=512, type=int)
    cmd.add_argument('--min_len', default=2, type=int)
    cmd.add_argument('--max_line', default=-1, type=int)
    cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--local_rank', default=-1, type=int, help='multi gpu training')
    cmd.add_argument('--epochs', default=10, type=int, help='training epochs')
    cmd.add_argument('--model_path', type=str, required=False, default=None)
    cmd.add_argument('--warm_up', type=float, default=0.01)
    cmd.add_argument('--tree_lstm', default=False, action='store_true')
    cmd.add_argument('--do_statistic', default=False, action='store_true')
    cmd.add_argument('--log_step', default=100, type=int)
    cmd.add_argument('--save_step', default=500, type=int)

    args = cmd.parse_args(sys.argv[1:])

    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)  # for multi-process in a single machine with multiple GPUs.
        global_rank = args.local_rank
        while True:
            try:
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

    config = AutoConfig.from_pretrained(args.config_path)
    logging.info(f'initialize model on {global_rank}')
    if args.tree_lstm:
        from model.r2d2_lstm import R2D2TreeLSTM
        model = R2D2TreeLSTM(config)
    else:
        from model.r2d2 import R2D2
        model = R2D2(config)

    if args.model_path is not None:
        model.from_pretrain(args.model_path)
    logging.info(f'move model to gpu:{global_rank}')
    model.to(device)
    set_seed(404)

    logging.info(f'start loading dataset on {global_rank}')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_path, config=config)
    dataset = BatchSelfRegressionLineDataset(args.corpus_path, tokenizer, config=config,
                                             batch_max_len=args.max_batch_len, min_len=args.min_len,
                                             batch_size=args.batch_size, max_line=args.max_line,
                                             input_type=args.input_type)

    apex_enable = False
    if global_rank == -1:
        dataloader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                                collate_fn=dataset.collate_batch)
        n_gpu = 1
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=t_total)
    elif global_rank >= 0:
        n_gpu = 1
        logging.info(f'initialize apex on {global_rank}')
        dataloader = DataLoader(dataset, batch_size=1, sampler=DistributedSampler(dataset, shuffle=False),
                                collate_fn=dataset.collate_batch, drop_last=True)
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=t_total)
        try:
            from apex import amp
            from apex.parallel import DistributedDataParallel
            logging.info(f'enable apex successful on {global_rank}')
            t_total = int(len(dataloader) * args.epochs)
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            model = DistributedDataParallel(model, delay_allreduce=True)
            apex_enable = True
        except:
            logging.error('import apex failed')
            sys.exit(-1)
    is_master = args.local_rank == -1 or global_rank == 0
    if not os.path.exists(args.output_dir) and is_master:
        os.mkdir(args.output_dir)
    if is_master:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.output_dir, 'training_log.txt'), mode='a', encoding="utf-8")
        logger.addHandler(fh)
    else:
        logger = logging
    trainer = DifferentiableTreeTrainer(model, device=device, tokenizer=tokenizer, logger=logger,
                                        is_master=is_master, n_gpu=n_gpu, apex_enable=apex_enable,
                                        do_statistic=args.do_statistic)

    trainer.train(dataloader, optimizer, scheduler, log_step=args.log_step, save_step=args.save_step,
                  output_dir=args.output_dir, epochs=args.epochs)