import random
import torch
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import sys
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
import os
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from model.model_factory import create_model
from reader.lazy_loader import LazyLoader
from reader.dataset import GPT2Dataset
from reader.data_collator import DefaultCollator
from utils.model_loader import get_max_epoch_step, load_checkpoint
from utils.tree_utils import get_tree_from_merge_trajectory
from utils.misc import gpt_token


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer(object):
    def __init__(self, 
                 model,
                 tokenizer,
                 device,
                 logger,
                 is_master=True,
                 lr=5e-5):
        self.model = model
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
              output_dir,
              enable_gpt=True,
              log_steps=100, save_steps=100, 
              coeff=1.0,
              coeff_decline=0.0001,
              max_norm=1.0, max_recover_step=-1,
              accumulation_steps=1):

        total_step = len(data_loader)

        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.train()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
            # schedule=torch.profiler.schedule(
            #     wait=2,
            #     warmup=2,
            #     active=5),
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./result'),
            record_shapes=True,
            profile_memory=True
        ) as p:
            for step, inputs in enumerate(epoch_iterator):
                if step <= max_recover_step:
                    continue
                max_recover_step = -1

                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)

                with torch.cuda.amp.autocast():
                    result = self.model(**inputs, coeff=max(0, coeff - step * coeff_decline))

                try:
                    scaler.scale(result.loss / accumulation_steps).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    if (step + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                except RuntimeError as e:
                    self.logger.error(e)
                finally:
                    if (step + 1) % accumulation_steps == 0:
                        optimizer.zero_grad()


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--batch_size', default=8, type=int, help='training batch size')
    cmd.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    cmd.add_argument('--parser_lr', default=1e-3, type=float, help='learning rate')
    cmd.add_argument('--r2d2_config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--gpt_config_path', required=True, type=str, help='config for gpt')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--r2d2_only', action='store_true')
    cmd.add_argument('--accumulation_steps', default=1)
    cmd.add_argument('--model_type', choices=['r2d2-gen', 'gpt', 'llama', 'r2d2-gen-fast'], default='r2d2-gen')
    cmd.add_argument('--num_samples', type=int, default=100000)
    cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--checkpoint_dir', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--pretrain_dir', default=None, type=str)
    cmd.add_argument('--coeff', type=float, default=1.0)
    cmd.add_argument('--pool_size', type=int, default=2)
    cmd.add_argument('--seed', type=int, default=404)
    cmd.add_argument('--disable_group', action='store_true')
    cmd.add_argument('--warm_up', type=float, default=0.01)
    cmd.add_argument('--log_steps', default=100, type=int)
    cmd.add_argument('--save_steps', default=500, type=int)
    cmd.add_argument('--cache_dir', required=False, default=None, type=str)

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

    model = create_model(args.model_type, args.r2d2_config_path, args.gpt_config_path)

    max_epoch = -1
    max_step = -1
    
    if args.pretrain_dir is not None:
        model.from_pretrain(args.pretrain_dir)
        logger.info("load from pretrain dir successfully")
    elif args.checkpoint_dir is not None:
        max_epoch, max_step = get_max_epoch_step(args.output_dir, 'model*_*.bin')
        print(f'detect max_epoch: {max_epoch}, max_step:{max_step}')
        if max_epoch >= 0:
            logger.info(f'load from checkpoint, turn: {max_epoch}_{max_step}')
            model.from_pretrain(os.path.join(args.output_dir, f'model{max_epoch}_{max_step}.bin'))
            # TODO: add loading from checkpoint for the parser
    
    logger.info(f'move model to gpu:{global_rank}')
    model.to(device)
    set_seed(args.seed)

    logger.info(f'start loading dataset on {global_rank}')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    lazy_loader = LazyLoader(args.corpus_path, is_array=True)
    dataset = GPT2Dataset(lazy_loader, num_samples=args.num_samples)
    
    collator_fn = DefaultCollator(not args.disable_group).generative_r2d2_collate_fn

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
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=t_total)
    elif global_rank >= 0:
        n_gpu = 1
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset),
                                collate_fn=collator_fn, num_workers=args.pool_size)
        t_total = len(dataloader)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW([{"params": model_params},
                           {"params": parser_params, "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
       
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=t_total)
        model = DDP(model, find_unused_parameters=True)
    
    scaler = torch.cuda.amp.GradScaler()
    
    if max_epoch >= 0:
        modules = [optimizer, scheduler, scaler]
        files = [f'optimizer{max_epoch}_{max_step}.pt', f'scheduler{max_epoch}_{max_step}.pt', \
                f'scaler{max_epoch}_{max_step}.pt']
        load_checkpoint(modules, files, args.checkpoint_dir)
    
    trainer = Trainer(model, device=device, tokenizer=tokenizer, logger=logger,
                      is_master=is_master)

    trainer.train(dataloader, optimizer, scheduler, scaler,
                  args.output_dir,
                  log_steps=args.log_steps, save_steps=args.save_steps,
                  max_recover_step=max_step, coeff=args.coeff)