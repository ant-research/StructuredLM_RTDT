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
import deepspeed
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

        for step, inputs in enumerate(epoch_iterator):
            if step <= max_recover_step:
                continue
            max_recover_step = -1

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.cuda.amp.autocast():
                loss, splits = self.model(**inputs, enable_gpt=enable_gpt, coeff=max(0, coeff - step * coeff_decline))

            try:
                model.backward(loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                model.step()
            except RuntimeError as e:
                self.logger.error(e)
            finally:
                pass

            if step % log_steps == 0 and step > 0:
                self.logger.info(f'progress:{step}/{total_step} loss: {loss.item()}')
                with torch.no_grad():
                    # output generated binary tree
                    if self.is_master and splits is not None:
                        # output binary trees for different iteration epochs
                        # sent_id = np.random.randint(inputs['input_ids'].shape[0])
                        sent_id = 0
                        seq_len = inputs["masks"][sent_id].sum()
                        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][sent_id].cpu().data.numpy())
                        self.logger.info(f"input sentence: {input_tokens}")
                        tokens = [gpt_token(t) for t in input_tokens]
                        split_points = [_ for _ in reversed(splits[sent_id, :seq_len - 1].cpu().data.numpy())]
                        merged_tree = get_tree_from_merge_trajectory(split_points, seq_len, tokens)
                        self.logger.info(f"parsed tree : {merged_tree}")
            if step % save_steps == 0 and step > 0:
                model.save_checkpoint(output_dir, f'{step}')

        if self.is_master:
            model.save_checkpoint(output_dir, 'final')

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    cmd.add_argument('--parser_lr', default=1e-3, type=float, help='learning rate')
    cmd.add_argument('--r2d2_config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--gpt_config_path', required=True, type=str, help='config for gpt')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--r2d2_only', action='store_true')
    cmd.add_argument('--accumulation_steps', default=1)
    cmd.add_argument('--model_type', choices=['r2d2-gen', 'gpt', 'llama'], default='r2d2-gen')
    cmd.add_argument('--num_samples', type=int, default=100000)
    cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--checkpoint_dir', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--pretrain_dir', default=None, type=str)
    cmd.add_argument('--coeff', type=float, default=1.0)
    cmd.add_argument('--pool_size', type=int, default=2)
    cmd.add_argument('--seed', type=int, default=404)
    cmd.add_argument('--disable_group', action='store_true')
    cmd.add_argument('--warm_up_steps', type=int, default=1000)
    cmd.add_argument('--log_steps', default=100, type=int)
    cmd.add_argument('--save_steps', default=500, type=int)
    cmd.add_argument('--cache_dir', required=False, default=None, type=str)
    cmd.add_argument('--local_rank', default=-1, type=int)
    cmd = deepspeed.add_config_arguments(cmd)

    args = cmd.parse_args(sys.argv[1:])

    local_rank = args.local_rank
    if local_rank >= 0:
        print(f'local rank: {local_rank}')
        torch.cuda.set_device(local_rank)  # for multi-process in a single machine with multiple GPUs.
        global_rank = local_rank

        deepspeed.init_distributed()
        print(f'init_distributed ok')
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
    
    collator_fn = DefaultCollator(not args.disable_group, pool_size=args.pool_size).generative_r2d2_collate_fn

    parser_params = []
    model_params = []
    for name, params in model.named_parameters():
        if name.find('.parser.') > 0:
            parser_params.append(params)
        else:
            model_params.append(params)


    optimizer = AdamW([{"params": model_params},
                        {"params": parser_params, "lr": args.parser_lr}],
                        lr=args.lr, correct_bias=False)

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_steps,
    #                                             num_training_steps=t_total)
    model, optimizer, dataloader, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer,
                                                    training_data=dataset,
                                                    collate_fn=collator_fn,
                                                    model_parameters=model.parameters())
    
    if max_epoch >= 0:
        modules = [optimizer]
        files = [f'optimizer{max_epoch}_{max_step}.pt']
        load_checkpoint(modules, files, args.checkpoint_dir)
    
    trainer = Trainer(model, device=device, tokenizer=tokenizer, logger=logger,
                      is_master=is_master)

    trainer.train(dataloader,
                  args.output_dir,
                  enable_gpt = not args.r2d2_only,
                  log_steps=args.log_steps, save_steps=args.save_steps,
                  max_recover_step=max_step, coeff=args.coeff)