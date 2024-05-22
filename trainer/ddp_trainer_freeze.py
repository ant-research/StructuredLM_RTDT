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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from model.model_factory import create_model
from reader.lazy_loader import LazyLoader
from reader.dataset import GPT2Dataset
from torch.utils.data.distributed import DistributedSampler
from reader.data_collator import DefaultCollator
from utils.model_loader import get_max_epoch_step, load_checkpoint
from utils.tree_utils import get_tree_from_merge_trajectory
from utils.misc import gpt_token


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


def _scalar(val):
    if isinstance(val, torch.Tensor):
        return val.item()
    return val

class LinearProgressScheduler:
    def __init__(self, start, end, proportion, total_steps):
        # e.g. proportion = 0.8
        # then val will go from start to end at previous 80% steps and keep end in the last 20% steps
        self._start = start
        self._end = end
        self._total_steps = total_steps * proportion
        self._val = start

    def update(self, current_step):
        r = min(1.0, current_step / self._total_steps)
        return self._start * (1 - r) + self._end * r

    @property
    def value(self):
        return self._val


class Trainer(object):
    def __init__(self, 
                 model,
                 freeze_struct_params,
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
        self.freeze_struct_params = freeze_struct_params

        self.device = device
        self.lr = lr

    def train(self, 
              data_loader: DataLoader, 
              optimizer, 
              scheduler, 
              scaler,
              output_dir,
              amp_dtype=torch.float16,
              coeff_scheduler=None,
              temp_scheduler=None,
              log_steps=100, save_steps=100, 
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

            with torch.cuda.amp.autocast(dtype=amp_dtype):
                coeff = 1.0 if coeff_scheduler is None else coeff_scheduler.value
                temperature = 1.0 if temp_scheduler is None else temp_scheduler.value
                result = self.model(**inputs, coeff=coeff, temperature=temperature)

            try:
                scaler.scale(result.non_freeze_loss / accumulation_steps).backward(retain_graph=True)
                self.freeze_struct_params(True)
                scaler.scale(result.freeze_loss / accumulation_steps).backward()
                self.freeze_struct_params(False)
                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            except RuntimeError as e:
                self.logger.error(e)
            finally:
                if (step + 1) % accumulation_steps == 0:
                    optimizer.zero_grad()

            if step % log_steps == 0 and step > 0:
                self.logger.info(f'progress:{step}/{total_step} loss: {_scalar(result.loss)} gpt loss: {_scalar(result.gpt_loss)} ' + \
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
            if step % save_steps == 0 and step > 0:
                try:
                    torch.save(self.model.state_dict(),
                            os.path.join(output_dir, f"model0_{step}.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer0_{step}.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler0_{step}.pt"))
                    
                    if scaler is not None:
                        torch.save(scaler.state_dict(), os.path.join(output_dir, f'scaler0_{step}.pt'))
                except:
                    pass

        if self.is_master:
            while True:
                try:
                    torch.save(self.model.state_dict(), os.path.join(output_dir, f'model.bin'))
                    break
                except:
                    time.sleep(5)

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--batch_size', default=8, type=int, help='training batch size')
    cmd.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    cmd.add_argument('--parser_lr', default=1e-3, type=float, help='learning rate')
    cmd.add_argument('--r2d2_config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--gpt_config_path', required=True, type=str, help='config for gpt')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--ext_vocab_path', required=False, default=None, type=str, help='external vocab path')
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--accumulation_steps', default=1)
    cmd.add_argument('--model_type', choices=['r2d2-gen', 'gpt', 'llama', 'r2d2', 'r2d2-gen-fast', 'r2d2-fast', 'r2d2-gen-fast-struct', 'r2d2-gen-fast-ext'], default='r2d2-gen')
    cmd.add_argument('--num_samples', type=int, default=100000)
    cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--checkpoint_dir', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--pretrain_dir', default=None, type=str)
    cmd.add_argument('--coeff_start', type=float, default=1.0)
    cmd.add_argument('--coeff_end', type=float, default=0)
    cmd.add_argument('--coeff_proportion', type=float, default=0.8)
    cmd.add_argument('--temperature_start', type=float, default=1.0)
    cmd.add_argument('--temperature_end', type=float, default=0.1)
    cmd.add_argument('--temperature_proportion', type=float, default=0.8)
    cmd.add_argument('--pool_size', type=int, default=4)
    cmd.add_argument('--max_seq_len', type=int, default=1024)
    cmd.add_argument('--seed', type=int, default=404)
    cmd.add_argument('--fix_embedding', action='store_true')
    cmd.add_argument('--disable_group', action='store_true')
    cmd.add_argument('--warm_up', type=float, default=0.01)
    cmd.add_argument('--log_steps', default=100, type=int)
    cmd.add_argument('--gradient_checkpoint', action='store_true')
    # cmd.add_argument('--gpt_loss_coeff', type=float, default=1.0)
    cmd.add_argument('--compile', action='store_true')
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

    model = create_model(args.model_type, args.r2d2_config_path, args.gpt_config_path, args.fix_embedding, args.gradient_checkpoint)

    max_epoch = -1
    max_step = -1
    
    if args.pretrain_dir is not None:
        model.from_pretrain(args.pretrain_dir, strict=False)
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
    org_model = model
    def freeze_struct_params(freeze):
        org_model.r2d2.swith_struct_params(freeze)

    # named_par_list = list(model.named_parameters())
    # unused_parser_indices = "248 249"
    # unused_parser_indices = [int(t) for t in unused_parser_indices.split()]
    # for idx in unused_parser_indices:
    #     print(named_par_list[idx][0])

    set_seed(args.seed)

    logger.info(f'start loading dataset on {global_rank}')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    lazy_loader = LazyLoader(args.corpus_path, is_array=True)
    dataset = GPT2Dataset(lazy_loader, num_samples=args.num_samples, max_seq_len=args.max_seq_len)
    print(f'total samples: {args.num_samples}')
    
    collator = DefaultCollator(not args.disable_group, \
                               external_vocab_path=args.ext_vocab_path)
    collator_fn = collator.generative_r2d2_collate_fn_ext

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
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset, shuffle=False),
                                collate_fn=collator_fn, num_workers=args.pool_size)
        t_total = len(dataloader)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW([{"params": model_params},
                           {"params": parser_params, "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
       
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=t_total)
        model = DDP(model)
        
    coeff_scheduler = LinearProgressScheduler(args.coeff_start, args.coeff_end, args.coeff_proportion, t_total)
    temp_scheduler = LinearProgressScheduler(args.temperature_start, args.temperature_end, args.temperature_proportion, t_total)
    scaler = torch.cuda.amp.GradScaler()
    
    if max_epoch >= 0:
        modules = [optimizer, scheduler, scaler]
        files = [f'optimizer{max_epoch}_{max_step}.pt', f'scheduler{max_epoch}_{max_step}.pt', \
                f'scaler{max_epoch}_{max_step}.pt']
        load_checkpoint(modules, files, args.checkpoint_dir)

    # force setting base learning rate
    scheduler.base_lrs = [args.lr, args.parser_lr]
    
    trainer = Trainer(model, freeze_struct_params, collator, device=device, tokenizer=tokenizer, logger=logger,
                      is_master=is_master, num_workers=args.pool_size)

    amp_dtype=torch.float16
    if torch.cuda.is_bf16_supported():
        amp_dtype=torch.bfloat16

    trainer.train(dataloader, optimizer, scheduler, scaler,
                  args.output_dir,
                  amp_dtype=amp_dtype,
                  coeff_scheduler=coeff_scheduler,
                  temp_scheduler=temp_scheduler,
                  log_steps=args.log_steps, save_steps=args.save_steps,
                  max_recover_step=max_step)