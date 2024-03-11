# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Qingyang Zhu

import math
import argparse
from email.policy import strict
import logging
import os
import shutil
import time
import sys

from reader.data_collator import GlueCollator
# from sklearn import metrics
sys.path.append(os.getcwd())
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from trainer.model_factory import create_classification_model
import evaluate
from reader.reader_factory import create_glue_dataset
import glob
from utils.tree_utils import get_tree_from_merge_trajectory

import torch.nn.functional as F

from trainer.fast_r2d2_pretrain import set_seed
from utils.model_loader import get_max_epoch

TASK_MAPPING = {
    'sst-2': 'sst2',
    'mnli-mm': 'mnli_mismatched',
    'mnli': 'mnli_matched',
    'cola': 'cola',
    'qqp': 'qqp',
    'mrpc': 'mrpc',
    'wnli': 'wnli',
    'qnli': 'qnli',
    'rte': 'rte'
}

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

def get_max_epoch_model(output_dir):
    fn_dir_list = glob.glob(os.path.join(output_dir, "model*bin"))
    if not fn_dir_list:
        return None

    def get_epoch_num(fn):
        if "/" in fn:
            fn = fn.rsplit("/", 1)[1]
        if fn.replace("model", "").replace(".bin", "") == '':
            epoch = 0
        else:epoch = int(fn.replace("model", "").replace(".bin", ""))
        return epoch

    epoch_set = set([get_epoch_num(fn) for fn in fn_dir_list])
    if epoch_set:
        return max(epoch_set)
    else:
        return None


class GlueTrainer:
    def __init__(self, model, is_master, tokenizer, device, logger, lr=5e-5, apex_enable=False, n_gpu=1):
        self.model = model
        self.tokenizer = tokenizer
        self.is_master = is_master
        self.logger = logger

        self.device = device
        self.lr = lr
        self.n_gpu = n_gpu
        self.apex_enable = apex_enable

    def train(
            self,
            data_loader: DataLoader,
            optimizer,
            scheduler,
            scaler,
            output_dir,
            log_step,
            eval_step,
            epochs,
            evaluator,
            eval_dataloader,
            recover_epoch=None,
            update_epoch=None, # callback
            max_grad_norm=1.0,
            metric=None
    ):

        train_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = len(data_loader)
        # training_loss, training_step = 0, 0
        acc = 0.
        max_acc = 0.
        self.model.train()
        best_eval_acc = 0
        for epoch in train_iterator:
            if update_epoch is not None:
                update_epoch(epoch)
            data_loader.dataset.shuffle()
            if recover_epoch and epoch <= recover_epoch:
                logger.info(f"Skip epoch {epoch}")
                if epoch == recover_epoch:
                    if self.is_master and eval_dataloader is not None:
                        self.model.eval()
                        eval_result = self.eval(eval_dataloader, metric)
                        self.logger.info(f"epoch{epoch}, eval acc: {eval_result}")
                        self.model.train()
                continue

            if isinstance(data_loader, DataLoader) and isinstance(data_loader.sampler, MyDistributedSampler):
                self.logger.info(f"Set sampler epoch: {epoch}")
                data_loader.sampler.set_epoch(epoch)
                data_loader.sampler.refresh_total_size()

            epoch_iterator = tqdm(data_loader, desc="Train Iteration")
            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.cuda.amp.autocast():
                    results = self.model(**inputs)

                result_loss = results['loss']
                loss = sum(result_loss) if isinstance(result_loss, list) else result_loss

                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                except RuntimeError as e:
                    self.logger.error(e)
                finally:
                    """
                    NOTE:
                    In PyTorch 1.1.0 and later, you should call them in the opposite order: 
                    `optimizer.step()` before `lr_scheduler.step()`.  
                    Failure to do this will result in PyTorch skipping the first value of the learning rate schedule.
                    See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
                    """
                    optimizer.zero_grad()

                if step % log_step == 0 and step > 0:
                    with torch.no_grad():
                        self.model.eval()
                        labels = inputs['labels']
                        results = self.model(**inputs)
                        probs = results['predict']
                        total = 0
                        hits = 0
                        if isinstance(probs, torch.Tensor):
                            hits = (probs.argmax(dim=1) == labels).sum()
                            total = labels.shape[0]
                        elif isinstance(probs, list):
                            assert len(probs) == len(labels)
                            total = len(labels)
                            for pred_label_set, gold_labels in zip(probs, labels):
                                if set(pred_label_set) == set(gold_labels):
                                    hits += 1

                        if isinstance(result_loss, torch.Tensor):
                            loss_expr = loss.item()
                        else:
                            loss_expr = ','.join([f'{l.item()}' for l in result_loss])
                        self.logger.info(f"epoch{epoch} progress: {step}/{total_step} avg training loss: {loss_expr}, training acc: {hits / max(1, total)}")
                        if self.is_master:
                            # output binary trees for different iteration epochs
                            seq_len = inputs["attention_mask"][0].sum()
                            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().data.numpy())
                            self.logger.info(f"input token: {' '.join(tokens)}")
                            
                            if 'trees' in results:
                                trees_dict = results['trees'][-1]
                                split_points = [_ for _ in reversed(
                                    trees_dict['split_points'][0, 0, :].cpu().data.numpy()[:seq_len-1])]
                                _, merged_tree = get_tree_from_merge_trajectory(split_points, seq_len, tokens)
                                self.logger.info(f"parsed tree : {merged_tree}")
                        self.model.train()

            # If eval not enabled: save every epoch
            if self.is_master:
                torch.save(self.model.state_dict(), os.path.join(output_dir, f"model{epoch}.bin"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer{epoch}.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler{epoch}.pt"))
                torch.save(scaler.state_dict(), os.path.join(output_dir, f'scaler{epoch}.pt'))

            if self.is_master and eval_dataloader is not None:
                self.model.eval()
                eval_result = self.eval(eval_dataloader, metric)
                self.logger.info(f"epoch{epoch}, eval acc: {eval_result}")
                self.model.train()
            

    def eval(self, eval_dataloader, metric):
        epoch_iterator = tqdm(eval_dataloader, desc="Eval Iteration")
        with torch.no_grad():
            predictions = []
            references = []
            for _, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                labels = inputs['labels']
                results = self.model(**inputs)
                probs = results['predict']
                pred = torch.argmax(probs, dim=-1)
                # num += len(labels)
                predictions.extend(pred.tolist())
                references.extend(labels)
                # hit += torch.sum(pred==labels)
        return metric.compute(predictions=predictions, references=references)
        # return hit/num


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("Arguments for glue trainer")
    cmd.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before " "performing a backward/update pass.",
    )
    cmd.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    cmd.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    cmd.add_argument("--parser_lr", default=1e-3, type=float)
    cmd.add_argument("--config_path", required=True, type=str, help="bert model config")
    cmd.add_argument("--vocab_dir", required=True, type=str, help="Directory to the vocabulary")
    cmd.add_argument("--task_type", required=True, type=str, help="Specify the glue task")
    cmd.add_argument("--glue_dir", required=True, type=str, help="path to the directory of glue dataset")
    cmd.add_argument('--cache_dir', required=False, default=None, type=str)
    cmd.add_argument("--max_batch_len", default=512, type=int)
    cmd.add_argument("--mask_epoch", type=int, default=10)
    cmd.add_argument("--max_batch_size", default=32, type=int)
    cmd.add_argument("--output_dir", required=True, type=str, help="save dir")
    cmd.add_argument("--num_samples", default=100, type=int)
    cmd.add_argument("--epochs", default=10, type=int, help="training epochs")
    cmd.add_argument("--pretrain_dir", type=str, required=False, default=None)
    cmd.add_argument("--warm_up", type=float, default=0.01)
    cmd.add_argument("--log_step", default=50, type=int)
    cmd.add_argument("--eval_step", default=5000, type=int)
    cmd.add_argument("--empty_label_idx", default=-1, type=int)
    cmd.add_argument("--seed", default=2023, type=int)
    cmd.add_argument("--noise_corpus", default=None, type=str)
    cmd.add_argument("--enable_epoch_eval", default=False, action='store_true')
    cmd.add_argument("--model_name", type=str, required=True)
    cmd.add_argument("--tree_path", required=False, type=str, default=None)
    cmd.add_argument("--enable_shortcut", default=False, action='store_true')
    cmd.add_argument("--add_mlm_task", default=False, action='store_true')
    cmd.add_argument("--collator_fn", choices=['seperate', 'concat'], default='seperate')

    args = cmd.parse_args(sys.argv[1:])
    
    set_seed(args.seed)
    
    logging.getLogger().setLevel(logging.INFO)
    
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = -1
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)  # for multi-process in a single machine with multiple GPUs.
        global_rank = local_rank
        while True:
            try:
                torch.distributed.init_process_group(backend="nccl", init_method="env://")
                if torch.distributed.is_initialized():
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
            device = torch.device("cuda:0")

    is_master = local_rank == -1 or global_rank == 0
    if not os.path.exists(args.output_dir) and is_master:
        os.makedirs(args.output_dir, exist_ok=True)
    if is_master:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.output_dir, "training_log.txt"), mode="a", encoding="utf-8")
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)
        logger.addHandler(fh)
    else:
        logger = logging

    logger.info(f'initialize model on {global_rank}')

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    enable_dp = 'dp' in args.model_name.split('_')

    dataset = create_glue_dataset(tokenizer, enable_dp, args.task_type, args.glue_dir, 
                                  'train', args.max_batch_len, args.max_batch_size, 
                                  sampler="sequential", noise_corpus=args.noise_corpus, 
                                  empty_label_idx=args.empty_label_idx, tree_path=args.tree_path, 
                                  enable_shortcut=args.enable_shortcut,
                                  cache_dir=args.cache_dir)
    # TODO 1: update model factory 

    model = create_classification_model(args.model_name, dataset.model_type, args.config_path, \
                                        len(dataset.labels), args.pretrain_dir)
    mask_id = dataset._tokenizer.convert_tokens_to_ids('[MASK]')
    sep_id = dataset._tokenizer.convert_tokens_to_ids('[SEP]')
    collator = GlueCollator(dataset.model_type, mask_id, sep_id, mask_rate=0.15, mask_epochs=args.mask_epoch)
    if args.collator_fn == 'seperate':
        collator_fn = collator.mlm_collator
    elif args.collator_fn == 'concat':
        collator_fn = collator.mlm_concat_collator
    model.to(device)
    parser_params = []
    other_params = []
    for name, params in model.named_parameters():
        if name.startswith('parser'):
            parser_params.append(params)
        else:
            other_params.append(params)

    apex_enable = False
    
    recover_epoch = get_max_epoch(args.output_dir, pattern='model*.bin')
    
    if recover_epoch is not None and recover_epoch >= 0:
        model.from_checkpoint(os.path.join(args.output_dir, f"model{recover_epoch}.bin"))
        print('load from checkpoint: {recover_epoch}')

    if global_rank == -1: 
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=SequentialSampler(dataset),
            collate_fn=collator_fn,
        )
        n_gpu = 1
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW([{"params": other_params},
                           {"params": parser_params, "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
        logging.info(
            f"Rank {global_rank} using SequentialSampler, total steps: {t_total}, warm up steps: {warm_up_steps}"
        )

    elif global_rank >= 0:
        n_gpu = 1
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=MyDistributedSampler(dataset, shuffle=False),
            collate_fn=collator_fn,
            drop_last=True,
        )
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW([{"params": other_params},
                           {"params": parser_params, "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
        logging.info(
            f"Rank {global_rank} using MyDistributedSampler, total steps: {t_total}, warm up steps: {warm_up_steps}"
        )
        model = DDP(model, find_unused_parameters=True)

    if is_master:
        special_ids = [x for x in range(30)]
        special_toks = tokenizer.convert_ids_to_tokens(special_ids)
        tok_id = " ".join([f"{si}:{st}" for si, st in zip(special_ids, special_toks)])
        logger.info(f"Special id token pairs: {tok_id}")
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            if os.path.isfile(args.config_path):
                shutil.copyfile(args.config_path,
                                os.path.join(args.output_dir, 'config.json'))
            else:
                shutil.copyfile(os.path.join(args.vocab_dir, 'config.json'),
                                os.path.join(args.output_dir, 'config.json'))
            shutil.copyfile(os.path.join(args.vocab_dir, 'vocab.txt'),
                            os.path.join(args.output_dir, 'vocab.txt'))
        except RuntimeError:
            pass
    else:
        logger = logging

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total)
    scaler = torch.cuda.amp.GradScaler()
    
    if recover_epoch is not None and recover_epoch >= 0:
        optimizer_recover_checkpoint = os.path.join(args.output_dir, f"optimizer{recover_epoch}.pt")
        optimizer.load_state_dict(torch.load(optimizer_recover_checkpoint, map_location="cpu"))

        scheduler_checkpoint = os.path.join(args.output_dir, f"scheduler{recover_epoch}.pt")
        scheduler.load_state_dict(torch.load(scheduler_checkpoint, map_location="cpu"))
        
        scaler_checkpoint = os.path.join(args.output_dir, f"scaler{recover_epoch}.pt")
        scaler.load_state_dict(torch.load(scaler_checkpoint, map_location='cpu'))

    if args.enable_epoch_eval:
        eval_dataset = create_glue_dataset(tokenizer, enable_dp, args.task_type, args.glue_dir, 
                                           'dev', args.max_batch_len, args.max_batch_size, 
                                           sampler="sequential", noise_corpus=args.noise_corpus, 
                                           empty_label_idx=args.empty_label_idx, tree_path=args.tree_path, enable_shortcut=args.enable_shortcut,
                                           cache_dir=args.cache_dir) # concat_pair for r2d2_iter
        eval_collator = GlueCollator(eval_dataset.model_type, mask_id, sep_id, mask_rate=0.0, mask_epochs=0)
        if args.collator_fn == 'seperate':
            eval_collate_fn = eval_collator.mlm_collator
        elif args.collator_fn == 'concat':
            eval_collate_fn = eval_collator.mlm_concat_collator
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            sampler=SequentialSampler(eval_dataset),
            collate_fn=eval_collate_fn,
        )
        evaluator = None
        # evaluator = GlueEvaluater(
        #     model,
        #     device=device,
        #     tokenizer=tokenizer,
        #     logger=logger)
    else:
        eval_dataloader = None
        evaluator = None

    trainer = GlueTrainer(
        model,
        device=device,
        tokenizer=tokenizer,
        logger=logger,
        is_master=is_master,
        n_gpu=n_gpu,
        apex_enable=apex_enable
    )
    
    metric = None
    # if args.task_type=="cola":
    # from datasets import load_metric
    # metric = load_metric("glue", TASK_MAPPING[args.task_type]) # None
    metric = evaluate.load('glue', TASK_MAPPING[args.task_type])

    trainer.train(
        dataloader,
        optimizer,
        scheduler,
        scaler,
        evaluator=evaluator,
        eval_dataloader=eval_dataloader,
        log_step=args.log_step,
        eval_step=args.eval_step,
        output_dir=args.output_dir,
        epochs=args.epochs,
        update_epoch=collator.set_epoch,
        max_grad_norm=args.max_grad_norm,
        recover_epoch=recover_epoch,
        metric=metric
    )
