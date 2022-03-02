# coding=utf-8
# Copyright (c) 2021 Ant Group

import argparse
import logging
import os
import time

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AdamW, AutoConfig, get_linear_schedule_with_warmup
from model.fast_r2d2_downstream import FastR2D2Classification, FastR2D2CrossSentence
from reader.glue_reader import R2D2GlueReader
import sys
import glob


def get_max_epoch_model(output_dir):
    fn_dir_list = glob.glob(os.path.join(output_dir, "model*bin"))
    if not fn_dir_list:
        return None

    def get_epoch_num(fn):
        if "/" in fn:
            fn = fn.rsplit("/", 1)[1]
        epoch = int(fn.replace("model", "").replace(".bin", ""))
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
            output_dir,
            log_step,
            epochs,
            num_samples,
            recover_epoch=None,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
    ):

        if recover_epoch is not None:
            optimizer_recover_checkpoint = os.path.join(output_dir, f"optimizer.pt")
            optimizer.load_state_dict(torch.load(optimizer_recover_checkpoint, map_location="cpu"))

            scheduler_checkpoint = os.path.join(output_dir, f"scheduler.pt")
            scheduler.load_state_dict(torch.load(scheduler_checkpoint, map_location="cpu"))

            if self.apex_enable:
                amp_checkpoint = os.path.join(args.output_dir, f"amp.pt")
                amp.load_state_dict(torch.load(amp_checkpoint, map_location="cpu"))

        train_iterator = trange(0, int(epochs), desc="Epoch")
        total_step = len(data_loader)
        training_loss, training_step = 0, 0
        for epoch in train_iterator:
            if recover_epoch and epoch <= recover_epoch:
                logger.info(f"Skip epoch {epoch}")
                continue

            if isinstance(data_loader, DataLoader) and isinstance(data_loader.sampler, DistributedSampler):
                self.logger.info(f"Set sampler epoch: {epoch}")
                data_loader.sampler.set_epoch(epoch)

            epoch_iterator = tqdm(data_loader, desc="Iteration")

            for step, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                loss = self.model(**inputs, num_samples=num_samples)
                if self.n_gpu > 1:
                    loss = loss.mean()

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if self.apex_enable:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.apex_enable:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                training_step += 1
                training_loss = training_loss + (loss.item() - training_loss) / training_step
                if step % log_step == 0 and step > 0:
                    with torch.no_grad():
                        self.logger.info(f"progress: {step}/{total_step} avg training loss: {training_loss}")
            if self.is_master:
                torch.save(self.model.state_dict(), os.path.join(output_dir, f"model{epoch}.bin"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler.pt"))
                if self.apex_enable:
                    torch.save(amp.state_dict(), os.path.join(output_dir, "amp.pt"))
        if self.is_master:
            torch.save(self.model.state_dict(), os.path.join(output_dir, f"model.bin"))


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("Arguments for glue trainer")
    cmd.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before " "performing a backward/update pass.",
    )
    cmd.add_argument("--max_grad_norm", default=0.0, type=float, help="Max gradient norm.")
    cmd.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    cmd.add_argument("--parser_lr", default=1e-3, type=float)
    cmd.add_argument("--config_path", required=True, type=str, help="bert model config")
    cmd.add_argument("--vocab_dir", required=True, type=str, help="Directory to the vocabulary")
    cmd.add_argument("--task_type", required=True, type=str, help="Specify the glue task")
    cmd.add_argument("--glue_dir", required=True, type=str, help="path to the directory of glue dataset")
    cmd.add_argument("--max_batch_len", default=512, type=int)
    cmd.add_argument("--max_batch_size", default=32, type=int)
    cmd.add_argument("--output_dir", required=True, type=str, help="save dir")
    cmd.add_argument("--local_rank", default=-1, type=int, help="multi gpu training")
    cmd.add_argument("--num_samples", default=100, type=int)
    cmd.add_argument("--epochs", default=10, type=int, help="training epochs")
    cmd.add_argument("--pretrain_dir", type=str, required=False, default=None)
    cmd.add_argument("--warm_up", type=float, default=0.01)
    cmd.add_argument("--log_step", default=100, type=int)
    cmd.add_argument("--apex_mode", default="O1", type=str)
    cmd.add_argument("--sampler", choices=["random", "sequential"], default="sequential", help="sampling input data")

    args = cmd.parse_args(sys.argv[1:])

    logging.getLogger().setLevel(logging.INFO)

    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)  # for multi-process in a single machine with multiple GPUs.
        global_rank = args.local_rank
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

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    dataset = R2D2GlueReader(
        args.task_type,
        args.glue_dir,
        "train",
        tokenizer,
        max_batch_len=args.max_batch_len,
        max_batch_size=args.max_batch_size,
        batch_by_len=args.sampler == "sequential",
    )

    config = AutoConfig.from_pretrained(args.config_path)

    if dataset.model_type == 'single':
        model = FastR2D2Classification(config, len(dataset.labels))
    elif dataset.model_type == 'pair':
        model = FastR2D2CrossSentence(config, len(dataset.labels))

    if args.pretrain_dir is not None:
        model_path = os.path.join(args.pretrain_dir, 'model.bin')
        parser_path = os.path.join(args.pretrain_dir, 'parser.bin')
        model.from_pretrain(model_path, parser_path)

    recover_epoch = get_max_epoch_model(args.output_dir)
    if recover_epoch is not None:
        model_recover_checkpoint = os.path.join(args.output_dir, f"model{recover_epoch}.bin")
        logging.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint)
        model.load_model(model_recover_checkpoint)

    model.to(device)
    apex_enable = False
    global_rank = args.local_rank
    if global_rank == -1:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=RandomSampler(dataset) if args.sampler == "random" else SequentialSampler(dataset),
            collate_fn=dataset.collate_batch,
        )
        n_gpu = 1
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW([{"params": model.r2d2.parameters()},
                           {"params": model.classifier.parameters()},
                           {"params": model.parser.parameters(), "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
        logging.info(
            f"Rank {global_rank} uses {args.sampler} sampler, total steps: {t_total}, warm up steps: {warm_up_steps}"
        )

        try:
            from apex import amp

            logging.info(f"Enable apex successful on {global_rank}")
            t_total = int(len(dataloader) * args.epochs)
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_mode)
        except Exception as e:
            logging.error(e)
            logging.error("import apex failed")
    elif global_rank >= 0:
        n_gpu = 1
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=DistributedSampler(dataset, shuffle=(args.sampler == "random")),
            collate_fn=dataset.collate_batch,
            drop_last=True,
        )
        t_total = int(len(dataloader) * args.epochs)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW([{"params": model.r2d2.parameters()},
                           {"params": model.classifier.parameters()},
                           {"params": model.parser.parameters(), "lr": args.parser_lr}],
                          lr=args.lr, correct_bias=False)
        logging.info(
            f"Rank {global_rank} uses {args.sampler} sampler, total steps: {t_total}, warm up steps: {warm_up_steps}"
        )
        try:
            from apex import amp
            from apex.parallel import DistributedDataParallel

            logging.info(f"Enable apex successful on {global_rank}")
            t_total = int(len(dataloader) * args.epochs)
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_mode)
            model = DistributedDataParallel(model, delay_allreduce=True)
            apex_enable = True
        except Exception as e:
            logging.error(e)
            logging.error("import apex failed")
            sys.exit(-1)

    is_master = args.local_rank == -1 or global_rank == 0
    if not os.path.exists(args.output_dir) and is_master:
        os.makedirs(args.output_dir, exist_ok=True)
    if is_master:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.output_dir, "training_log.txt"), mode="a", encoding="utf-8")
        logger.addHandler(fh)
        special_ids = [x for x in range(30)]
        special_toks = tokenizer.convert_ids_to_tokens(special_ids)
        tok_id = " ".join([f"{si}:{st}" for si, st in zip(special_ids, special_toks)])
        logger.info(f"Special id token pairs: {tok_id}")

    else:
        logger = logging

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total)
    trainer = GlueTrainer(
        model,
        device=device,
        tokenizer=tokenizer,
        logger=logger,
        is_master=is_master,
        n_gpu=n_gpu,
        apex_enable=apex_enable
    )

    trainer.train(
        dataloader,
        optimizer,
        scheduler,
        log_step=args.log_step,
        output_dir=args.output_dir,
        epochs=args.epochs,
        num_samples=args.num_samples,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        recover_epoch=recover_epoch,
    )
