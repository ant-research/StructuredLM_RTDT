import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModel
from model.fast_r2d2_downstream import FastR2D2Classification
import sys
from tqdm import tqdm
import time
import multiprocessing
from reader.memory_line_reader import BatchSelfRegressionLineDataset


class Processor(multiprocessing.Process):
    def __init__(self, args, task_queue) -> None:
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue


class SpeedEvaluater:
    def __init__(self, model, force_encoding, tokenizer, device, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.force_encoding = force_encoding

        self.device = device

    def eval(
        self,
        data_loader: DataLoader,
        call_fn
    ):
        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.eval()
        start = time.time()
        with torch.no_grad():
            for _, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                #self.model(**inputs, force_encoding=self.force_encoding)
                call_fn(self.model, inputs, self.force_encoding)
        end = time.time()
        print(f'total time cost: {end - start}')


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("Evaluate inference speed.")
    cmd.add_argument("--config_path", required=True, type=str, help="bert model config")
    cmd.add_argument("--vocab_dir", required=True, type=str, help="Directory to the vocabulary")
    cmd.add_argument("--corpus_path", required=True, type=str)
    cmd.add_argument("--model_dir", required=True, type=str)
    cmd.add_argument("--model", default='fast-r2d2', choices=['bert', 'fast-r2d2', 'r2d2'])
    cmd.add_argument("--r2d2_mode", default='cky', choices=['cky', 'forced'], type=str)
    cmd.add_argument("--turn", default='', type=str)
    cmd.add_argument("--input_type", choices=['txt', 'ids'], default='txt')
    cmd.add_argument("--batch_size", default=50, type=int)
    cmd.add_argument("--max_batch_len", default=51200000, type=int)

    args = cmd.parse_args(sys.argv[1:])

    logging.getLogger().setLevel(logging.INFO)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    dataset = BatchSelfRegressionLineDataset(
        args.corpus_path,
        tokenizer,
        batch_max_len=args.max_batch_len,
        min_len=1,
        batch_size=args.batch_size,
        input_type=args.input_type
    )

    config = AutoConfig.from_pretrained(args.config_path)

    model_call_fn = None
    if args.model == 'fast-r2d2':
        model = FastR2D2Classification(config, 1)
        if args.model_dir is not None:
            model_path = os.path.join(args.model_dir, f'model{args.turn}.bin')
            parser_path = os.path.join(args.model_dir, f'parser{args.turn}.bin')
            model.from_pretrain(model_path, parser_path)
        model_call_fn = lambda model, inputs, forced: model(**inputs, force_encoding=forced)
    elif args.model == 'bert':
        model = AutoModel.from_pretrained(args.model_dir)
        def call_transformer(model, inputs, forced):
            inputs.pop('atom_spans')
            model(**inputs)
        model_call_fn = call_transformer
    elif args.model == 'r2d2':
        from model.r2d2 import R2D2
        model = R2D2(config)
        model_call_fn = lambda model, inputs, forced: model(**inputs)

    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        collate_fn=dataset.collate_batch,
    )

    logger = logging

    evaluator = SpeedEvaluater(
        model,
        device=device,
        force_encoding=args.r2d2_mode=='forced',
        tokenizer=tokenizer,
        logger=logger
    )

    evaluator.eval(
        dataloader,
        model_call_fn
    )