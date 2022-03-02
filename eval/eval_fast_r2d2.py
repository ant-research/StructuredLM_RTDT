import argparse
import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from model.fast_r2d2_downstream import FastR2D2Classification, FastR2D2CrossSentence
from reader.glue_reader import R2D2GlueReader
from datasets import load_metric
import sys


TASK_MAPPING = {
    'sst-2': 'sst2',
    'mnli-mm': 'mnli_mismatched',
    'mnli': 'mnli_matched',
    'cola': 'cola',
    'qqp': 'qqp'
}


class GlueEvaluater:
    def __init__(self, model, force_encoding, tokenizer, device, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.force_encoding = force_encoding

        self.device = device

    def eval(
        self,
        data_loader: DataLoader,
        metric
    ):

        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.eval()
        pred_labels = []
        gold_labels = []
        with torch.no_grad():
            for _, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                labels = inputs.pop('labels')
                with torch.no_grad():
                    probs = self.model(**inputs, force_encoding=self.force_encoding)
                predict_labels = probs.argmax(dim=-1)
                for pred_label in predict_labels:
                    pred_labels.append(pred_label)
                for gold_label in labels:
                    gold_labels.append(gold_label)
        result = metric.compute(predictions=np.array(pred_labels), references=np.array(gold_labels))
        print(f'eval result {result}')


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("The testing components of")
    cmd.add_argument("--config_path", required=True, type=str, help="bert model config")
    cmd.add_argument("--vocab_dir", required=True, type=str, help="Directory to the vocabulary")
    cmd.add_argument("--model_dir", required=True, type=str)
    cmd.add_argument("--task_type", required=True, type=str, help="Specify the glue task")
    cmd.add_argument("--glue_dir", required=True, type=str, help="path to the directory of glue dataset")
    cmd.add_argument("--r2d2_mode", default='cky', choices=['cky', 'forced'], type=str)
    cmd.add_argument("--turn", default='', type=str)
    cmd.add_argument("--max_batch_len", default=512, type=int)
    cmd.add_argument("--max_batch_size", default=32, type=int)

    args = cmd.parse_args(sys.argv[1:])

    logging.getLogger().setLevel(logging.INFO)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    dataset = R2D2GlueReader(
        args.task_type,
        args.glue_dir,
        "dev",
        tokenizer,
        max_batch_len=args.max_batch_len,
        max_batch_size=args.max_batch_size,
    )

    config = AutoConfig.from_pretrained(args.config_path)
    metric = load_metric("glue", TASK_MAPPING[args.task_type])

    if dataset.model_type == 'single':
        model = FastR2D2Classification(config, len(dataset.labels))
    elif dataset.model_type == 'pair':
        model = FastR2D2CrossSentence(config, len(dataset.labels))

    if args.model_dir is not None:
        model_path = os.path.join(args.model_dir, f'model{args.turn}.bin')
        model.load_model(model_path)

    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        collate_fn=dataset.collate_batch,
    )

    logger = logging

    evaluator = GlueEvaluater(
        model,
        device=device,
        force_encoding=args.r2d2_mode=='forced',
        tokenizer=tokenizer,
        logger=logger
    )

    evaluator.eval(
        dataloader,
        metric
    )
