import argparse
from cgi import print_arguments
import logging
import os
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from model.fast_r2d2_downstream import FastR2D2Classification, FastR2D2CrossSentence
from model.fast_r2d2_dp_classification import FastR2d2DPClassification
from model.fast_r2d2_dp_classification import FastR2d2DPClassification
from reader.snips_reader import SnipsReader
from sklearn.metrics import accuracy_score
from trainer.model_factory import create_classification_model
from utils.model_loader import load_model


class ClassificationEvaluater:
    def __init__(self, model, force_encoding, tokenizer, device, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.force_encoding = force_encoding
        self.device = device

    def eval(
        self,
        data_loader: DataLoader,
        draw_heatmap=False,
        model_dir=None
    ):
        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.eval()
        pred_labels = []
        gold_labels = []
        if draw_heatmap and not os.path.exists(model_dir+'/heatmap'):
            os.mkdir(os.path.join(model_dir, 'heatmap'))
        with torch.no_grad():
            for _, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                labels = inputs['labels']
                if draw_heatmap:
                    inputs['keep_weights'] = True
                with torch.no_grad():
                    # results = self.model(**inputs, force_encoding=self.force_encoding)
                    results = self.model(**inputs)
                    probs = results['predict']
                if isinstance(probs, torch.Tensor):
                    predict_labels = probs.argmax(dim=-1)
                    for pred_label in predict_labels:
                        pred_labels.append(pred_label.tolist())
                else:
                    for label_list in probs:
                        if len(label_list) == 1:
                            pred_labels.append(label_list[0])
                        else:
                            pred_labels.append(-1)
                if isinstance(labels, torch.Tensor):
                    gold_labels.extend(labels.tolist())
                else:
                    gold_labels.extend([lb[0] for lb in labels])
        result = accuracy_score(pred_labels, gold_labels)
        self.logger.info(f'eval result {result}')
        return result


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("The testing components of")
    cmd.add_argument("--config_path", required=True, type=str, help="bert model config")
    cmd.add_argument("--vocab_dir", required=True, type=str, help="Directory to the vocabulary")
    cmd.add_argument("--model_dir", required=True, type=str)
    cmd.add_argument("--task_type", required=True, type=str, help="Specify the glue task")
    cmd.add_argument("--glue_dir", required=True, type=str, help="path to the directory of glue dataset")
    cmd.add_argument("--r2d2_mode", default='cky', choices=['cky', 'forced'], type=str)
    cmd.add_argument("--turn", default='', type=str)
    cmd.add_argument("--acc", default='', type=str)
    cmd.add_argument("--max_batch_len", default=2048, type=int)
    cmd.add_argument("--max_batch_size", default=32, type=int)
    cmd.add_argument("--model_name", required=True, type=str)
    cmd.add_argument("--tree_path", default=None, required=False, type=str)

    args = cmd.parse_args(sys.argv[1:])

    logging.getLogger().setLevel(logging.INFO)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)


    enable_dp = 'dp' in args.model_name.split('_')
    dataset = SnipsReader(
        args.data_dir,
        tokenizer,
        batch_max_len=args.max_batch_len,
        batch_size=args.max_batch_size,
        mode="test",
        task=args.task,
    )

    model = create_classification_model(args.model_name, dataset.model_type, args.config_path, len(dataset.labels), None, None)

    if args.model_dir is not None:
        model_path = os.path.join(args.model_dir, f'model{args.turn}.bin')
        if hasattr(model, "load_model"):
            model.load_model(model_path)
        else:
            load_model(model, model_path)

    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        collate_fn=dataset.collate_batch,
    )

    logger = logging

    evaluator = ClassificationEvaluater(
        model,
        device=device,
        force_encoding=args.r2d2_mode=='forced',
        tokenizer=tokenizer,
        logger=logger
    )

    evaluator.eval(
        dataloader,
        draw_heatmap=False,
        model_dir=args.model_dir
    )
