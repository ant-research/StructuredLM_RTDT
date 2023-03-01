# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xinyu Kong


import argparse
from copy import copy
import sys
import logging
import os
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from reader.multi_label_reader import MultiLabelReader
from reader.snips_reader import SnipsReader
from reader.slu_reader import StanfordLUReader
from reader.sls_reader import SLSReader
from utils.model_loader import load_model
from sklearn.metrics import f1_score
from trainer.model_factory import create_bert_model


class MultiLabelEvaluater:
    def __init__(self, model, tokenizer, device, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.device = device

    def eval(
        self,
        dev_dataloader: DataLoader,
        train_dataloader=None
    ):

        epoch_iterato_dev = tqdm(dev_dataloader, desc="Iteration")
        # epoch_iterato_train = tqdm(train_dataloader, desc="Iteration")
        id2label_dict = copy.deepcopy(dev_dataloader.dataset.id2label_dict)
        id2label_dict[len(id2label_dict)] = 'terminal'
        id2label_dict[len(id2label_dict)] = 'nonterminal'
        self.model.eval()
        with torch.no_grad():
            y_preds = []
            y_trues_onehot = []
            for _, inputs in enumerate(epoch_iterato_dev):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                labels = inputs.pop('labels')
                target = np.full((len(labels), len(dev_dataloader.dataset.labels)), fill_value=0).tolist()
                for batch_i, intents_i in enumerate(labels):
                    for intent_idx in intents_i:
                        target[batch_i][intent_idx] = 1
                y_trues_onehot.extend(target)
                with torch.no_grad():
                    results = self.model(**inputs)
                    probs = results['predict']
                    for prob in probs:
                        if not isinstance(probs, torch.Tensor):
                            y_preds.append(prob)
                        else:
                            y_preds.append(prob.tolist())
            if not isinstance(probs, torch.Tensor):
                # dp mode
                y_preds_onehot = [[0 for _ in range(len(id2label_dict)-2)] for _ in range(len(y_preds))]
                for batch_i, intents_i in enumerate(y_preds):
                    for intent_idx in intents_i:
                        y_preds_onehot[batch_i][intent_idx] = 1      
                # max_acc = accuracy_score(y_trues,y_preds_onehot)
                f1_micro = f1_score(y_trues_onehot, y_preds_onehot, average='micro')
                f1_weighted = f1_score(y_trues_onehot, y_preds_onehot, average='weighted')
            else:
                if train_dataloader is None:
                    threshold_micro = 0.5
                    threshold_weighted = 0.5
                else:
                    threshold_micro, threshold_weighted = self.get_threshold(train_dataloader)
                y_preds_onehot_micro = np.array(copy.deepcopy(y_preds))
                y_preds_onehot_micro[y_preds_onehot_micro>=threshold_micro] = 1
                y_preds_onehot_micro[y_preds_onehot_micro<threshold_micro] = 0
                y_preds_onehot_micro = y_preds_onehot_micro.tolist()
                y_preds_onehot_weighted = np.array(copy.deepcopy(y_preds))
                y_preds_onehot_weighted[y_preds_onehot_weighted>=threshold_weighted] = 1
                y_preds_onehot_weighted[y_preds_onehot_weighted<threshold_weighted] = 0
                y_preds_onehot_weighted = y_preds_onehot_weighted.tolist()
                # acc = accuracy_score(y_trues_onehot,y_preds_onehot)
                f1_micro = f1_score(y_trues_onehot, y_preds_onehot_micro, average='micro')
                # precision_micro = precision_score(y_trues, y_preds, average='micro')
                # recall_micro = recall_score(y_trues, y_preds, average='micro')
                # single label f1
                # single_precision = precision_score(y_trues, y_preds, average=None)
                # single_recall = recall_score(y_trues, y_preds, average=None)
                # single_f1 = f1_score(y_trues, y_preds, average=None)
                f1_weighted = f1_score(y_trues_onehot, y_preds_onehot_weighted, average='weighted')

                self.logger.info(f'eval threshold_micro {threshold_micro} threshold_weighted {threshold_weighted}')
            """

            count = 0
            for _, inputs in enumerate(epoch_iterato_train):
                if count>=100:
                    break
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.no_grad():
                    results = self.model(**inputs)
                    probs = results['predict']
                if draw_heatmap:
                    get_cache_id = lambda x: x.decode_cache_id
                    # get_cache_id = lambda x: x.cache_id
                    key_nodes_list = results['key_nodes_list']
                    cls_probs = results['cls_probs']
                    for batch_id, nodes_order in enumerate(key_nodes_list):
                        predict_json = {'mode':'train'}
                        predict_json['text'] = ' '.join(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][batch_id]))
                        for node in nodes_order:
                            predict_json[id2label_dict[cls_probs[get_cache_id(node)].tolist().index(max(cls_probs[get_cache_id(node)]))]] = \
                                [' '.join(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][batch_id][node.i:node.j + 1])), node.i, node.j + 1]
                        predict_result.append(predict_json)
                        count+=1
            """
            
        # self.logger.info(f'eval result acc {max_acc}')
        self.logger.info(f'eval result f1_micro {f1_micro} f1_weighted {f1_weighted}')
        return f1_micro

    def get_threshold(self, train_dataloader):
        train_iterato_dev = tqdm(train_dataloader, desc="threshold iteration")
        id2label_dict = copy.deepcopy(train_dataloader.dataset.id2label_dict)
        id2label_dict[len(id2label_dict)] = 'terminal'
        id2label_dict[len(id2label_dict)] = 'nonterminal'
        self.model.eval()
        threshold = 0.01
        max_f1_micro = 0.
        max_f1_weighted = 0.
        with torch.no_grad():
            y_preds = []
            y_trues = []
            for _, inputs in enumerate(train_iterato_dev):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                y_trues.extend(inputs.pop('labels'))
                with torch.no_grad():
                    results = self.model(**inputs)
                    probs = results.get('predict', [])
                    for prob in probs:
                        y_preds.append(prob.tolist())
        y_trues_onehot = [[0 for _ in range(len(id2label_dict)-2)] for _ in range(len(y_preds))]
        for batch_i, intents_i in enumerate(y_trues):
            for intent_idx in intents_i:
                y_trues_onehot[batch_i][intent_idx] = 1
        while threshold <= 0.99:
            y_preds_onehot = np.array(copy.deepcopy(y_preds))
            y_preds_onehot[y_preds_onehot>=threshold] = 1
            y_preds_onehot[y_preds_onehot<threshold] = 0
            y_preds_onehot = y_preds_onehot.tolist()
            # acc = accuracy_score(y_trues_onehot,y_preds_onehot)
            # if acc>max_acc:
                # max_acc =acc
            f1 = f1_score(y_trues_onehot, y_preds_onehot, average='micro')
            if f1>max_f1_micro:
                max_f1_micro = f1
                best_threshold_micro = threshold
            f1 = f1_score(y_trues_onehot, y_preds_onehot, average='weighted')
            if f1>max_f1_weighted:
                max_f1_weighted = f1
                best_threshold_weighted = threshold
            threshold+=0.01
        return best_threshold_micro, best_threshold_weighted


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("The testing components of")
    cmd.add_argument("--config_path", required=True, type=str, help="bert model config")
    cmd.add_argument("--vocab_dir", required=True, type=str, help="Directory to the vocabulary")
    cmd.add_argument("--model_dir", required=True, type=str)
    cmd.add_argument("--data_dir", required=True, type=str, help="path to the directory of dataset")
    cmd.add_argument("--output_dir", default='data/test_atis_slots/')
    cmd.add_argument("--turn", default='', type=str)
    cmd.add_argument("--draw_heatmap", default=True, type=bool)
    cmd.add_argument("--datasets", type=str, required=True)
    cmd.add_argument("--domain", type=str, required=False)
    cmd.add_argument("--task", type=str, choices=["NER", "intent"], default="NER")
    cmd.add_argument("--model_name", required=True, type=str, help='args: bert_[multilabel]*')
    cmd.add_argument("--sampler", choices=["random", "sequential"], default="random", help="sampling input data")
    cmd.add_argument("--max_batch_len", default=512, type=int)
    cmd.add_argument("--max_batch_size", default=32, type=int)

    args = cmd.parse_args(sys.argv[1:])

    logging.getLogger().setLevel(logging.INFO)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    if args.datasets == 'snips':
        dev_dataset = SnipsReader(
            args.data_dir,
            tokenizer,
            batch_max_len=args.max_batch_len,
            batch_size=args.max_batch_size,
            mode="test",
            task=args.task,
        )
    elif args.datasets == 'atis':
        dev_dataset = MultiLabelReader(
            args.data_dir,
            tokenizer,
            batch_max_len=args.max_batch_len,
            batch_size=args.max_batch_size,
            mode="test",
            task=args.task,
        )
        train_dataset = MultiLabelReader(
            args.data_dir,
            tokenizer,
            batch_max_len=args.max_batch_len,
            batch_size=args.max_batch_size,
            mode="train",
            task=args.task,
        )
    elif args.datasets == 'stanfordLU':
        dev_dataset = StanfordLUReader(
            args.data_dir,
            tokenizer,
            batch_max_len=args.max_batch_len,
            batch_size=args.max_batch_size,
            mode="test",
            domain=args.domain,
            task=args.task,
        )
        train_dataset = StanfordLUReader(
            args.data_dir,
            tokenizer,
            batch_max_len=args.max_batch_len,
            batch_size=args.max_batch_size,
            mode="train",
            domain=args.domain,
            task=args.task,
        )
    elif args.datasets == 'sls':
        dev_dataset = SLSReader(
            args.data_dir,
            tokenizer,
            batch_max_len=args.max_batch_len,
            batch_size=args.max_batch_size,
            random=args.sampler != "sequential",
            domain=args.domain,
            mode="test",
        )
        train_dataset = SLSReader(
            args.data_dir,
            tokenizer,
            batch_max_len=args.max_batch_len,
            batch_size=args.max_batch_size,
            mode="train",
            domain=args.domain,
        )

    model = create_bert_model(args.model_name, args.config_path, len(dev_dataset.labels), None, None)

    if args.model_dir is not None:
        model_path = os.path.join(args.model_dir, f'model{args.turn}.bin')
        load_model(model, model_path)

    model.to(device)

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=1,
        sampler=SequentialSampler(dev_dataset),
        collate_fn=dev_dataset.collate_batch_bert,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=SequentialSampler(train_dataset),
        collate_fn=train_dataset.collate_batch_bert,
    )

    logger = logging

    evaluator = MultiLabelEvaluater(
        model,
        device=device,
        tokenizer=tokenizer,
        logger=logger
    )

    evaluator.eval(
        dev_dataloader,
        train_dataloader=None
    )
