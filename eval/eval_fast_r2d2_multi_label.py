import argparse
import sys
import json
import logging
import os
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from trainer.model_factory import create_multi_label_classification_model
from reader.multi_label_reader import MultiLabelReader
from reader.snips_reader import SnipsReader
from reader.slu_reader import StanfordLUReader
from reader.sls_reader import SLSReader
from experiments.f1_evaluator import F1Evaluator
from utils.model_loader import load_model
from utils.misc import convert_char_span_to_tokenized_span_atis
from collections import deque
from sklearn.metrics import precision_score, recall_score, f1_score
# from multiprocessing.spawn import import_main_path
# from operator import imod
# from ast import arg
# from cgi import print_arguments
# from functools import total_ordering
from sklearn.metrics import accuracy_score, precision_score, recall_score



class MultiLabelEvaluater:
    def __init__(self, model, force_encoding, tokenizer, device, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.force_encoding = force_encoding
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
            y_trues = []
            for _, inputs in enumerate(epoch_iterato_dev):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                y_trues.extend(inputs['labels'])
                with torch.no_grad():
                    results = self.model(**inputs)
                    probs = results.get('predict', [])
                    for prob in probs:
                        if not isinstance(probs, torch.Tensor):
                            y_preds.append(prob)
                        else:
                            y_preds.append(prob.tolist())
            if not isinstance(probs, torch.Tensor):
                # 意味着是dp模式
                y_preds_onehot = [[0 for _ in range(len(id2label_dict)-2)] for _ in range(len(y_preds))]
                y_trues_onehot = [[0 for _ in range(len(id2label_dict)-2)] for _ in range(len(y_preds))]
                for batch_i, intents_i in enumerate(y_preds):
                    for intent_idx in intents_i:
                        y_preds_onehot[batch_i][intent_idx] = 1
                for batch_i, intents_i in enumerate(y_trues):
                    for intent_idx in intents_i:
                        y_trues_onehot[batch_i][intent_idx] = 1
                f1_micro = f1_score(y_trues_onehot, y_preds_onehot, average='micro')
                f1_weighted = f1_score(y_trues_onehot, y_preds_onehot, average='weighted')
                # max_acc = accuracy_score(y_trues_onehot,y_preds_onehot)
            else:
                if train_dataloader is None:
                    threshold_micro = 0.5
                    threshold_weighted = 0.5
                else:
                    threshold_micro, threshold_weighted = self.get_threshold(train_dataloader)
                y_trues_onehot = [[0 for _ in range(len(id2label_dict)-2)] for _ in range(len(y_preds))]
                for batch_i, intents_i in enumerate(y_trues):
                    for intent_idx in intents_i:
                        y_trues_onehot[batch_i][intent_idx] = 1
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
                f1_weighted = f1_score(y_trues_onehot, y_preds_onehot_weighted, average='weighted')
                # precision_weighted = precision_score(y_trues, y_preds, average='weighted')
                # recall_weighted = recall_score(y_trues, y_preds, average='weighted')
                # 单类别指标
                # single_precision = precision_score(y_trues, y_preds, average=None)
                # single_recall = recall_score(y_trues, y_preds, average=None)
                # single_f1 = f1_score(y_trues, y_preds, average=None)
                self.logger.info(f'eval threshold_micro {threshold_micro} threshold_weighted {threshold_weighted}')
        # self.logger.info(f'eval result {acc}')
        self.logger.info(f'eval result f1_micro {f1_micro} f1_weighted {f1_weighted}')
        return f1_micro
        
    
    def get_threshold(self, train_dataloader):
        train_iterato_dev = tqdm(train_dataloader, desc="threshold iteration")
        id2label_dict = copy.deepcopy(train_dataloader.dataset.id2label_dict)
        id2label_dict[len(id2label_dict)] = 'terminal'
        id2label_dict[len(id2label_dict)] = 'nonterminal'
        self.model.eval()
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
                        y_preds.append(list(prob))
        y_trues_onehot = [[0 for _ in range(len(id2label_dict)-2)] for _ in range(len(y_preds))]
        for batch_i, intents_i in enumerate(y_trues):
            for intent_idx in intents_i:
                y_trues_onehot[batch_i][intent_idx] = 1

        threshold = 0.01
        max_f1_micro = 0.
        max_f1_weighted=0.
        while threshold <= 0.99:
            y_preds_onehot = np.array(copy.deepcopy(y_preds))
            y_preds_onehot[y_preds_onehot>=threshold] = 1
            y_preds_onehot[y_preds_onehot<threshold] = 0
            y_preds_onehot = y_preds_onehot.tolist()
            # acc = accuracy_score(y_trues_onehot,y_preds_onehot)
            # if acc>max_acc:
            #     max_acc =acc
            f1 = f1_score(y_trues_onehot, y_preds_onehot, average='micro')
            if f1>max_f1_micro:
                max_f1_micro = f1
                best_threshold_micro = threshold
            f1 = f1_score(y_trues_onehot, y_preds_onehot, average='weighted')
            if f1>max_f1_weighted:
                max_f1_weighted = f1
                best_threshold_weighted = threshold
            threshold+=0.05
        return best_threshold_micro, best_threshold_weighted
    
    def eval_atis_ner_fixed(self,dev_dataloader: DataLoader,output_dir='',span_bucket="1,2,5"):
        epoch_iterator = tqdm(dev_dataloader, desc="Iteration")
        f1_evaluator = F1Evaluator(dev_dataloader.dataset.id2label_dict.keys(), list(map(lambda x:int(x), span_bucket.split(','))))
        label2id_dict = copy.deepcopy(dev_dataloader.dataset.label2id_dict)
        id2label_dict = copy.deepcopy(dev_dataloader.dataset.id2label_dict)
        id2label_dict[len(id2label_dict)] = 'terminal'
        id2label_dict[len(id2label_dict)] = 'nonterminal'
        self.model.eval()
        with torch.no_grad():
            for _, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                results = self.model(**inputs)
                root_nodes = results['roots']
                atom_spans = inputs['atom_spans']
                offset_mapping = inputs['offset_mapping']
                entities = inputs['entities']
                mean_span_length = inputs['mean_span_length']
                
                for id, root_node in enumerate(root_nodes):
                    label_span_collector = f1_evaluator.create_label_span_collector(entities[id], label2id_dict, [offset_mapping[id]])
                    node_queue = deque()
                    node_queue.append(root_node)
                    while len(node_queue) > 0:
                        current_node = node_queue.popleft()
                        label_id = current_node.label
                        if id2label_dict[label_id] == 'nonterminal':
                            if not current_node.is_leaf and not (current_node.i, current_node.j) in atom_spans[id]:
                                node_queue.append(current_node.left)
                                node_queue.append(current_node.right)
                        else:
                            if id2label_dict[label_id] != 'terminal':
                                token_start = current_node.i
                                token_end = current_node.j + 1
                                if label_id not in label_span_collector:
                                    label_span_collector[label_id] = [None, [(-1, -1)]]
                                if label_span_collector[label_id][0] is None:
                                    label_span_collector[label_id][0] = []
                                for token in [i for i in range(token_start,token_end)]:
                                    if token not in label_span_collector[label_id][0]:
                                        label_span_collector[label_id][0].append(token)
                    f1_evaluator.update(label_span_collector, mean_span_length[id])
        # f1_evaluator.print_results()
        self.logger.info(f'eval result f1 {2 * f1_evaluator.f1_hit_total / (f1_evaluator.f1_pred_total + f1_evaluator.f1_true_total)}')
        self.logger.info(f'eval result f1_mean {f1_evaluator.f1_mean}')
        for bucket_name, bucket in f1_evaluator.f1_bucket.items():
            self.logger.info(f'bucket: {bucket_name} f1: {bucket.f1} f1_mean: {bucket.f1_mean} ratio: {bucket.ratio} entity_count: {bucket.entity_count}')
        return f1_evaluator.f1_mean


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("The testing components of")
    cmd.add_argument("--config_path", required=True, type=str, help="bert model config")
    cmd.add_argument("--vocab_dir", required=True, type=str, help="Directory to the vocabulary")
    cmd.add_argument("--model_dir", required=True, type=str)
    cmd.add_argument("--pretrain_dir", type=str, required=False, default=None)
    cmd.add_argument("--data_dir", required=True, type=str, help="path to the directory of dataset")
    cmd.add_argument("--output_dir", default='data/test_atis_slots/')
    cmd.add_argument("--r2d2_mode", default='cky', choices=['cky', 'forced'], type=str)
    cmd.add_argument("--turn", default='', type=str)
    cmd.add_argument("--draw_heatmap", default=True, type=bool)
    cmd.add_argument("--datasets", type=str, required=True)
    cmd.add_argument("--domain", type=str, required=False)
    cmd.add_argument("--task", type=str, choices=["NER", "intent"], default="NER")
    cmd.add_argument("--model_name", required=True, type=str, help='args: fastr2d2_[dp/overlap/miml/root/exclusive]*')
    cmd.add_argument("--sampler", choices=["random", "sequential"], default="random", help="sampling input data")
    cmd.add_argument("--max_batch_len", default=512, type=int)
    cmd.add_argument("--max_batch_size", default=32, type=int)
    cmd.add_argument("--span_bucket", type=str, default='1,2,5', help="span bucket (1,2),(2,5),(5,)")

    args = cmd.parse_args(sys.argv[1:])

    logging.getLogger().setLevel(logging.INFO)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    
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
    config = AutoConfig.from_pretrained(args.config_path)

    model = create_multi_label_classification_model(args.model_name, args.config_path, len(dev_dataset.labels), args.pretrain_dir)

    if args.model_dir is not None:
        model_path = os.path.join(args.model_dir, f'model{args.turn}.bin')
        load_model(model, model_path)

    model.to(device)

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=1,
        sampler=SequentialSampler(dev_dataset),
        collate_fn=dev_dataset.collate_batch,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=SequentialSampler(train_dataset),
        collate_fn=train_dataset.collate_batch,
    )

    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.output_dir, "eval_result.txt"), mode="a", encoding="utf-8")
    logger.addHandler(fh)

    evaluator = MultiLabelEvaluater(
        model,
        device=device,
        force_encoding=args.r2d2_mode=='forced',
        tokenizer=tokenizer,
        logger=logger
    )

    if args.task == 'NER':
        evaluator.eval_atis_ner_fixed(
            dev_dataloader,
            output_dir=args.output_dir,
            span_bucket=args.span_bucket,
        )
    else:
        evaluator.eval(
            dev_dataloader,
            train_dataloader=train_dataloader
        )