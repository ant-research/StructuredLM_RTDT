# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xinyu Kong

import argparse
import json
import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
from reader.reader_factory import create_glue_dataset

from trainer.model_factory import create_classification_model
from sklearn.metrics import accuracy_score
from collections import deque


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
        metric=None,
        output_dir=None,
        shortcut_type="span",
    ):
        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.eval()
        pred_labels = []
        gold_labels = []
        output_data = []
        total = 0.
        precision = 0.
        with torch.no_grad():
            for _, inputs in enumerate(epoch_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                labels = inputs['labels']
                inputs['traverse_all'] = True
                with torch.no_grad():
                    results = self.model(**inputs)
                    probs = results['predict']
                
                span_total = 0.
                span_recall = 0.
                for input, label, node, root_node in zip(inputs['input_ids'], inputs["labels"], results['node_interpretabel'], results["roots"]):
                    span_total += 1
                    label = label[0]
                    node_queue = deque()
                    node_queue.append(root_node)
                    leaf_nodes_sorted = self.token_score_2(root_node, label)
                    if self.span_recall(input, root_node, label,data_loader.dataset.posi_shortcut, data_loader.dataset.neg_shortcut) is True:
                        span_recall += 1
                    # leaf_nodes.sort(reverse=True,key=lambda x:self.token_score(x.ancestor_scores,label))
                    if shortcut_type == "st":
                        leaf_nodes_sorted = leaf_nodes_sorted[:1]
                    else:
                        leaf_nodes_sorted = leaf_nodes_sorted[:4]
                    for node in leaf_nodes_sorted:
                        total += 1
                        if self.tokenizer.convert_ids_to_tokens(input[node.i:node.j+1])[0] in data_loader.dataset.posi_shortcut + data_loader.dataset.neg_shortcut:
                            precision += 1
                    data_interpretable = {"interpretable_span":[self.tokenizer.convert_ids_to_tokens(input[node.i:node.j+1]) for node in leaf_nodes_sorted[:4]],
                                            "sentence":self.tokenizer.convert_ids_to_tokens(input),
                                            "others":self.tokenizer.convert_ids_to_tokens(input[:node.i]) + \
                                            self.tokenizer.convert_ids_to_tokens(input[node.j+1:])}
                    output_data.append(data_interpretable)
                if isinstance(probs, torch.Tensor):
                    predict_labels = probs.argmax(dim=-1)
                    for pred_label in predict_labels:
                        pred_labels.append(pred_label.tolist())
                else:
                    for label_list in probs:
                        if len(label_list) == 1:
                            pred_labels.append(label_list[0])
                        else:
                            pred_labels.append(1)
                if isinstance(labels, torch.Tensor):
                    gold_labels.extend(labels.tolist())
                else:
                    gold_labels.extend([lb[0] for lb in labels])
        if metric is None:
            result = accuracy_score(pred_labels, gold_labels)
        else:
            result = metric.compute(predictions=np.array(pred_labels), references=np.array(gold_labels))
        if output_dir is not None:
            with open(os.path.join(output_dir,"interpretable_results.txt"),'w') as f:
                for line in output_data:
                    json.dump(line,f)
                    f.write('\n')
        self.logger.info(f'shortcut precision {precision/total}')
        self.logger.info(f'shortcut span reall  {span_recall/span_total}')
        self.logger.info(f'eval result {result}')
        return result
    
    def token_score(self, ancestor_scores, label_id):
        token_score = 1
        for score in ancestor_scores:
            token_score = token_score * (1 - score[label_id])
        token_score = 1 - pow(token_score, 1/len(ancestor_scores))
        return token_score
    
    def token_score_2(self, node, label):
        if node.is_leaf:
            return [node]
        if node.left.logits[label] >= node.right.logits[label]:
            nodes_1st = self.token_score_2(node.left, label)
            nodes_2nd = self.token_score_2(node.right, label)
        else:
            nodes_1st = self.token_score_2(node.right, label)
            nodes_2nd = self.token_score_2(node.left, label)
        nodes = nodes_1st + nodes_2nd
        return nodes
    
    def span_recall(self, input, root_node, label, posi_shortcut, neg_shortcut):
        node_queue = deque()
        node_queue.append(root_node)
        while len(node_queue) > 0:
            current_node = node_queue.popleft()
            sentence = self.tokenizer.convert_ids_to_tokens(input[current_node.i:current_node.j+1])
            if current_node.label == label and (sentence == posi_shortcut or sentence == neg_shortcut):
                return True
            if not current_node.is_leaf:
                node_queue.append(current_node.left)
                node_queue.append(current_node.right)
        return False

        


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
    cmd.add_argument("--max_batch_len", default=1000000, type=int)
    cmd.add_argument("--max_batch_size", default=32, type=int)
    cmd.add_argument("--model_name", required=True, type=str)
    cmd.add_argument("--empty_label_idx", default=-1, type=int)
    cmd.add_argument("--tree_path", default=None, required=False, type=str)
    cmd.add_argument("--mode", default="dev", required=False, type=str)
    cmd.add_argument("--shortcut_type", choices=["st", "span"], default="span")

    args = cmd.parse_args(sys.argv[1:])

    logging.getLogger().setLevel(logging.INFO)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    metric = None
    if args.task_type=="cola":
        from datasets import load_metric
        metric = load_metric("glue", TASK_MAPPING[args.task_type]) # None

    enable_dp = 'dp' in args.model_name.split('_')
    dataset = create_glue_dataset(tokenizer, enable_dp, args.task_type, args.glue_dir, 
                                  args.mode, args.max_batch_len, args.max_batch_size, empty_label_idx=args.empty_label_idx, 
                                  tree_path=args.tree_path, sampler='sequential',enable_shortcut=True if args.mode=="shortcut_test" else False,
                                  shortcut_type=args.shortcut_type)
    model = create_classification_model(args.model_name, dataset.model_type, args.config_path, len(dataset.labels), None, None)

    # if args.model_dir is not None:
    #     model_path = os.path.join(args.model_dir, f'model{args.turn}.bin')
    #     if hasattr(model, "load_model"):
    #         model.load_model(model_path)
    #     else:
    #         load_model(model, model_path)


    for i in range(0,10):
        model_path = os.path.join(args.model_dir, f'model{i}.bin')
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
            metric=metric,
            output_dir=args.model_dir,
            shortcut_type=args.shortcut_type
        )
