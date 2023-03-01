# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

import argparse
from re import A
import sys
import os
sys.path.append(os.getcwd())
from captum.attr import IntegratedGradients
from experiments.f1_evaluator import F1Evaluator
from reader.multi_label_reader import MultiLabelReader
from reader.sls_reader import SLSReader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from utils.model_loader import load_model
import torch
import logging
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from experiments.f1_evaluator import F1Evaluator
import torch.nn.functional as F
import numpy as np


class BertForMultiIntentWrapper(nn.Module):
    def __init__(self, pretrain_model_dir, num_labels) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(pretrain_model_dir)
        self.encoder = AutoModel.from_pretrained(pretrain_model_dir)
        
        self.cls = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.intermediate_size),# nn.Linear(self.config.input_dim, self.config.intermediate_size),
                                 nn.GELU(),
                                 nn.Linear(self.config.intermediate_size, num_labels))


    def forward(self, embeddings: torch.Tensor):
        # ASSUME [CLS] is provided in input_ids
        results = self.encoder(inputs_embeds=embeddings)
        hidden_states = results.last_hidden_state
        logits = self.cls(hidden_states[:, 0, :])
        return torch.sigmoid(logits)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser("Arguments for integrated gradients evaluator")
    cmd.add_argument("--pretrain_dir", default='data/bert_12_wiki_103')
    cmd.add_argument("--data_dir", default='data/ATIS')
    cmd.add_argument("--dataset_mode", default='test', type=str)
    cmd.add_argument("--save_path", type=str, default='data/save/bert_wiki103_atis_ner_20/model19.bin')
    cmd.add_argument("--output_path", type=str, default='data/save/bert_wiki103_atis_ner_20/')
    cmd.add_argument("--eval_step", type=int, default=1000)
    cmd.add_argument("--dataset", type=str,default='data/ATIS')
    cmd.add_argument("--domain", type=str,default='movie_eng')
    cmd.add_argument("--threshold", type=float, default=0.5)
    cmd.add_argument("--span_bucket", type=str, default='1,2,5', help="span bucket (1,2),(2,5),(5,)")

    args = cmd.parse_args()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.output_path, "ig_result_mask_{}.txt".format(args.threshold)), mode="a", encoding="utf-8")
    logger.addHandler(fh)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_dir)
    if args.dataset == 'sls':
        dataset = SLSReader(
            args.data_dir,
            tokenizer,
            batch_max_len=100000,
            batch_size=1,
            domain=args.domain,
            mode="test",
        )
    elif args.dataset == 'atis':
        dataset = MultiLabelReader(args.data_dir,
                                tokenizer,
                                batch_max_len=1000000,
                                batch_size=1,
                                task='NER',
                                mode=args.dataset_mode,
                                ig=True)
    model = BertForMultiIntentWrapper(args.pretrain_dir, len(dataset.labels))
    load_model(model, args.save_path)
    model.to(device)
    model.eval()

    ig = IntegratedGradients(model)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        collate_fn=dataset.collate_batch_bert,
    )

    epoch_iterator = tqdm(dataloader, desc="Iteration")
    label2idx = dataset.label2id_dict
    cls_bias = 5

    f1_evaluator = F1Evaluator(dataset.id2label_dict.keys(), list(map(lambda x:int(x),args.span_bucket.split(','))))
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        input_ids = inputs['input_ids']
        input_embeddings = model.encoder.embeddings(input_ids, token_type_ids=None, position_ids=None)
        results = model(input_embeddings)
        pred_labels = (results > 0.5).nonzero().to('cpu').data.numpy()
        offset_mapping = inputs['offset_mapping']
        entities = inputs['entities']
        mean_span_length = inputs['mean_span_length']
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].to('cpu').data.numpy())

        label_span_collector = f1_evaluator.create_label_span_collector(entities[0], label2idx, offset_mapping)
        for _, label_idx in pred_labels:
        # for label_idx in inputs['labels'][0]:
            model.zero_grad()
            
            target_label_id = torch.tensor([label_idx], device=device)
            input_embeddings = model.encoder.embeddings(input_ids, token_type_ids=None, position_ids=None)

            mask_ids = torch.zeros_like(input_ids).fill_(tokenizer.vocab['[MASK]'])
            mask_embeddings = model.encoder.embeddings(mask_ids, token_type_ids=None, position_ids=None)
            attributions_ig = ig.attribute(input_embeddings, baselines=mask_embeddings, target=target_label_id, n_steps=200)

            # attributions_ig = ig.attribute(input_embeddings, target=target_label_id, n_steps=200)
            scores = attributions_ig.sum(dim=-1).squeeze(-1)
            scores_norm = scores / torch.norm(scores)

            filtered_indices = (scores_norm > args.threshold).nonzero()
            filtered_pos = []
            for _, pos in filtered_indices.to('cpu').data.numpy():
                filtered_pos.append(pos - 1)  # [CLS] tokes one position

            if label_idx not in label_span_collector:
                label_span_collector[label_idx] = [None, [(-1, -1)]]
            label_span_collector[label_idx][0] = filtered_pos

        f1_evaluator.update(label_span_collector, mean_span_length[0])
        if step != 0 and step % args.eval_step == 0:
            f1_evaluator.print_results(labelidx2name=dataset.id2label_dict)

    logger.info(f'f1 {2 * f1_evaluator.f1_hit_total / (f1_evaluator.f1_pred_total + f1_evaluator.f1_true_total)}')
    logger.info(f'f1_mean {f1_evaluator.f1_mean}')
    for bucket_name, bucket in f1_evaluator.f1_bucket.items():
            logger.info(f'bucket: {bucket_name} f1: {bucket.f1} f1_mean: {bucket.f1_mean} ratio: {bucket.ratio} entity_count: {bucket.entity_count}')
    for label_idx, val in f1_evaluator.f1_recorder.items():
        if dataset.id2label_dict is None or label_idx not in dataset.id2label_dict:
            logger.info(f'label_idx: {label_idx}, f1: {val[0]}')
        else:
            logger.info(f'label: {dataset.id2label_dict[label_idx]}, f1: {val[0]}')