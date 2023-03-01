
# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xinyu Kong

import argparse
from re import A
import sys
import os
sys.path.append(os.getcwd())
from captum.attr import IntegratedGradients
from reader.glue_reader import GlueReaderWithShortcut
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from utils.model_loader import load_model
import torch
import logging
import torch.nn as nn
from transformers import AutoConfig, AutoModel
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
    cmd.add_argument("--task_type", required=True, type=str, help="Specify the glue task")
    cmd.add_argument("--glue_dir", required=True, type=str, help="path to the directory of glue dataset")
    cmd.add_argument("--dataset_mode", default='test', type=str)
    cmd.add_argument("--save_path", type=str, default='data/save/bert_wiki103_atis_ner_20/model19.bin')
    cmd.add_argument("--output_path", type=str, default='data/save/bert_wiki103_atis_ner_20/')
    cmd.add_argument("--eval_step", type=int, default=1000)
    cmd.add_argument("--shortcut_type", choices=["st", "span"], default="span")
    

    args = cmd.parse_args()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.output_path, "ig_result_{}.txt"), mode="a", encoding="utf-8")
    logger.addHandler(fh)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_dir)
    dataset = GlueReaderWithShortcut(args.task_type,
                                    args.glue_dir,
                                    "shortcut_test",
                                    tokenizer,
                                    max_batch_len=1000000,
                                    max_batch_size=1,
                                    random=True,
                                    shortcut_type=args.shortcut_type)
    model = BertForMultiIntentWrapper(args.pretrain_dir, len(dataset.labels))
    for i in range(0,10):
        load_model(model, args.save_path.format(i))
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
        total = 0.
        precision = 0.
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            input_ids = inputs['input_ids']
            input_embeddings = model.encoder.embeddings(input_ids, token_type_ids=None, position_ids=None)
            results = model(input_embeddings)
            # pred_labels = (results > 0.5).nonzero().to('cpu').data.numpy()
            # tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].to('cpu').data.numpy())

            # # label_span_collector = f1_evaluator.create_label_span_collector(entities[0], label2idx, offset_mapping)
            # for _, label_idx in pred_labels:
            # for label_idx in inputs['labels'][0]:
            label_idx = torch.argmax(results).tolist()
            model.zero_grad()
            
            target_label_id = torch.tensor([label_idx], device=device)
            input_embeddings = model.encoder.embeddings(input_ids, token_type_ids=None, position_ids=None)
            attributions_ig = ig.attribute(input_embeddings, target=target_label_id, n_steps=200)
            scores = attributions_ig.sum(dim=-1).squeeze(-1)
            scores_norm = scores / torch.norm(scores)

            token_score_dict = dict(zip(input_ids.tolist()[0],scores_norm.tolist()[0]))
            token_sorted = sorted(token_score_dict.keys(),key=lambda s:token_score_dict[s],reverse=True)
            if args.shortcut_type == "st":
                token_sorted = token_sorted[:1]
            elif args.shortcut_type == "span":
                token_sorted = token_sorted[:4]
            for token in token_sorted:
                total += 1
                if tokenizer.convert_ids_to_tokens(token) in dataset.posi_shortcut + dataset.neg_shortcut:
                    precision += 1
        print("shortcut precision is {}".format(precision/total+1e-5))
        logger.info("shortcut precision is {}".format(precision/total+1e-5))