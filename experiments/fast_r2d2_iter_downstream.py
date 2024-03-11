# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Qingyang Zhu

from multiprocessing import pool
from functools import reduce

import os
from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.fast_r2d2_iter_attn import FastR2D2Plus
from model.topdown_parser import LSTMParser, TransformerParser
from model.fast_r2d2_functions import force_contextualized_inside_outside, force_encode
from utils.model_loader import load_model
from utils.tree_utils import find_span_in_tree, flatten_trees
import logging


class FastR2D2IterClassification(nn.Module):
    """

        iterative up-and-down with top-down parser
        base model: fast_r2d2_iter.py

    """
    def __init__(self, config, label_num, transformer_parser=False, pretrain_dir=None, model_loss=False,
                 share=False):
        super().__init__()
        if share:
            from model.fast_r2d2_iter_attn_share import FastR2D2Plus as FastR2D2PlusShare
            self.r2d2 = FastR2D2PlusShare(config)
        else:
            self.r2d2 = FastR2D2Plus(config)
        if not transformer_parser:
            self.parser = LSTMParser(config)
        else:
            self.parser = TransformerParser(config)
        # load pretrained model
        if pretrain_dir is not None:
            model_path = os.path.join(pretrain_dir, 'model.bin') 
            self.r2d2.from_pretrain(model_path, strict=True)
            parser_path = os.path.join(pretrain_dir, 'parser.bin') 
            load_model(self.parser, parser_path)
            logging.info('FastR2D2IterClassification load pretrained model successfully')

        self.classifier = nn.Linear(config.hidden_size, label_num)
        
        self.label_num = label_num
        self.model_loss = model_loss
        
    def from_checkpoint(self, model_path):
        load_model(self, model_path)
        self.r2d2._tie_weights()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None,
                tgt_ids: torch.Tensor = None,
                noise_coeff: float = 1.0):
        if tgt_ids is not None and not torch.all(tgt_ids == -1):
            s_indices = self.parser(input_ids, attention_mask, noise_coeff=noise_coeff)
            results = self.r2d2(input_ids, tgt_ids=tgt_ids, masks=attention_mask, merge_trajectory=s_indices, 
                                pairwise=True, recover_tree=True) 
            mlm_loss = results['loss']
            if torch.all(torch.isnan(mlm_loss)):
                mlm_loss.fill_(0.0)
            pooling_embedding = results['cls_embedding']
            logits = self.classifier(pooling_embedding)
            target_tree = results['trees'][-1]
            kl_loss = self.parser(input_ids, attention_mask,
                                  split_masks=target_tree['split_masks'],
                                  split_points=target_tree['split_points'])
        else:
            outputs = force_contextualized_inside_outside(input_ids, attention_mask, self.parser, 
                                                          self.r2d2, self.r2d2.device, pairwise=False)
            pooling_embedding = outputs[:, 0, :]
            logits = self.classifier(pooling_embedding)
            mlm_loss = torch.zeros((1,), device=self.r2d2.device)
            kl_loss = torch.zeros((1,), device=self.r2d2.device)

        if self.training:
            
            loss = F.cross_entropy(logits, labels)
            return {"loss": [loss, mlm_loss, kl_loss]}
        else:
            ret_results = {}
            ret_results["predict"] = F.softmax(logits, dim=-1)
            return ret_results # probs + trees for eval 
        
class FastR2D2IterSpanClassification(FastR2D2IterClassification):
    def __init__(self, config, label_num, transformer_parser=False, pretrain_dir=None, 
                 model_loss=False, finetune_parser=True, num_repr=1,
                 tokenizer=None, criteria='bce', share=False):
        super().__init__(config, label_num, transformer_parser, pretrain_dir, model_loss, share)

        self.finetune_parser = finetune_parser
        self.num_repr = num_repr
        self.tokenizer = tokenizer
        self.criteria = criteria
        logging.info(f"training criterion is {self.criteria}")
        self.training_criterion = nn.BCEWithLogitsLoss()
        self.classifier = nn.Sequential(
                            nn.Linear(num_repr * config.hidden_size, config.intermediate_size),
                            nn.Tanh(),
                            nn.LayerNorm(config.intermediate_size),
                            nn.Dropout(0.2),
                            nn.Linear(config.intermediate_size, label_num)
                            )# classifier for span task (refer to work at TTIC)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs):
        if isinstance(input_ids, dict): #
            batch_dict = input_ids
            input_ids = batch_dict['subwords']['r2d2']
            targets = batch_dict.get('targets')
            if targets:
                targets = targets['r2d2']
            parser_ids = batch_dict.get('parser_subwords')
            if parser_ids:
                parser_ids = parser_ids['r2d2']
            attention_mask = (input_ids != self.tokenizer.pad_token_id).int()
            spans_1 = batch_dict['spans1']['r2d2']
            if batch_dict['spans2'] != {}:
                spans_2 = batch_dict['spans2']['r2d2']
            else:
                spans_2 = None
            atom_spans = batch_dict['atom_spans']
        else:
            atom_spans = kwargs.get('atom_spans') 
            spans_1 = kwargs.get('spans1')
            spans_2 = kwargs.get('spans2')
        B = len(spans_1)
        query_batch_idx = [[i]*self.num_repr
                           for i in range(B)
                           for _ in range(len(spans_1[i]))]
        query_batch_idx = reduce(lambda xs, x: xs + x, query_batch_idx, [])

        if not self.finetune_parser:
            self.parser.eval()
            
        if targets is not None and not torch.all(targets == -1):
            s_indices = self.parser(parser_ids, attention_mask, atom_spans=atom_spans) 

            results = self.r2d2(input_ids, 
                                tgt_ids=targets, # added for MLM
                                masks=attention_mask, 
                                merge_trajectory=s_indices,
                                atom_spans=atom_spans,
                                recover_tree=True)
            span_embeddings = results['contextualized_embeddings'] # B x num(span) x E
            mlm_loss = results['loss']
            if torch.all(torch.isnan(mlm_loss)):
                mlm_loss.fill_(0.0)
            target_tree = results['trees'][-1]
            kl_loss = self.parser(input_ids, attention_mask, 
                                  split_masks=target_tree['split_masks'],
                                  split_points=target_tree['split_points'])
            # span tasks
            tree_batch = results['trees'][0] # List[PyNode]
            span_cache_indices = [] # 

            # only used to mark each node its corresponding index during the flattening process
            flatten_nodes = flatten_trees(tree_batch)

            # find tgt spans in structured tree nodes
            for i, (root, span1) in enumerate(zip(tree_batch, spans_1)):
                # 'span1': [[st1, ed1], [st2, ed2], ...]]
                # 'span2': [...]

                span2 = spans_2[i] if spans_2 else None

                for j, span_1 in enumerate(span1):
                    st_1, ed_1 = span_1
                    tgt_node_1 = find_span_in_tree(root, st_1, ed_1)
                    cache_id_1 = tgt_node_1.flatten_id
                    tgt_node_2 = None
                    # if span1 not found, no need for span2
                    if span2:
                        assert self.num_repr == 2
                        assert len(span1) == len(span2)
                        span_2 = span2[j]
                        st_2, ed_2 = span_2
                        tgt_node_2 = find_span_in_tree(root, st_2, ed_2)
                        cache_id_2 = tgt_node_2.flatten_id
                    assert tgt_node_1 and ( tgt_node_2 is not None or self.num_repr==1 )

                    span_cache_indices.append(cache_id_1)
                    if self.num_repr == 2:
                        span_cache_indices.append(cache_id_2)
            # span_cache_indices = [span1, (span2,) span1, (span2,) ... ] (flattened)

            # retrieve all the span embeddings needed from cache
            tgt_span_embeddings = span_embeddings[query_batch_idx, span_cache_indices, :] # num_span x E
            repr_dimension = tgt_span_embeddings.shape[1]
            tgt_span_embeddings = torch.reshape(tgt_span_embeddings, (-1, self.num_repr * repr_dimension))
            preds = self.classifier(tgt_span_embeddings)
            
        else:
            flatten_nodes_batch = []
            span_embeddings, s_indices = force_contextualized_inside_outside(input_ids, attention_mask, self.parser, 
                                                          self.r2d2, self.r2d2.device, pairwise=False,
                                                          flatten_nodes_batch=flatten_nodes_batch,
                                                          atom_spans=atom_spans)
            span_emb_indices = []
            # find tgt spans in flatten tree nodes
            for i, span1 in enumerate(spans_1):
                span2 = spans_2[i] if spans_2 else None
                for j, span_1 in enumerate(span1):
                    st_1, ed_1 = span_1
                    for k, node in enumerate(flatten_nodes_batch[i]):
                        node_i, node_j = node.i, node.j
                        if st_1 == node_i and ed_1 == node_j:
                            span_emb_indices.append(k) # hit
                            break
                    else:
                        l = []
                        for node in flatten_nodes_batch[i]:
                            l.append((node.i, node.j))
                        torch.set_printoptions(profile='full')
                        logging.info(f"input ids {input_ids[i]}")
                        logging.info(f"sentence length {input_ids[i].shape[0]}")
                        logging.info(f"atom spans {atom_spans[i]}")
                        logging.info(f"s_indices {s_indices[i]}")
                        logging.info(f"parsed tree {l}")
                        assert False, f"Cannot find target span1 {span_1}"
                    if span2:
                        assert self.num_repr == 2
                        assert len(span1) == len(span2)
                        span_2 = span2[j]
                        st_2, ed_2 = span_2
                        for k, node in enumerate(flatten_nodes_batch[i]):
                            node_i, node_j = node.i, node.j
                            if st_2 == node_i and ed_2 == node_j:
                                span_emb_indices.append(k) # hit
                                break
                        else:
                            l = []
                            for node in flatten_nodes_batch[i]:
                                l.append((node.i, node.j))
                            assert False, f"Cannot find target span2 {span_2} in the parsed tree {l} of sentence of length {input_ids[0].shape[0]}"
                            
            tgt_span_embeddings = span_embeddings[query_batch_idx, span_emb_indices, :] # num_span x E
            repr_dimension = tgt_span_embeddings.shape[1]
            tgt_span_embeddings = torch.reshape(tgt_span_embeddings, (-1, self.num_repr * repr_dimension))
            preds = self.classifier(tgt_span_embeddings)
                      
            mlm_loss = torch.zeros((1,), device=self.r2d2.device)
            kl_loss = torch.zeros((1,), device=self.r2d2.device)
            
            target_tree = None

        return {'preds':preds, 'model_loss':[mlm_loss, kl_loss], 'trees_dict':target_tree}
    
class FastR2D2SpanClassification(nn.Module):
    def __init__(self, config, label_num, pretrain_dir=None, num_repr=1, tokenizer=None, finetune_parser=True, criteria='ce') -> None:
        super().__init__()
        self.num_repr = num_repr
        self.tokenizer = tokenizer
        self.label_num = label_num
        from model.fast_r2d2 import FastR2D2
        self.r2d2 = FastR2D2(config)
        self.parser = LSTMParser(config)
        if pretrain_dir is not None:
            model_path = os.path.join(pretrain_dir, 'model.bin')
            self.r2d2.from_pretrain(model_path, strict=True)
            parser_path = os.path.join(pretrain_dir, 'parser.bin')
            load_model(self.parser, parser_path)
            logging.info('FastR2D2Classification load pretrained model successfully')
        self.finetune_parser = finetune_parser
        self.classifier = nn.Sequential(
                            nn.Linear(num_repr * config.hidden_size, config.intermediate_size),
                            nn.Tanh(),
                            nn.LayerNorm(config.intermediate_size),
                            nn.Dropout(0.2),
                            nn.Linear(config.intermediate_size, label_num)
                            )# classifier for span task (refer to work at TTIC)
        self.criteria = criteria
        self.training_criterion = nn.BCEWithLogitsLoss()
            
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs):
        if isinstance(input_ids, dict):
            batch_dict = input_ids
            input_ids = batch_dict['subwords']['fastr2d2']
            targets = batch_dict.get('targets')
            if targets:
                targets = targets['fastr2d2']
            parser_ids = batch_dict.get('parser_subwords')
            if parser_ids:
                parser_ids = parser_ids['fastr2d2']
            attention_mask = (input_ids != self.tokenizer.pad_token_id).int()
            spans_1 = batch_dict['spans1']['fastr2d2']
            if batch_dict['spans2'] != {}:
                spans_2 = batch_dict['spans2']['fastr2d2']
            else:
                spans_2 = None
            atom_spans = batch_dict['atom_spans']
        else:
            atom_spans = kwargs.get('atom_spans') 
            spans_1 = kwargs.get('spans1')
            spans_2 = kwargs.get('spans2')
        B = len(spans_1)
        query_batch_idx = [[i]*self.num_repr
                           for i in range(B)
                           for _ in range(len(spans_1[i]))]
        query_batch_idx = reduce(lambda xs, x: xs + x, query_batch_idx, [])

        if not self.finetune_parser:
            self.parser.eval()
            
        if targets is not None and not torch.all(targets == -1):
            s_indices = self.parser(parser_ids, attention_mask, atom_spans=atom_spans) 

            results = self.r2d2(input_ids, 
                                tgt_ids=targets, # added for MLM
                                masks=attention_mask, 
                                merge_trajectories=s_indices,
                                atom_spans=atom_spans,
                                recover_tree=True)
            span_embeddings = results['tensor_cache'] # B x num(span) x E
            lm_loss = results['loss']
            if torch.all(torch.isnan(lm_loss)):
                lm_loss.fill_(0.0)
            target_tree = results['trees'][-1]
            kl_loss = self.parser(input_ids, attention_mask, 
                                  split_masks=target_tree['split_masks'],
                                  split_points=target_tree['split_points'])
            # span tasks
            tree_batch = results['trees'][0] # List[PyNode]
            span_cache_indices = [] 

            # find tgt spans in structured tree nodes
            for i, (root, span1) in enumerate(zip(tree_batch, spans_1)):
                # 'span1': [[st1, ed1], [st2, ed2], ...]]
                # 'span2': [...]

                span2 = spans_2[i] if spans_2 else None

                for j, span_1 in enumerate(span1):
                    st_1, ed_1 = span_1
                    tgt_node_1 = find_span_in_tree(root, st_1, ed_1)
                    cache_id_1 = tgt_node_1.cache_id
                    tgt_node_2 = None
                    # if span1 not found, no need for span2
                    if span2:
                        assert self.num_repr == 2
                        assert len(span1) == len(span2)
                        span_2 = span2[j]
                        st_2, ed_2 = span_2
                        tgt_node_2 = find_span_in_tree(root, st_2, ed_2)
                        cache_id_2 = tgt_node_2.cache_id
                    assert tgt_node_1 and ( tgt_node_2 is not None or self.num_repr==1 )

                    span_cache_indices.append(cache_id_1)
                    if self.num_repr == 2:
                        span_cache_indices.append(cache_id_2)
            # span_cache_indices = [span1, (span2,) span1, (span2,) ... ] (flattened)

            # retrieve all the span embeddings needed from cache
            tgt_span_embeddings = span_embeddings[span_cache_indices, :]  # expect: num_span *  E
            repr_dimension = tgt_span_embeddings.shape[1]
            tgt_span_embeddings = torch.reshape(tgt_span_embeddings, (-1, self.num_repr * repr_dimension))
            preds = self.classifier(tgt_span_embeddings)
            
        else:
            _, span_embeddings, tree_batch, _ = force_encode(self.parser, self.r2d2, input_ids, attention_mask, atom_spans)
            span_emb_indices = []
            # find tgt spans in flatten tree nodes
            for i, (root, span1) in enumerate(zip(tree_batch, spans_1)):
                # 'span1': [[st1, ed1], [st2, ed2], ...]]
                # 'span2': [...]

                span2 = spans_2[i] if spans_2 else None

                for j, span_1 in enumerate(span1):
                    st_1, ed_1 = span_1
                    tgt_node_1 = find_span_in_tree(root, st_1, ed_1)
                    cache_id_1 = tgt_node_1.cache_id
                    tgt_node_2 = None
                    # if span1 not found, no need for span2
                    if span2:
                        assert self.num_repr == 2
                        assert len(span1) == len(span2)
                        span_2 = span2[j]
                        st_2, ed_2 = span_2
                        tgt_node_2 = find_span_in_tree(root, st_2, ed_2)
                        cache_id_2 = tgt_node_2.cache_id
                    assert tgt_node_1 and ( tgt_node_2 is not None or self.num_repr==1 )

                    span_emb_indices.append(cache_id_1)
                    if self.num_repr == 2:
                        span_emb_indices.append(cache_id_2)
                            
            tgt_span_embeddings = span_embeddings[span_emb_indices, :] # num_span x E
            repr_dimension = tgt_span_embeddings.shape[1]
            tgt_span_embeddings = torch.reshape(tgt_span_embeddings, (-1, self.num_repr * repr_dimension))
            preds = self.classifier(tgt_span_embeddings)
                      
            lm_loss = torch.zeros((1,), device=self.r2d2.device)
            kl_loss = torch.zeros((1,), device=self.r2d2.device)
            
            target_tree = None

        return {'preds':preds, 'model_loss':[lm_loss, kl_loss], 'trees_dict':target_tree}