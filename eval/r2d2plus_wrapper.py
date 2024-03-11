# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

import torch
import os
from transformers import AutoTokenizer, AutoConfig
from data_structure.r2d2_tree import PyNode
from data_structure.syntax_tree import BinaryNode
from utils.misc import get_sentence_from_words, _align_spans
from utils.model_loader import load_model
from model.topdown_parser import LSTMParser, TransformerParser
from eval.tree_wrapper import TreeDecoderWrapper
from utils.tree_utils import get_tree_from_merge_trajectory, get_tree_from_merge_trajectory_in_word


class R2D2ParserWrapper(TreeDecoderWrapper):
    def __init__(self, 
                 config_path, 
                 model_cls, 
                 model_path, 
                 parser_path,
                 parser_only, 
                 sep_word, 
                 device, 
                 in_word=False):
        config_dir = os.path.dirname(config_path)
        config = AutoConfig.from_pretrained(config_path)
        self._model = model_cls(config)
        self._parser = LSTMParser(config)
        self._sep_word = sep_word
        self._model.from_pretrain(model_path)
        self._parser_only = parser_only
        self._in_word = in_word
        load_model(self._parser, parser_path)
        
        self._model.to(device)
        self._parser.to(device)


        self._device = device
        self._model.eval()
        self._parser.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(config_dir, config=config, use_fast=True)

    def _transfer_to_binary_node(self, node: PyNode, tokens, atom_spans=None):
        if node.is_leaf:
            pos = node.pos #if indices_mapping is None else indices_mapping[node.pos]
            return BinaryNode(None, None, pos, tokens[pos])
        else:
            return BinaryNode(self._transfer_to_binary_node(node.left, tokens, atom_spans),
                              self._transfer_to_binary_node(node.right, tokens, atom_spans),
                              None, None)
            
    def _transfer_to_binary_node_with_indices_mapping(self, node: PyNode, tokens, atom_spans, indices_mapping):
        if node.is_leaf or [node.i, node.j] in atom_spans:
            pos = indices_mapping[node.pos]
            return BinaryNode(None, None, pos, tokens[pos])
        else:
            return BinaryNode(self._transfer_to_binary_node_with_indices_mapping(node.left, tokens, atom_spans, indices_mapping),
                              self._transfer_to_binary_node_with_indices_mapping(node.right, tokens, atom_spans, indices_mapping),
                              None, None)

    def _parse(self, input_ids, attention_mask, atom_spans=None):
        return self._parser(input_ids, attention_mask, atom_spans=atom_spans)

    def __call__(self, tokens):
        sentence, spans = get_sentence_from_words(tokens, self._sep_word)
        outputs = self._tokenizer.encode_plus(sentence,
                                              add_special_tokens=False,
                                              return_offsets_mapping=True)
        new_spans = outputs['offset_mapping']
        word_starts, word_ends = _align_spans(spans, new_spans)
        atom_spans = [] # minimal span should be a whole word
        indices_mapping = [0] * len(outputs['input_ids'])
        for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
            if ed > st:
                atom_spans.append([st, ed])
            for idx in range(st, ed + 1):
                indices_mapping[idx] = pos

        if not self._in_word:
            model_inputs = {"input_ids": torch.tensor([outputs['input_ids']]).to(self._device),
                            "masks": torch.tensor([outputs['attention_mask']]).to(self._device)}
            parser_inputs = {
                "input_ids": torch.tensor([outputs['input_ids']]).to(self._device),
                "attention_mask": torch.tensor([outputs['attention_mask']]).to(self._device)
            }
        else:
            model_inputs = {"input_ids": torch.tensor([outputs['input_ids']]).to(self._device),
                            "masks": torch.tensor([outputs['attention_mask']]).to(self._device),
                            "atom_spans": [atom_spans]}
            parser_inputs = {"input_ids": torch.tensor([outputs['input_ids']]).to(self._device),
                            "attention_mask": torch.tensor([outputs['attention_mask']]).to(self._device),
                            "atom_spans": [atom_spans]}
        with torch.no_grad():
            if not self._in_word:
                merge_trajectories = self._parse(**parser_inputs)
                if self._parser_only:
                    root, tree_expr = get_tree_from_merge_trajectory(merge_trajectories[0], len(outputs['input_ids']),
                                                self._tokenizer.convert_ids_to_tokens(outputs['input_ids']))
                else:
                    results = self._model(**model_inputs, merge_trajectory=merge_trajectories.clone(), recover_tree=True)
            
                    root = results['trees'][0][0]

                binary_root = self._transfer_to_binary_node(root,
                                                            self._tokenizer.convert_ids_to_tokens(outputs['input_ids']))
            else:
                merge_trajectories = self._parse(**parser_inputs)
                if self._parser_only:
                    root, _ = get_tree_from_merge_trajectory_in_word(merge_trajectories[0], len(outputs['input_ids']), 
                                                                     atom_spans, indices_mapping, tokens)
                    binary_root = self._transfer_to_binary_node(root, tokens, atom_spans)
                else:
                    results = self._model(**model_inputs, merge_trajectory=merge_trajectories.clone(), recover_tree=True)
            
                    root = results['trees'][0][0]
                    binary_root = self._transfer_to_binary_node_with_indices_mapping(root, tokens, atom_spans, indices_mapping)
                    
            return binary_root


    def print_binary_ptb(self, tokens):
        tree = self(tokens)
        return tree.to_tree(ptb=True)

    def print_latex_tree(self, tokens):
        tree = self(tokens)
        return tree.to_latex_tree()

