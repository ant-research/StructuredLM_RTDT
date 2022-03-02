# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

import torch
import os
from transformers import AutoTokenizer, AutoConfig
from data_structure.syntax_tree import BinaryNode
from utils.misc import get_sentence_from_words, _align_spans
from utils.model_loader import load_model
from model.topdown_parser import TopdownParser
from eval.tree_wrapper import TreeDecoderWrapper
from data_structure.basic_structure import AtomicSpans
from utils.tree_utils import get_tree_from_merge_trajectory, get_tree_from_merge_trajectory_in_word


class ChartModelWrapper(TreeDecoderWrapper):
    def __init__(self, config_path, vocab_path, model_path, sep_word, device, window_size, in_word, lstm=False):
        config = AutoConfig.from_pretrained(config_path)
        if not lstm:
            from model.r2d2 import R2D2
            self._model = R2D2(config, window_size)
        else:
            from model.r2d2_lstm import R2D2TreeLSTM
            self._model = R2D2TreeLSTM(config, window_size)
        self._sep_word = sep_word
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        trans_state_dict = {}
        for key, val in state_dict.items():
            key = key.replace('module.', '')
            trans_state_dict[key] = val
        self.in_word = in_word
        self._model.load_state_dict(trans_state_dict)
        self._model.to(device)
        self._device = device
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(vocab_path, config=config, use_fast=True)

    def _transfer_to_binary_node(self, model_node, tokens, force_spans = None):
        if len(model_node.candidates) == 0:
            return BinaryNode(None, None, model_node.pos, tokens[model_node.pos])
        else:
            for pair in model_node.candidates:
                left = pair.left
                right = pair.right
                if force_spans is None or (force_spans.is_valid_span(left.height, left.pos)
                    and force_spans.is_valid_span(right.height, right.pos)):
                    return BinaryNode(self._transfer_to_binary_node(left, tokens, force_spans),
                                      self._transfer_to_binary_node(right, tokens, force_spans),
                                      None, None)
            raise Exception('error in parsing')

    def __call__(self, tokens):
        sentence, spans = get_sentence_from_words(tokens, self._sep_word)
        outputs = self._tokenizer.encode_plus(sentence,
                                              add_special_tokens=False,
                                              return_offsets_mapping=True)
        new_spans = outputs['offset_mapping']
        word_starts, word_ends = _align_spans(spans, new_spans)
        indices_mapping = [0] * (max(word_ends) + 1)
        for token_i, (st, ed) in enumerate(zip(word_starts, word_ends)):
            for pos_i in range(st, ed + 1):
                indices_mapping[pos_i] = token_i
        force_spans = None
        if self.in_word:
            word_piece_spans = []
            word_piece_spans_token = []
            for token_i, (st, ed) in enumerate(zip(word_starts, word_ends)):
                for pos_i in range(st, ed + 1):
                    indices_mapping[pos_i] = token_i
                if st != ed:
                    word_piece_spans.append([st, ed])
                    word_piece_spans_token.append(tokens[token_i])
            force_spans = AtomicSpans(word_piece_spans, word_piece_spans_token)
            inputs = {"input_ids": torch.tensor([outputs['input_ids']]).to(self._device),
                      "attention_mask": torch.tensor([outputs['attention_mask']]).to(self._device),
                      "atom_spans": [AtomicSpans(word_piece_spans)]}
        else:
            inputs = {"input_ids": torch.tensor([outputs['input_ids']]).to(self._device),
                      "attention_mask": torch.tensor([outputs['attention_mask']]).to(self._device)}
        with torch.no_grad():
            loss, trees = self._model(**inputs)
        root = trees[0].root

        binary_root = self._transfer_to_binary_node(root,
                                                    self._tokenizer.convert_ids_to_tokens(outputs['input_ids']),
                                                    force_spans)
        if self.in_word:
            binary_root = binary_root.convert(force_spans)
            indices_mapping = list(range(len(tokens)))
        return binary_root, indices_mapping


    def print_binary_ptb(self, tokens):
        tree, _ = self(tokens)
        return tree.to_tree(ptb=True)


class R2D2ParserWrapper(TreeDecoderWrapper):
    def __init__(self, config_path, model_cls, model_path, parser_path, parser_only, sep_word, device, in_word=False):
        config_dir = os.path.dirname(config_path)
        config = AutoConfig.from_pretrained(config_path)
        self._model = model_cls(config)
        self._parser = TopdownParser(config)
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

    def _transfer_to_binary_node(self, node, tokens):
        if node.is_leaf:
            return BinaryNode(None, None, node.pos, tokens[node.pos])
        else:
            return BinaryNode(self._transfer_to_binary_node(node.left, tokens),
                              self._transfer_to_binary_node(node.right, tokens),
                              None, None)

    def __call__(self, tokens):
        sentence, spans = get_sentence_from_words(tokens, self._sep_word)
        outputs = self._tokenizer.encode_plus(sentence,
                                              add_special_tokens=False,
                                              return_offsets_mapping=True)
        new_spans = outputs['offset_mapping']
        word_starts, word_ends = _align_spans(spans, new_spans)
        atom_spans = []
        indices_mapping = [0] * len(outputs['input_ids'])
        for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
            if ed > st:
                atom_spans.append([st, ed])
            for idx in range(st, ed + 1):
                indices_mapping[idx] = pos

        if not self._in_word:
            inputs = {"input_ids": torch.tensor([outputs['input_ids']]).to(self._device),
                    "attention_mask": torch.tensor([outputs['attention_mask']]).to(self._device)}
        else:
            inputs = {"input_ids": torch.tensor([outputs['input_ids']]).to(self._device),
                    "attention_mask": torch.tensor([outputs['attention_mask']]).to(self._device),
                    "atom_spans": [atom_spans]}
        with torch.no_grad():
            if not self._in_word:
                merge_trajectories = self._parser(**inputs)
                if self._parser_only:
                    root, tree_expr = get_tree_from_merge_trajectory(merge_trajectories[0], len(outputs['input_ids']),
                                                self._tokenizer.convert_ids_to_tokens(outputs['input_ids']))
                else:
                    results = self._model(**inputs, merge_trajectories=merge_trajectories.clone(), recover_tree=True)
            
                    root = results['tables'][0].root.best_node

                binary_root = self._transfer_to_binary_node(root,
                                                            self._tokenizer.convert_ids_to_tokens(outputs['input_ids']))
            else:
                merge_trajectories = self._parser(**inputs)
                if self._parser_only:
                    root, _ = get_tree_from_merge_trajectory_in_word(merge_trajectories[0], len(outputs['input_ids']), 
                                                                             atom_spans, indices_mapping, tokens)

                else:
                    results = self._model(**inputs, merge_trajectories=merge_trajectories.clone(), recover_tree=True)
            
                    root = results['tables'][0].root.best_node

                binary_root = self._transfer_to_binary_node(root, tokens)
            return binary_root


    def print_binary_ptb(self, tokens):
        tree = self(tokens)
        return tree.to_tree(ptb=True)