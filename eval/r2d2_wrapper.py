# coding=utf-8
# Copyright (c) 2021 Ant Group

import torch
from transformers import AutoTokenizer, AutoConfig

from data_structure.syntax_tree import BinaryNode
from eval.eval_tools import get_sentence_from_words, _align_spans
from eval.tree_wrapper import TreeDecoderWrapper
from data_structure.basic_structure import AtomicSpans


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