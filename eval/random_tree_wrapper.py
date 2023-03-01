# coding=utf-8
# Copyright (c) 2021 Ant Group

from transformers import BertTokenizerFast
from eval.eval_tools import get_sentence_from_words, _align_spans, to_binary_root
from eval.tree_wrapper import TreeDecoderWrapper
import numpy as np


class VannilaRandomTreeDecoder(TreeDecoderWrapper):
    def __init__(self):
        pass

    def __call__(self, tokens):
        tokenized_text = [_ for _ in tokens]
        tokenized_copy = [_ for _ in tokens]
        while len(tokenized_text) > 2:
            merge_idx = int(np.random.rand() * (len(tokenized_text) - 1))
            left = tokenized_text.pop(merge_idx)
            right = tokenized_text.pop(merge_idx)
            tokenized_text.insert(merge_idx, [left, right])
        if len(tokenized_text) == 2:
            tree, _ = to_binary_root(tokenized_copy, tokenized_text, start=0)
            return tree, list(range(len(tokens)))
        else:
            return None, None

    def print_binary_ptb(self, tokens) -> str:
        tree, _ = self(tokens)
        return tree.to_tree(ptb=True)


class VannilaLeftChainTreeDecoder(TreeDecoderWrapper):
    def __init__(self):
        pass

    def __call__(self, tokens):
        tokenized_text = [_ for _ in tokens]
        tokenized_copy = [_ for _ in tokens]
        while len(tokenized_text) > 2:
            merge_idx = 0
            left = tokenized_text.pop(merge_idx)
            right = tokenized_text.pop(merge_idx)
            tokenized_text.insert(merge_idx, [left, right])
        if len(tokenized_text) == 2:
            tree, _ = to_binary_root(tokenized_copy, tokenized_text, start=0)
            return tree, list(range(len(tokens)))
        else:
            return None, None

    def print_binary_ptb(self, tokens) -> str:
        tree, _ = self(tokens)
        return tree.to_tree(ptb=True)


class VannilaRightChainTreeDecoder(TreeDecoderWrapper):
    def __init__(self):
        pass

    def __call__(self, tokens):
        tokenized_text = [_ for _ in tokens]
        tokenized_copy = [_ for _ in tokens]
        while len(tokenized_text) > 2:
            merge_idx = len(tokenized_text) - 2
            left = tokenized_text.pop(merge_idx)
            right = tokenized_text.pop(merge_idx)
            tokenized_text.insert(merge_idx, [left, right])
        if len(tokenized_text) == 2:
            tree, _ = to_binary_root(tokenized_copy, tokenized_text, start=0)
            return tree, list(range(len(tokens)))
        else:
            return None, None

    def print_binary_ptb(self, tokens) -> str:
        tree, _ = self(tokens)
        return tree.to_tree(ptb=True)


class RandomBottomUpTreeDecoder(TreeDecoderWrapper):
    def __init__(self, vocab_path, sep=' '):
        self._tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
        self._sep = sep

    def __call__(self, sents):
        sentence, spans = get_sentence_from_words(sents, self._sep)
        outputs = self._tokenizer.encode_plus(sentence,
                                              add_special_tokens=False,
                                              return_offsets_mapping=True)
        new_spans = outputs['offset_mapping']
        word_starts, word_ends = _align_spans(spans, new_spans)
        indices_mapping = [0] * (max(word_ends) + 1)
        for token_i, (st, ed) in enumerate(zip(word_starts, word_ends)):
            for pos_i in range(st, ed + 1):
                indices_mapping[pos_i] = token_i
        tokenized_text = self._tokenizer.tokenize(sentence)
        tokenized_copy = [_ for _ in tokenized_text]
        while len(tokenized_text) > 2:
            merge_idx = int(np.random.rand() * (len(tokenized_text) - 1))
            left = tokenized_text.pop(merge_idx)
            right = tokenized_text.pop(merge_idx)
            tokenized_text.insert(merge_idx, [left, right])
        if len(tokenized_text) == 2:
            tree, _ = to_binary_root(tokenized_copy, tokenized_text, start=0)
            return tree, indices_mapping
        else:
            return None, None


class RandomTopDownTreeDecoder(TreeDecoderWrapper):
    def __init__(self, vocab_path, sep=' '):
        self._tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
        self._sep = sep

    def _split(self, tokens):
        if len(tokens) >= 2:
            split_idx = int(np.random.rand() * (len(tokens) - 1))
            left = self._split(tokens[0:split_idx + 1])
            right = self._split(tokens[split_idx + 1:])
            return [left, right]
        else:
            assert isinstance(tokens[0], str)
            return tokens[0]

    def __call__(self, sents):
        sentence, spans = get_sentence_from_words(sents, self._sep)
        outputs = self._tokenizer.encode_plus(sentence,
                                              add_special_tokens=False,
                                              return_offsets_mapping=True)
        new_spans = outputs['offset_mapping']
        word_starts, word_ends = _align_spans(spans, new_spans)
        indices_mapping = [0] * (max(word_ends) + 1)
        for token_i, (st, ed) in enumerate(zip(word_starts, word_ends)):
            for pos_i in range(st, ed + 1):
                indices_mapping[pos_i] = token_i
        tokenized_text = self._tokenizer.tokenize(sentence)
        tokenized_copy = [_ for _ in tokenized_text]
        tokenized_text = self._split(tokenized_text)
        if len(tokenized_text) == 2:
            tree, _ = to_binary_root(tokenized_copy, tokenized_text, start=0)
            return tree, indices_mapping
        else:
            return None, None


class RightChainDecoder(TreeDecoderWrapper):
    def __init__(self, vocab_path, sep=' '):
        self._tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
        self._sep = sep

    def __call__(self, sents):
        sentence, spans = get_sentence_from_words(sents, self._sep)
        outputs = self._tokenizer.encode_plus(sentence,
                                              add_special_tokens=False,
                                              return_offsets_mapping=True)
        new_spans = outputs['offset_mapping']
        word_starts, word_ends = _align_spans(spans, new_spans)
        indices_mapping = [0] * (max(word_ends) + 1)
        for token_i, (st, ed) in enumerate(zip(word_starts, word_ends)):
            for pos_i in range(st, ed + 1):
                indices_mapping[pos_i] = token_i
        tokenized_text = self._tokenizer.tokenize(sentence)
        tokenized_copy = [_ for _ in tokenized_text]
        while len(tokenized_text) > 2:
            merge_idx = len(tokenized_text) - 2
            left = tokenized_text.pop(merge_idx)
            right = tokenized_text.pop(merge_idx)
            tokenized_text.insert(merge_idx, [left, right])
        if len(tokenized_text) == 2:
            tree, _ = to_binary_root(tokenized_copy, tokenized_text, start=0)
            return tree, indices_mapping
        else:
            return None, None

    def print_binary_ptb(self, tokens) -> str:
        pass


class LeftChainDecoder(TreeDecoderWrapper):
    def __init__(self, vocab_path, sep=' '):
        self._tokenizer = BertTokenizerFast.from_pretrained(vocab_path)
        self._sep = sep

    def __call__(self, sents):
        sentence, spans = get_sentence_from_words(sents, self._sep)
        outputs = self._tokenizer.encode_plus(sentence,
                                              add_special_tokens=False,
                                              return_offsets_mapping=True)
        new_spans = outputs['offset_mapping']
        word_starts, word_ends = _align_spans(spans, new_spans)
        indices_mapping = [0] * (max(word_ends) + 1)
        for token_i, (st, ed) in enumerate(zip(word_starts, word_ends)):
            for pos_i in range(st, ed + 1):
                indices_mapping[pos_i] = token_i
        tokenized_text = self._tokenizer.tokenize(sentence)
        tokenized_copy = [_ for _ in tokenized_text]
        while len(tokenized_text) > 2:
            merge_idx = 0
            left = tokenized_text.pop(merge_idx)
            right = tokenized_text.pop(merge_idx)
            tokenized_text.insert(merge_idx, [left, right])
        if len(tokenized_text) == 2:
            tree, _ = to_binary_root(tokenized_copy, tokenized_text, start=0)
            return tree, indices_mapping
        else:
            return None, None

    def print_binary_ptb(self, tokens) -> str:
        pass
