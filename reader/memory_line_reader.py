# coding=utf-8
# Copyright (c) 2021 Ant Group

from torch.utils.data import Dataset
import torch
import codecs
import copy
from transformers.tokenization_bert import BertTokenizer as Tokenizer
from typing import List, Dict
import random

from data_structure.basic_structure import AtomicSpans, Span
from utils.conll_utils import ConllReader
import logging


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
    )
    return mask


class BatchSelfRegressionLineDataset(Dataset):

    def __init__(self, path, tokenizer: Tokenizer, config, batch_max_len, batch_size,
                 min_len=2, max_line=1000, input_type='txt'):
        super().__init__()
        self._lines = []
        self._tokenizer = tokenizer
        self._batch_max_len = batch_max_len
        self._config = config
        self._batch_size = batch_size

        assert input_type in ['txt', 'ids']

        with codecs.open(path, mode='r', encoding='utf-8') as f:
            for _line in f:
                if input_type == 'txt':
                    tokens = self._tokenizer.tokenize(_line.strip())
                    if min_len < len(tokens) < self._batch_max_len:
                        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
                        self._lines.append(token_ids)
                elif input_type == 'ids':
                    token_ids = [int(t_id) for t_id in _line.strip().split()]
                    if min_len < len(token_ids) < self._batch_max_len:
                        self._lines.append(token_ids)
                if len(self._lines) > max_line > 0:
                    break
        self.shuffle()

    def batchify(self):
        logging.info('batchify')
        len_dict = {}
        for tokens in self._lines:
            arr = len_dict.get(len(tokens), [])
            arr.append(tokens)
            len_dict[len(tokens)] = arr
        len_keys = list(len_dict.keys())
        len_keys.sort(key=lambda x: x, reverse=True)
        rest_lines = len(self._lines)
        batches = []
        while rest_lines > 0:
            rest_len = self._batch_max_len
            current_batch = []
            while rest_len > 0 and len(current_batch) < self._batch_size:
                next_len = -1
                for key_len in len_keys:
                    if 0 < key_len <= rest_len and len(len_dict[key_len]) > 0:
                        next_len = key_len
                        break
                if next_len != -1:
                    assert len(len_dict) > 0
                    tokens = len_dict[next_len].pop()
                    current_batch.append(tokens)
                    rest_len -= next_len
                    rest_lines -= 1
                else:
                    break
            if len(current_batch) == 0:
                # no sentence to add
                break
            batches.append(current_batch)
        return batches

    def shuffle(self):
        random.shuffle(self._lines)
        self._batches = self.batchify()

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx):
        return self._batches[idx]

    def collate_batch(self, ids_batch: List[List[int]]) -> Dict[str, torch.Tensor]:
        assert len(ids_batch) == 1
        ids_batch = ids_batch[0]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            masked_input_ids = copy.deepcopy(input_ids)
            input_ids_batch.append(masked_input_ids + [self._tokenizer.pad_token_id] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))

        return {"input_ids": torch.tensor(input_ids_batch),
                "attention_mask": torch.tensor(mask_batch)}


class ConllDatset(Dataset):
    def __init__(self, conll_path, tokenizer: Tokenizer, max_len=128):
        conll_reader = ConllReader('tree', -1, logging)
        trees = conll_reader.from_conll_file(conll_path)

        self._items = []
        self._tokenizer = tokenizer
        for t in trees:
            text = ''.join(t.words)
            if len(text) > max_len:
                continue
            char_seq = []
            for c in text.strip():
                char_seq.append(c)
            ids = tokenizer.convert_tokens_to_ids(char_seq)
            # indices mapping
            indices_mapping = [0] * len(ids)
            iter_i = 0
            for w_idx, word in enumerate(t.words):
                for _ in word:
                    indices_mapping[iter_i] = w_idx
                    iter_i += 1
            self._items.append((ids, indices_mapping, t))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, item):
        return self._items[item]

    def collate_batch(self, batch) -> Dict[str, torch.Tensor]:
        lens = map(lambda a: len(a[0]), batch)
        input_max_len = max(lens)
        atomic_span_arr = []
        parents = []
        for input_ids, indices_mapping, t in batch:
            input_ids_batch = []
            mask_batch = []

            input_ids_batch.append(
                input_ids + [0] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))
            spans = []
            current_span = None
            prev_mapping_idx = -1
            for idx, mapping_idx in enumerate(indices_mapping):
                if prev_mapping_idx != mapping_idx:
                    # start a new span
                    current_span = Span(idx, idx)
                    spans.append(current_span)
                    prev_mapping_idx = mapping_idx
                else:
                    if current_span:
                        current_span.end = idx
            atomic_spans = AtomicSpans(spans)
            atomic_span_arr.append(atomic_spans)
            parents.append(t.parents)

        return {"input_ids": torch.tensor(input_ids_batch),
                "attention_mask": torch.tensor(mask_batch),
                "atom_spans": atomic_span_arr,
                "parents": parents}