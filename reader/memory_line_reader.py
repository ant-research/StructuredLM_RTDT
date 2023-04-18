# coding=utf-8
# Copyright (c) 2021 Ant Group


from torch.utils.data import Dataset
import torch
import copy
from typing import List, Dict
import random
from utils.misc import _align_spans, get_sentence_from_words
from abc import ABC, abstractmethod
import linecache
import logging


EMPTY_HISTORY = "[EMPTY]"
AGENT = "[AGENT]"
USER = "[USER]"
TOPIC = "[TOPIC]"


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


class InputItem:
    def __init__(self, ids, atom_spans=None, **kwargs) -> None:
        self.ids = ids
        self.atom_spans = atom_spans
        self.kwargs = kwargs

    def __getattr__(self, key):
        if key in self.kwargs:
            return self.kwargs[key]
        else:
            return None


class BatchByLengthDataset(Dataset, ABC):
    def __init__(self, 
                data_path_or_dir, 
                tokenizer, 
                batch_max_len, 
                max_batch_size,
                min_len=2,
                max_line=-1,
                random=False,
                **kwargs) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._batch_max_len = batch_max_len
        self._max_batch_size = max_batch_size
        self._min_len = min_len
        self._max_line = max_line
        self._random = random
        self._lines = self._load_dataset(data_path_or_dir, **kwargs)
        self._batches = self.shuffle(**kwargs)
    
    def _pre_shuffle(self, **kwargs):
        pass

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx):
        return self._batches[idx]

    def _batchify(self, items):
        if not self._random:
            logging.info("batchify")
            len_dict = {}
            for input_item in items:
                arr = len_dict.get(len(input_item.ids), [])
                arr.append(input_item)
                len_dict[len(input_item.ids)] = arr
            len_keys = list(len_dict.keys())
            len_keys.sort(key=lambda x: x, reverse=True)
            rest_lines = len(items)
            batches = []
            while rest_lines > 0:
                rest_len = self._batch_max_len
                current_batch = []
                while rest_len > 0 and len(current_batch) < self._max_batch_size:
                    next_len = -1
                    for key_len in len_keys:
                        if 0 < key_len <= rest_len and len(len_dict[key_len]) > 0:
                            next_len = key_len
                            break
                    if next_len != -1:
                        assert len(len_dict) > 0
                        item = len_dict[next_len].pop()
                        current_batch.append(item)
                        rest_len -= next_len
                        rest_lines -= 1
                    else:
                        break
                if len(current_batch) == 0:
                    # no sentence to add
                    break
                batches.append(current_batch)
            return batches  # [_ for _ in reversed(batches)]
        else:
            logging.info("batchify")
            batches = []
            current_batch = []
            current_len_sum = 0
            for item in items:
                if (current_len_sum + len(item.ids)) >= self._batch_max_len or \
                    len(current_batch) >= self._max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_len_sum = 0
                current_batch.append(item)
                current_len_sum += len(item.ids)
            if len(current_batch) > 0:
                batches.append(current_batch)
            return batches

    def shuffle(self, **kwargs):
        self._pre_shuffle(**kwargs)
        random.shuffle(self._lines)
        return self._batchify(self._lines)
        
    @abstractmethod
    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        pass

    @abstractmethod
    def collate_batch(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        pass


class BatchSelfRegressionLineDataset(BatchByLengthDataset):
    def __init__(self, path, tokenizer, batch_max_len, batch_size,
                 min_len=2, max_line=-1, random=False, **kwargs):
        '''
        params:
        random: True: for randomly batch sentences
                False: batch sentences in similar length
        '''
        super().__init__(path, tokenizer, batch_max_len, batch_size, 
                         min_len, max_line, random, **kwargs)

    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        input_type = kwargs['input_type']
        seperator = kwargs['seperator'] if 'seperator' in kwargs else None
        assert input_type in ["txt", "ids"]

        input_item_list = []
        lines = linecache.getlines(data_path_or_dir)
        for _line in lines:
            token_ids = None
            atom_spans = None
            if input_type == "txt":
                if seperator is None:
                    tokens = self._tokenizer.tokenize(_line.strip())
                    token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
                else:
                    try:
                        sentence, spans = get_sentence_from_words(_line.strip().split(seperator), seperator)
                        outputs = self._tokenizer.encode_plus(sentence,
                                                                add_special_tokens=False,
                                                                return_offsets_mapping=True)
                        new_spans = outputs['offset_mapping']
                        word_starts, word_ends = _align_spans(spans, new_spans)
                        atom_spans = []
                        for st, ed in zip(word_starts, word_ends):
                            if st != ed:
                                atom_spans.append([st, ed])
                        token_ids = outputs['input_ids']
                        atom_spans = atom_spans
                    except Exception:
                        pass
            elif input_type == "ids":
                parts = _line.strip().split('|')
                token_ids = [int(t_id) for t_id in parts[0].split()]
                # tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
                if len(parts) > 1:
                    spans = parts[1].split(';')
                    atom_spans = []
                    for span in spans:
                        vals = span.split(',')
                        if len(vals) == 2:
                            atom_spans.append([int(vals[0]), int(vals[1])])
            if self._min_len < len(token_ids) < self._batch_max_len:
                input_item_list.append(InputItem(token_ids, atom_spans))
            if len(input_item_list) > self._max_line > 0:
                break
        return input_item_list

    def _batchify(self, items):
        if not self._random:
            logging.info("batchify")
            len_dict = {}
            for input_item in items:
                arr = len_dict.get(len(input_item.ids), [])
                arr.append(input_item)
                len_dict[len(input_item.ids)] = arr
            len_keys = list(len_dict.keys())
            len_keys.sort(key=lambda x: x, reverse=True)
            rest_lines = len(items)
            batches = []
            while rest_lines > 0:
                rest_len = self._batch_max_len
                current_batch = []
                while rest_len > 0 and len(current_batch) < self._max_batch_size:
                    next_len = -1
                    for key_len in len_keys:
                        if 0 < key_len <= rest_len and len(len_dict[key_len]) > 0:
                            next_len = key_len
                            break
                    if next_len != -1:
                        assert len(len_dict) > 0
                        item = len_dict[next_len].pop()
                        current_batch.append(item)
                        rest_len -= next_len
                        rest_lines -= 1
                    else:
                        break
                if len(current_batch) == 0:
                    # no sentence to add
                    break
                batches.append(current_batch)
            return batches  # [_ for _ in reversed(batches)]
        else:
            logging.info("batchify")
            batches = []
            current_batch = []
            current_len_sum = 0
            for item in items:
                if (current_len_sum + len(item.ids)) >= self._batch_max_len or \
                    len(current_batch) >= self._max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_len_sum = 0
                current_batch.append(item)
                current_len_sum += len(item.ids)
            if len(current_batch) > 0:
                batches.append(current_batch)
            return batches


    def collate_batch(self, items: List[List[InputItem]]) -> Dict[str, torch.Tensor]:
        ids_batch = [item.ids for item in items[0]]
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            masked_input_ids = copy.deepcopy(input_ids)
            input_ids_batch.append(masked_input_ids + [self._tokenizer.pad_token_id] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))

        return {"input_ids": torch.tensor(input_ids_batch), "attention_mask": torch.tensor(mask_batch),
                "atom_spans": [item.atom_spans for item in items[0]]}


class HugeBatchSelfRegressionLineDataset(BatchByLengthDataset):
    def __init__(self, path, tokenizer, batch_max_len, batch_size,
                 min_len=2, max_line=-1, random=False, **kwargs):
        '''
        params:
        random: True: for randomly batch sentences
                False: batch sentences in similar length
        '''
        super().__init__(path, tokenizer, batch_max_len, batch_size, 
                         min_len, max_line, random, **kwargs)


    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        logging.info('start loading dataset')
        lines = linecache.getlines(data_path_or_dir)
        if self._max_line != -1:
            lines = lines[:self._max_line]
        logging.info('dataset loaded')
        results = []
        for _line in lines:
            parts = _line.strip().split('|')
            token_ids_len = len(parts[0].split())
            if token_ids_len >= self._min_len:
                results.append([token_ids_len, _line])
        return results


    def _batchify(self, items):
        if not self._random:
            logging.info("batchify")
            len_dict = {}
            for sent_len, sent_line in items:
                arr = len_dict.get(sent_len, [])
                arr.append(sent_line)
                len_dict[sent_len] = arr
            len_keys = list(len_dict.keys())
            len_keys.sort(key=lambda x: x, reverse=True)
            rest_lines = len(items)
            batches = []
            while rest_lines > 0:
                rest_len = self._batch_max_len
                current_batch = []
                while rest_len > 0 and len(current_batch) < self._max_batch_size:
                    next_len = -1
                    for key_len in len_keys:
                        if 0 < key_len <= rest_len and len(len_dict[key_len]) > 0:
                            next_len = key_len
                            break
                    if next_len != -1:
                        assert len(len_dict) > 0
                        item = len_dict[next_len].pop()
                        current_batch.append(item)
                        rest_len -= next_len
                        rest_lines -= 1
                    else:
                        break
                if len(current_batch) == 0:
                    # no sentence to add
                    break
                batches.append(current_batch)
            return batches  # [_ for _ in reversed(batches)]
        else:
            logging.info("batchify")
            batches = []
            current_batch = []
            current_len_sum = 0
            for item in items:
                if (current_len_sum + item[0]) >= self._batch_max_len or \
                    len(current_batch) >= self._max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_len_sum = 0
                current_batch.append(item)
                current_len_sum += item[0]
            if len(current_batch) > 0:
                batches.append(current_batch)
            return batches


    def collate_batch(self, items: List[List]) -> Dict[str, torch.Tensor]:
        # conver to items
        input_items = []
        ids_batch = []
        for input_line in items[0]:
            parts = input_line.strip().split('|')
            token_ids = [int(t_id) for t_id in parts[0].split()]
            ids_batch.append(token_ids)
            if len(parts) > 1:
                spans = parts[1].split(';')
                atom_spans = []
                for span in spans:
                    vals = span.split(',')
                    if len(vals) == 2:
                        atom_spans.append([int(vals[0]), int(vals[1])])
            if self._min_len < len(token_ids) < self._batch_max_len:
                input_items.append(InputItem(token_ids, atom_spans))
        lens = map(lambda a: len(a), ids_batch)
        input_max_len = max(1, max(lens))

        input_ids_batch = []
        mask_batch = []

        for input_ids in ids_batch:
            masked_input_ids = copy.deepcopy(input_ids)
            input_ids_batch.append(masked_input_ids + [self._tokenizer.pad_token_id] * (input_max_len - len(input_ids)))
            mask_batch.append([1] * len(input_ids) + [0] * (input_max_len - len(input_ids)))

        return {"input_ids": torch.tensor(input_ids_batch), "attention_mask": torch.tensor(mask_batch),
                "atom_spans": [item.atom_spans for item in input_items]}