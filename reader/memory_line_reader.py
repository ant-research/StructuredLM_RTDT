# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

from itertools import accumulate
import os
from torch.utils.data import Dataset
import torch
from typing import List, Optional, overload
from utils.misc import _align_spans, get_sentence_from_words
from abc import ABC, abstractmethod
from filelock import FileLock
import linecache
import logging
import pickle
import numpy as np

EMPTY_HISTORY = "[EMPTY]"
AGENT = "[AGENT]"
USER = "[USER]"
TOPIC = "[TOPIC]"


logger = logging.getLogger(__name__)


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
                max_len=999,
                max_line=-1,
                random=False,
                cache_dir: Optional[str] = None,
                descending=True,
                **kwargs) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._batch_max_len = batch_max_len
        self._max_batch_size = max_batch_size
        self._min_len = min_len
        self._max_len = max_len
        self._max_line = max_line
        self._random = random
        self._data_path = data_path_or_dir
        self._cache_dir = cache_dir
        self._shuffle_id = 0
        self._descending = descending
        self._lines = self._load_dataset(data_path_or_dir, **kwargs)
        self._samples_info = self._generate_samples_info(self._lines)
        self.shuffle(**kwargs)
    
    def _pre_shuffle(self, **kwargs):
        pass

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, idx):
        # return self._batches[idx]
        items = []
        for sample_id in self._batches[idx]:
            items.append(self._lines[sample_id])
        return items
    
    def _generate_samples_info(self, items):
        samples_info = []
        for item_id, input_item in enumerate(items):
            samples_info.append([item_id, len(input_item.ids)])
        return samples_info

    def _batchify(self, sample_infos):
        directory, filename = os.path.split(self._data_path)
        cache_dir = self._cache_dir
        if cache_dir is not None and not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_shuffle_{self._shuffle_id}_{filename}",
        )

        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):
            if os.path.exists(cached_features_file):
                with open(cached_features_file, 'rb') as handle:
                    # READ shuffled data
                    batches = pickle.load(handle)
                logger.info(
                    f'Loading shuffled samples from {cached_features_file}'
                )
            else:
                # shuffle
                logger.info("batchify")
                if not self._random:
                    len_dict = {}
                    np.random.shuffle(sample_infos)
                    for sample_id, sample_len in sample_infos:
                        arr = len_dict.setdefault(sample_len, [])
                        arr.append(sample_id)
                    len_keys = list(len_dict.keys())
                    len_keys.sort(key=lambda x: x, reverse=True)
                    rest_lines = len(sample_infos)
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
                                sample_id = len_dict[next_len].pop()
                                current_batch.append(sample_id)
                                rest_len -= next_len
                                rest_lines -= 1
                            else:
                                break
                        if len(current_batch) == 0:
                            # no sentence to add
                            break
                        batches.append(current_batch)
                else:
                    batches = []
                    current_batch = []
                    current_len_sum = 0
                    for sample_id, sample_len in sample_infos:
                        if (current_len_sum + sample_len) >= self._batch_max_len or \
                            len(current_batch) >= self._max_batch_size:
                            batches.append(current_batch)
                            current_batch = []
                            current_len_sum = 0
                        current_batch.append(sample_id)
                        current_len_sum += sample_len
                    if len(current_batch) > 0:
                        batches.append(current_batch)
                with open(cached_features_file, 'wb') as out_f:
                    pickle.dump(batches, out_f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file}, total len: {len(batches)}"
                )
        self._shuffle_id += 1
        
        if not self._descending:
            print('not descending')
            batches = [_ for _ in reversed(batches)]
        
        self._batches = batches
        

    def shuffle(self, **kwargs):
        self._pre_shuffle(**kwargs)
        self._batchify(self._samples_info)
        
    @abstractmethod
    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[InputItem]:
        pass


class BatchSelfRegressionLineDataset(BatchByLengthDataset):
    def __init__(self, path, tokenizer, batch_max_len, batch_size,
                 min_len=2, max_len=999, max_line=-1, random=False, **kwargs):
        '''
        params:
        random: True: for randomly batch sentences
                False: batch sentences in similar length
        '''
        super().__init__(path, tokenizer, batch_max_len, batch_size, 
                         min_len, max_len, max_line, random, **kwargs)

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
            if self._min_len < len(token_ids) < self._max_len:
                input_item_list.append(InputItem(np.array(token_ids), atom_spans))
            if len(input_item_list) > self._max_line > 0:
                break
        return input_item_list

class HugeBatchSelfRegressionLineDataset(BatchByLengthDataset):
    def __init__(self, path, tokenizer, batch_max_len, batch_size,
                 min_len=2, max_len=999, max_line=-1, random=False, **kwargs):
        '''
        params:
        random: True: for randomly batch sentences
                False: batch sentences in similar length
        '''
        super().__init__(path, tokenizer, batch_max_len, batch_size, 
                         min_len, max_len, max_line, random, **kwargs)


    def _load_dataset(self, data_path_prefix, **kwargs) -> List[InputItem]:
        index_path = f'{data_path_prefix}.idx'
        content_path = f'{data_path_prefix}.data'
        with open(index_path, mode='rb') as handle:
            lens = pickle.load(handle)
            
        ends = list(accumulate(lens))
        
        with open(content_path, 'rb') as handle:
            self._mmap_file = np.memmap(handle, dtype=np.int32, mode='r', order='C')
        return ends
        
    def _generate_samples_info(self, ends):
        samples_info = []
        prev_end = 0
        for item_id, end in enumerate(ends):
            if end == prev_end:
                # document split:
                continue
            if end - prev_end > self._min_len and end - prev_end < self._max_len:
                samples_info.append([item_id, end - prev_end])
            prev_end = end
        return samples_info
    
    def __getitem__(self, idx):
        items = []
        for sample_id in self._batches[idx]:
            st = self._lines[sample_id - 1] if sample_id > 0 else 0
            end = self._lines[sample_id]

            id_arr = self._mmap_file[st: end]
            items.append(InputItem(id_arr))
        return items
    
    
class SentenceIndex:
    def __init__(self, doc_id, line_id, st, end) -> None:
        self.doc_id = doc_id
        self.line_id = line_id
        self.st = st
        self.end = end
        self.next_index = None

class HugeBatchInsideOutsideNSPDataset(BatchByLengthDataset):
    def __init__(self, path, tokenizer, batch_max_len, batch_size,
                 min_len=2, max_len=999, max_line=-1, random=False, **kwargs):
        '''
        params:
        random: True: for randomly batch sentences
                False: batch sentences in similar length
        '''
        super().__init__(path, tokenizer, batch_max_len, batch_size, 
                         min_len, max_len, max_line, random, **kwargs)


    def _load_dataset(self, data_path_prefix, **kwargs) -> List[InputItem]:
        index_path = f'{data_path_prefix}.idx'
        content_path = f'{data_path_prefix}.data'
        with open(index_path, mode='rb') as handle:
            lens = pickle.load(handle)
            
        ends = list(accumulate(lens))
        
        with open(content_path, 'rb') as handle:
            self._mmap_file = np.memmap(handle, dtype=np.int32, mode='r', order='C')
        return ends
        
    def _generate_samples_info(self, ends):
        samples_info = []
        prev_end = 0
        doc_id = 0
        prev_index = None
        for line_id, end in enumerate(ends):
            if end == prev_end:
                doc_id += 1
                prev_index = None
            if end - prev_end > self._min_len and end - prev_end < self._max_len:
                current_index = SentenceIndex(doc_id, line_id, prev_end, end)
                if prev_index is not None:
                    prev_index.next_index = current_index
                prev_end = end
                prev_index = current_index
                samples_info.append(current_index)
            prev_end = end
        return samples_info
    
    def _batchify(self, sample_infos):
        directory, filename = os.path.split(self._data_path)
        cache_dir = self._cache_dir
        if cache_dir is not None and not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_shuffle_{self._shuffle_id}_{filename}",
        )

        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):
            if os.path.exists(cached_features_file):
                with open(cached_features_file, 'rb') as handle:
                    # READ shuffled data
                    batches = pickle.load(handle)
                logger.info(
                    f'Loading shuffled samples from {cached_features_file}'
                )
            else:
                # shuffle
                logger.info("batchify")
                
                paired_samples = []
                for sent_index in sample_infos:
                    if sent_index.next_index is not None and np.random.rand() < 0.5:
                        # next sentence
                        paired_samples.append([sent_index, sent_index.next_index, 1])
                    else:
                        # random next sentence
                        while True:
                            rand_idx = np.random.randint(0, len(sample_infos))
                            random_sample = sample_infos[rand_idx]
                            if random_sample.doc_id != sent_index.doc_id:
                                paired_samples.append([sent_index, random_sample, 0])
                                break
                np.random.shuffle(paired_samples)
                
                if not self._random:
                    len_dict = {}                    
                    for sent_a, sent_b, is_next_sent in paired_samples:
                        sample_len = sent_a.end - sent_a.st + sent_b.end - sent_b.st
                        arr = len_dict.setdefault(sample_len, [])
                        arr.append([sent_a, sent_b, is_next_sent])
                    len_keys = list(len_dict.keys())
                    len_keys.sort(key=lambda x: x, reverse=True)
                    rest_lines = len(paired_samples)
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
                                sample_pair = len_dict[next_len].pop()
                                current_batch.append(sample_pair)
                                rest_len -= next_len
                                rest_lines -= 1
                            else:
                                break
                        if len(current_batch) == 0:
                            # no sentence to add
                            break
                        batches.append(current_batch)
                else:
                    batches = []
                    current_batch = []
                    current_len_sum = 0
                    for sample_pair in paired_samples:
                        sample_len = sample_pair[0].end - sample_pair[0].st + \
                            sample_pair[1].end - sample_pair[1].st
                        if (current_len_sum + sample_len) >= self._batch_max_len or \
                            len(current_batch) >= self._max_batch_size:
                            batches.append(current_batch)
                            current_batch = []
                            current_len_sum = 0
                        current_batch.append(sample_pair)
                        current_len_sum += sample_len
                    if len(current_batch) > 0:
                        batches.append(current_batch)
                with open(cached_features_file, 'wb') as out_f:
                    pickle.dump(batches, out_f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file}, total len: {len(batches)}"
                )
        self._shuffle_id += 1
        
        if not self._descending:
            print('not descending')
            batches = [_ for _ in reversed(batches)]
        
        self._batches = batches
    
    def __getitem__(self, idx):
        items = []
        for sample_a, sample_b, is_next_sent in self._batches[idx]:
            sample_a_end = self._lines[sample_a.line_id]
            sample_a_st = self._lines[sample_a.line_id - 1] if sample_a.line_id > 0 else 0

            id_arr_a = self._mmap_file[sample_a_st: sample_a_end]
            
            sample_b_end = self._lines[sample_b.line_id]
            sample_b_st = self._lines[sample_b.line_id - 1] if sample_b.line_id > 0 else 0

            id_arr_b = self._mmap_file[sample_b_st: sample_b_end]
            items.append([InputItem(id_arr_a), InputItem(id_arr_b), is_next_sent])
        return items
    
class HugeBatchChunkedSentencesDataset(BatchByLengthDataset):
    def __init__(self, path, tokenizer, batch_max_len, batch_size,
                 min_len=2, max_len=999, max_line=-1, random=False, chunk_size=512,
                 **kwargs):
        '''
        params:
        random: True: for randomly batch sentences
                False: batch sentences in similar length
        '''
        self._chunk_size = chunk_size
        super().__init__(path, tokenizer, batch_max_len, batch_size, 
                         min_len, max_len, max_line, random, chunk_size=chunk_size, 
                         **kwargs)

    def _load_dataset(self, data_path_prefix, **kwargs) -> List[InputItem]:
        index_path = f'{data_path_prefix}.idx'
        content_path = f'{data_path_prefix}.data'
        with open(index_path, mode='rb') as handle:
            lens = pickle.load(handle)
            
        ends = list(accumulate(lens))
        
        with open(content_path, 'rb') as handle:
            self._mmap_file = np.memmap(handle, dtype=np.int32, mode='r', order='C')
        return ends
        
    def _generate_samples_info(self, ends):
        # chunk sentences
        samples_info = []
        prev_end = 0
        doc_id = 0
        current_chunk = []
        current_chunk_size = 0
        for line_id, end in enumerate(ends):
            if end == prev_end:
                doc_id += 1
                prev_index = None
            if end - prev_end > self._min_len and end - prev_end < self._max_len:
                if current_chunk_size + end - prev_end + 1 > self._chunk_size:
                    samples_info.append([current_chunk, current_chunk_size])
                    current_chunk = []
                    current_chunk_size = 0
                current_index = SentenceIndex(doc_id, line_id, prev_end, end)
                current_chunk.append(current_index)
                current_chunk_size += end - prev_end + 1 # considering sep id for Transformers
                prev_end = end
            prev_end = end
        if len(current_chunk) > 0:
            samples_info.append([current_chunk, current_chunk_size])
        return samples_info
    
    def _batchify(self, sample_infos):
        directory, filename = os.path.split(self._data_path)
        cache_dir = self._cache_dir
        if cache_dir is not None and not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_shuffle_{self._shuffle_id}_{filename}",
        )

        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):
            if os.path.exists(cached_features_file):
                with open(cached_features_file, 'rb') as handle:
                    # READ shuffled data
                    batches = pickle.load(handle)
                logger.info(
                    f'Loading shuffled samples from {cached_features_file}'
                )
            else:
                np.random.shuffle(sample_infos)
                batches = []
                current_chunk = []
                chunk_sum_len = 0
                for chunk, chunk_size in sample_infos:
                    if chunk_sum_len + chunk_size > self._batch_max_len:
                        batches.append(current_chunk)
                        current_chunk = []
                        chunk_sum_len = 0
                    else:
                        chunk_sum_len += chunk_size
                        current_chunk.append(chunk)
                        
                if len(current_chunk) > 0:
                    batches.append(current_chunk)        
                
                with open(cached_features_file, 'wb') as out_f:
                    pickle.dump(batches, out_f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file}, total len: {len(batches)}"
                )
                
                
        self._shuffle_id += 1
        
        self._batches = batches
    
    def __getitem__(self, idx):
        items = []
        for chunk in self._batches[idx]:
            chunks = []
            for sent_idx in chunk:
                sample_a_end = self._lines[sent_idx.line_id]
                sample_a_st = self._lines[sent_idx.line_id - 1] if sent_idx.line_id > 0 else 0

                id_arr = self._mmap_file[sample_a_st: sample_a_end]
                
                chunks.append([InputItem(id_arr), sent_idx.doc_id])
            items.append(chunks)
        return items