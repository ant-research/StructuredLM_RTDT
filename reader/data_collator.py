from typing import List, Dict
import torch
import numpy as np
from multiprocessing.pool import ThreadPool
from collections import OrderedDict
from utils.vocab_builder import load_span_tokenizer
from utils.r2d2_span_tokenizer import SpanTokenizingSession
from utils.misc import align_spans
import cppbackend
import codecs


def fill_subarray(arr, st, ed, val):
    arr[st:ed, st:ed].fill(val)

def sent_collator(input_list):
    max_len = max(map(lambda x: len(x), input_list))
    padded_ids_list = []
    masks = []
    for input_ids in input_list:
        padded_len = max_len - len(input_ids)
        padded_ids = np.append(input_ids, np.array([0] * padded_len))
        padded_ids_list.append(padded_ids)
        masks.append([1] * len(input_ids) + [0] * padded_len)
        # mask = np.zeros((max_len, max_len))
        # # mask[:len(input_ids), :len(input_ids)].fill(1)
        # masks.append(mask)
    return {"input_ids": torch.tensor(padded_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long)}

class DefaultCollator:
    def __init__(self, enable_group=True, external_vocab_path=None):
        # enable_group is deprecated

        if external_vocab_path is not None:
            self.span_tokenizer = load_span_tokenizer(external_vocab_path)
        else:
            self.span_tokenizer = None

    def generative_r2d2_collate_fn(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        ids_list = []
        group_ids = []
        max_sent_len = 0
        chunk_ids_list = []
        # chunk_masks = []
        segment_ids_list = []
        span_indices = []
        max_input_len = max(map(lambda x: len(x['text']), input_list))
        

        for sent_id, item in enumerate(input_list):
            chunk_ids_list.append(item['text'])
            chunk_size = len(item['text'])
            # chunk_mask = np.zeros( (max_input_len, max_input_len) )
            segment_ids = np.zeros(max_input_len)
            segment_ids_list.append(segment_ids)
            # chunk_masks.append(chunk_mask)
            splits = item['sentence_splits']
            splits.append(chunk_size)

            prev_idx = 0
            # cppbackend.create_mask(chunk_mask, np.array(splits))
            for segment_id, split_idx in enumerate(splits):
                if split_idx > prev_idx:
                    ids_segment = item['text'][prev_idx: split_idx]
                    if self.span_tokenizer is not None:
                        results = self.span_tokenizer.tokenize(ids_segment)
                        span_idx = np.zeros((len(results),))
                        if len(results) > 0:
                            assert len(results) % 3 == 0
                            for group_id in range(len(results) // 3):
                                idx, span_len, span_id = results[group_id * 3: group_id * 3 + 3]
                                span_idx[group_id * 3] = idx - span_len + 1
                                span_idx[group_id * 3 + 1] = idx
                                span_idx[group_id * 3 + 2] = span_id
                                
                        span_indices.append(span_idx)
                    # ids_lens = np.floor(np.log10(ids_segment)) + 1
                    # splits = np.cumsum(np.array(list(ids_lens)) + 1) - 1
                    # target = ','.join([f'{id}' for id in tgt_ids])
                    ids_list.append(ids_segment)
                    # chunk_mask[prev_idx: split_idx, prev_idx: split_idx].fill(1)
                    # print(segment_id)
                    segment_ids[prev_idx: split_idx].fill(segment_id + 1)
                    group_ids.append(sent_id)
                    max_sent_len = max(max_sent_len, split_idx - prev_idx)
                prev_idx = split_idx

        # print(chunk_mask)
        # segment_ids = torch.tensor(segment_ids)
        # print(segment_ids)
        
        # padding
        masks = []
        for sent_i in range(len(ids_list)):
            pad_len = max_sent_len - len(ids_list[sent_i])
            masks.append(np.array([1] * len(ids_list[sent_i]) + [0] * pad_len, dtype=np.int32))
            # print(ids_list[sent_i])
            ids_list[sent_i] = np.append(np.array(ids_list[sent_i], dtype=np.int32), np.array([0] * pad_len))

        for chunk_id, chunk_ids in enumerate(chunk_ids_list):
            pad_len = max_input_len - len(chunk_ids)
            chunk_ids_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))

        return {"chunk_input_ids": torch.tensor(np.array(chunk_ids_list, dtype=np.int32), dtype=torch.long),
                "chunk_masks": torch.tensor(np.array(segment_ids_list, dtype=np.int32), dtype=torch.long),
                "input_ids": torch.tensor(np.array(ids_list, dtype=np.int32), dtype=torch.long), 
                "masks": torch.tensor(np.array(masks, dtype=np.int32), dtype=torch.long), 
                "group_ids": np.array(group_ids),
                "span_ids": span_indices}

    def generative_r2d2_collate_fn_ext(self, input_list) -> Dict[str, torch.Tensor]:
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        ids_list = []
        group_ids = []
        max_sent_len = 0
        chunk_ids_list = []
        # chunk_masks = []
        span_indices = []
        max_input_len = max(map(lambda x: len(x['text']), input_list))
        segment_ids_list = []
        external_dict = OrderedDict()
        external_vocab_idx = 1  # start from 1, 0 is reserved for empty span ids
        tokenizer_session = SpanTokenizingSession(self.span_tokenizer)
        for sent_id, item in enumerate(input_list):
            chunk_ids_list.append(item['text'])
            chunk_size = len(item['text'])
            # chunk_mask = np.zeros( (max_input_len, max_input_len) )
            # chunk_masks.append(chunk_mask)
            segment_ids = np.zeros(max_input_len)
            segment_ids_list.append(segment_ids)
            splits = item['sentence_splits']
            splits.append(chunk_size)

            prev_idx = 0
            # cppbackend.create_mask(chunk_mask, np.array(splits))
            for segment_id, split_idx in enumerate(splits):
                if split_idx > prev_idx:
                    ids_segment = item['text'][prev_idx: split_idx]
                    if self.span_tokenizer is not None:
                        span_idx = tokenizer_session.tokenize(ids_segment)
                        span_indices.append(span_idx)

                    ids_list.append(ids_segment)
                    # chunk_mask[prev_idx: split_idx, prev_idx: split_idx].fill(1)
                    segment_ids[prev_idx: split_idx].fill(segment_id + 1)
                    group_ids.append(sent_id)
                    max_sent_len = max(max_sent_len, split_idx - prev_idx)
                prev_idx = split_idx
        
        # print(chunk_mask)
        # segment_ids = torch.tensor(segment_ids)
        # print(segment_ids)
        # padding
        masks = []
        for sent_i in range(len(ids_list)):
            pad_len = max_sent_len - len(ids_list[sent_i])
            masks.append(np.array([1] * len(ids_list[sent_i]) + [0] * pad_len, dtype=np.int32))
            # print(ids_list[sent_i])
            ids_list[sent_i] = np.append(np.array(ids_list[sent_i], dtype=np.int32), np.array([0] * pad_len))
        
        for chunk_id, chunk_ids in enumerate(chunk_ids_list):
            pad_len = max_input_len - len(chunk_ids)
            chunk_ids_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
        
        if self.span_tokenizer is not None:
            external_ids = tokenizer_session.span_indices
        else:
            external_ids = None
                # "chunk_input_ids": torch.tensor(np.array()),
                # "chunk_masks": torch.tensor(),
        return {"chunk_input_ids": torch.tensor(np.array(chunk_ids_list, dtype=np.int32), dtype=torch.long),
                "chunk_masks": torch.tensor(np.array(segment_ids_list, dtype=np.int32), dtype=torch.long),
                "input_ids": torch.tensor(np.array(ids_list, dtype=np.int32), dtype=torch.long), 
                "masks": torch.tensor(np.array(masks, dtype=np.int32), dtype=torch.long), 
                "group_ids": np.array(group_ids),
                "span_ids": span_indices,
                "external_vocab_ids": external_ids}


class GlueCollator(DefaultCollator):
    def __init__(self, clstgt_ids, enable_group=True, external_vocab_path=None, padding=-1):
        self._clstgt_ids = clstgt_ids
        self._padding = padding
        super().__init__(enable_group=enable_group, external_vocab_path=external_vocab_path)

    def generative_r2d2_glue_collate_fn(self, input_list) -> Dict[str, torch.Tensor]:
        if self._padding != -1:
            for item in input_list:
                appen = np.array([0]*(self._padding - len(item['text'])))
                item['text'] = np.concatenate((item['text'], appen))
        origin_dict = self.generative_r2d2_collate_fn_ext(input_list)
        eos_labels = []
        for item in input_list:
            eos_labels.append(self._clstgt_ids[item["label"]])
        origin_dict["eos_labels"] = np.array(eos_labels)
        return origin_dict


class XSumCollator(DefaultCollator):
    def __init__(self, enable_group=True, external_vocab_path=None, padding=-1):
        self._padding = padding
        super().__init__(enable_group=enable_group, external_vocab_path=external_vocab_path)
    
    def generative_r2d2_xsum_collate_fn(self, input_list) -> Dict[str, torch.Tensor]:
        if self._padding != -1:
            for item in input_list:
                appen = np.array([0]*(self._padding - len(item['text'])))
                item['text'] = np.concatenate((item['text'], appen))
        origin_dict = self.generative_r2d2_collate_fn_ext(input_list)
        
        # new added
        chunk_summarys_list = []
        max_summary_len = max(map(lambda x: len(x['summary']), input_list))
        for sent_id, item in enumerate(input_list):
            chunk_summarys_list.append(item['summary'])
        for chunk_id, chunk_ids in enumerate(chunk_summarys_list):
            pad_len = max_summary_len - len(chunk_ids)
            chunk_summarys_list[chunk_id] = np.append(chunk_ids, np.array([-100] * pad_len))
        origin_dict["summarys"] = torch.tensor(np.array(chunk_summarys_list, dtype=np.int32), dtype=torch.long)
        
        return origin_dict

class TextCollator(DefaultCollator):
    def __init__(self, tokenizer, splitter, external_vocab_path=None):
        self.tokenizer = tokenizer
        self.splitter = splitter
        super().__init__(external_vocab_path=external_vocab_path)

    def collate_fn(self, input_list) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        atom_spans_batch = []
        for sentence in input_list:
            tokens, split_word = self.splitter(sentence)
            offset = 0
            spans = []
            for word in tokens:
                length = len(word)
                spans.append((offset, offset + length))
                offset += length + len(split_word)
            outputs = self.tokenizer.encode_plus(sentence,
                                                 add_special_tokens=False,
                                                 return_offsets_mapping=True)
            input_ids = outputs['input_ids']
            offset_mapping = outputs['offset_mapping']
            word_starts, word_ends = align_spans(spans, offset_mapping)
            atom_spans = [] # minimal span should be a whole word
            for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
                if ed > st:
                    atom_spans.append([st, ed])
            input_ids_list.append({'text':input_ids, 'sentence_splits': []})
            atom_spans_batch.append(atom_spans)

        out_dict = self.generative_r2d2_collate_fn_ext(input_ids_list)
        out_dict['atom_spans'] = atom_spans_batch
        return out_dict
