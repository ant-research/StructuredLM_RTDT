import json
import torch
from torch.utils.data import Dataset
import random


class SpanDataset(Dataset):
    label_dict = dict()
    encoder = None
    def __init__(self, path, encoder_dict, train_frac=1.0, length_filter=None, **kwargs):
        super().__init__()
        encoder_key_list = list(encoder_dict.keys())
        self.encoder_type = encoder_key_list[0]
        with open(path, 'r') as f:
            raw_data = f.readlines()

        if train_frac < 1.0:
            red_num_lines = int(len(raw_data) * train_frac)
            raw_data = raw_data[:red_num_lines]

        # preprocess
        self.data = list()
        filter_by_length_cnt = 0
        filter_by_empty_label_cnt = 0
        sample_cnt = 0 #

        for data in raw_data:
            instance = json.loads(data)
            words = instance['text'].split()

            if length_filter is not None and length_filter > 0:
                if len(words) > length_filter:
                    filter_by_length_cnt += 1
                    continue
            subwords_dict = {}
            subword_to_word_idx_dict = {}
            for encoder_key in encoder_key_list:
                subwords, subword_to_word_idx = encoder_dict[encoder_key].tokenize(words, get_subword_indices=True)
                subwords_dict[encoder_key] = subwords
                subword_to_word_idx_dict[encoder_key] = subword_to_word_idx
            span_label_pair = {}
            for item in instance['targets']:
                spans = []
                for span_key in ('span1', 'span2'):
                    if span_key in item:
                        span = tuple(item[span_key])
                        spans.append(span)
                
                sample_cnt += 1
                spans = tuple(spans)
                # in case of multiple labels for one span in one sentence
                if spans not in span_label_pair:
                    span_label_pair[spans] = set()

                label = item['label']
                self.add_label(label)
                span_label_pair[spans].add(self.label_dict[label])
                
            
            # span_label_pair contains all the spans and labels need to be predicted in current sentence
            # form: {(span1, span2[option]): {label1, label2, ...}, ...}
            
            # Process
            def _process_span_idx(span_idx, encoder_key):
                w2w_idx = subword_to_word_idx_dict[encoder_key]
                span_idx = self.get_tokenized_span_indices(w2w_idx, span_idx)
                return span_idx
            # spans : {
            #            'span1': {'glove':[[st1, ed1], [st2, ed2], ...]
            #                      'bert':[[st1, ed1], [st2, ed2], ...]},
            #            'span2': {...}
            #            'label': [{labels for first span in this sentence}, {labels for second span}, ...]
            #                                                                                                  }
            spans = {'span1': {}, 'span2': {}, 'label': []}
            for span in span_label_pair:
                for encoder_key in encoder_key_list:
                    if encoder_key not in spans['span1']:
                        spans['span1'][encoder_key] = []
                    spans['span1'][encoder_key].append(_process_span_idx(span[0], encoder_key))
                    if len(span) > 1:
                        if encoder_key not in spans['span2']:
                            spans['span2'][encoder_key] = []
                        spans['span2'][encoder_key].append(_process_span_idx(span[1], encoder_key))
                spans['label'].append(span_label_pair[span])

            labels = [list(x) for x in spans['label']]
            if len(labels) != 0:
                for encoder_key in encoder_key_list:
                    subwords_dict[encoder_key] = torch.tensor(subwords_dict[encoder_key]).long()
                    subword_to_word_idx_dict[encoder_key] = torch.tensor(subword_to_word_idx_dict[encoder_key]).long()
                instance_dict = {
                    'subwords': subwords_dict,
                    'subword_to_word_idx': subword_to_word_idx_dict,
                    'spans1': spans['span1'],
                    'spans2': spans['span2'] if len(spans['span2']) > 0 else None,
                    'labels': labels,
                    'seq_len': len(words),
                    'atom_spans': spans['span1'][encoder_key_list[0]] + spans['span2'][encoder_key_list[0]] if len(spans['span2']) \
                        else spans['span1'][encoder_key_list[0]]
                }

                self.data.append(
                    instance_dict
                )
            else:
                filter_by_empty_label_cnt += 1

        self.data.sort(key=self.instance_length_getter)

        self.length_map = {}
        for idx, rec in enumerate(self.data):
            self.length_map.setdefault(self.instance_length_getter(rec), 0)
            self.length_map[self.instance_length_getter(rec)] += 1

        self.info = {
            'size': len(self),
            f'filter_by_length_{length_filter}': filter_by_length_cnt,
            'filter_by_empty_labels': filter_by_empty_label_cnt,
            'total samples': sample_cnt,
        }

    def __len__(self):
        return len(self.data)

    def instance_length_getter(self, rec):
        return len(rec['subwords'][self.encoder_type])

    def __getitem__(self, index):
        return self.data[index]
    
    def reorder(self):
        map = {}
        maxlen = -1
        for item in self.data:
            l = self.instance_length_getter(item)
            if l not in map:
                map[l] = []
            map[l].append(item)
            if l > maxlen:
                maxlen = l
        order = []
        for l in range(maxlen+1):
            if l not in map:
                continue
            order.append(l)
        random.shuffle(order)
        res = []
        for item in order:
            res.extend(map[item])
        self.data = res

    @staticmethod
    def get_tokenized_span_indices(subword_to_word_idx, orig_span_indices):
        orig_start_idx, orig_end_idx = orig_span_indices
        start_idx = subword_to_word_idx.index(orig_start_idx)
        # Search for the index of the last subword
        end_idx = len(subword_to_word_idx) - 1 - subword_to_word_idx[::-1].index(orig_end_idx - 1)
        return [start_idx, end_idx]

    @classmethod
    def add_label(cls, label):
        if label not in cls.label_dict:
            cls.label_dict[label] = len(cls.label_dict)
