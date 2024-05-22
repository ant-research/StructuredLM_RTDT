from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random
import json
from utils.misc import _align_spans, get_sentence_from_words
from transformers import AutoTokenizer
from typing import Dict, List

# {"text": np.array, "sentence_splits": list, "summary": np.array(summary will always be treated as one sentence)}
class XSumDataset(data.Dataset):
    
    def __init__(self, data_dir, mode, tokenizer, document_threshold=900, threshold=1024, seperator=" ", **kwargs):
        data_name = data_dir + '/' + mode + '.json'
        self._tokenizer = tokenizer
        self.document_threshold = document_threshold 
        self.threshold = threshold
        self._lines = self._load_dataset(data_name, mode=mode, sep=seperator)
    
    def get_examples(self, datapath):
        with open(datapath, "r") as json_file:
            examples = json.load(json_file)
        return examples

    def _to_ids_and_atom_spans(self, text, seperator):
        if seperator is None:
            tokens = self._tokenizer.tokenize(text)
            ids = self._tokenizer.convert_tokens_to_ids(tokens)
            atom_spans = None
            indices_mapping = None
        else:
            sentence, spans = get_sentence_from_words(text.strip().split(seperator), seperator)
            outputs = self._tokenizer.encode_plus(sentence,
                                                  add_special_tokens=False,
                                                  return_offsets_mapping=True)
            new_spans = outputs['offset_mapping']
            word_starts, word_ends = _align_spans(spans, new_spans)
            atom_spans = []
            indices_mapping = []
            for st, ed in zip(word_starts, word_ends):
                if st != ed:
                    atom_spans.append((st, ed))
                indices_mapping.append([st, ed])
            ids = outputs['input_ids']
            atom_spans = atom_spans
        return ids, atom_spans, indices_mapping

    def _load_dataset(self, data_path_or_dir, **kwargs) -> List[Dict]:
        print(data_path_or_dir)
        self.mode = kwargs.pop('mode')

        seperator = None if 'sep' not in kwargs else kwargs['sep']
        self.input_examples = self.get_examples(data_path_or_dir)
        input_items = []
        for input_example in self.input_examples:
            text = []
            summarytext = []
            sentence_splits = [0]
            for document_sent in input_example["document"]:
                ids, atom_spans, indices_mapping = \
                    self._to_ids_and_atom_spans(document_sent, seperator)
                if sentence_splits[-1] + len(ids) <= self.document_threshold:
                    text = text + ids
                    sentence_splits.append(len(text))
                else:
                    sp = self.document_threshold - sentence_splits[-1]
                    text = text + ids[:sp]
                    sentence_splits.append(len(text))
                if sentence_splits[-1] >= self.document_threshold:
                    break
            text = text + [self._tokenizer.convert_tokens_to_ids("Summary"), self._tokenizer.convert_tokens_to_ids(":")]
            if self.mode == "train" or self.mode == "train_tiny":
                sentence_splits.append(len(text))
            for summary_sent in input_example["summary"]:
                ids, atom_spans, indices_mapping = \
                    self._to_ids_and_atom_spans(summary_sent, seperator)
                summarytext = summarytext + ids
                if self.mode == "train" or self.mode == "train_tiny":
                    if len(text) + len(ids) <= self.threshold:
                        text = text + ids
                    else:
                        sp = self.threshold - len(text)
                        text = text + ids[:sp]
                    if len(text) >= self.threshold:
                        break
                    # sentence_splits.append(sentence_splits[-1]+len(ids))
            text = np.array(text)
            summarytext = np.array(summarytext)
            current_item = {"text": text, "sentence_splits": sentence_splits[1:], "summary": summarytext}
            input_items.append(current_item)
        return input_items

    def __getitem__(self, idx):
        return self._lines[idx]

    def __len__(self):
        return len(self._lines)