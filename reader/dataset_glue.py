from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random
from utils.misc import _align_spans, get_sentence_from_words
from transformers import AutoTokenizer
from transformers.data.processors import glue_processors, glue_output_modes
from typing import Dict, List


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

class GlueDataset(data.Dataset):
    
    def __init__(self, task_name, data_dir, mode, tokenizer, seperator=" ", **kwargs):
        data_name = data_dir + '@@' + mode
        self._tokenizer = tokenizer
        self._lines = self._load_dataset(data_name, task_name=task_name, mode=mode, sep=seperator)

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
        if '@@' in data_path_or_dir:
            data_path_or_dir = data_path_or_dir.split('@@')[0]
            print(data_path_or_dir)
        task_name = kwargs.pop('task_name')
        self.mode = kwargs.pop('mode')

        seperator = None if 'sep' not in kwargs else kwargs['sep']
        glue_processor = glue_processors[task_name]()
        if self.mode == "train":
            self.input_examples = glue_processor.get_train_examples(data_path_or_dir)
        elif self.mode == "dev":
            self.input_examples = glue_processor.get_dev_examples(data_path_or_dir)
        self.labels = glue_processor.get_labels()
        self.output_mode = glue_output_modes[task_name]
        input_items = []
        self.model_type = "single"
        if self.input_examples[0].text_b is not None:
            self.model_type = "pair"
        else:
            self.model_type = "single"
        for input_example in self.input_examples:
            if task_name == "cola":
                input_example.text_a = input_example.text_a.replace('(','').replace(')','')
            ids_a, atom_spans_a, indices_mapping = \
                self._to_ids_and_atom_spans(input_example.text_a, seperator)

            if self.output_mode == "classification":
                label_idx = self.labels.index(input_example.label)
            elif self.output_mode == "regression":
                raise Exception("Regression not supported")
            else:
                raise Exception("Illegal output mode")
            if input_example.text_b is not None:
                ids_b, atom_spans_b, indices_mapping = self._to_ids_and_atom_spans(input_example.text_b, seperator)
                text = ids_a +[self._tokenizer.convert_tokens_to_ids("<|SEP|>")] + ids_b + [self._tokenizer.convert_tokens_to_ids("<|CLS|>")]
                text = np.array(text)
                current_item = {"text":text, "sentence_splits":[len(ids_a), len(ids_a)+1, len(ids_a)+1+len(ids_b)], "label":label_idx}
            else:
                text = ids_a + [self._tokenizer.convert_tokens_to_ids("<|CLS|>")]
                text = np.array(text)
                current_item = {"text":text, "sentence_splits":[len(ids_a)], "label":label_idx}
            if self.mode == "train":
                if task_name == "qqp":
                    if len(current_item["text"]) >= 100:
                        continue
                elif task_name == "mnli" or task_name == "mnli-mm" or task_name == "qnli":
                    if len(current_item["text"]) >= 150:
                        continue
                else:
                    pass
            input_items.append(current_item)
        return input_items

    def __getitem__(self, idx):
        return self._lines[idx]
    
    def get_output_mode(self):
        return self.output_mode

    def __len__(self):
        return len(self._lines)