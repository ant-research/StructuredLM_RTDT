from transformers import AutoTokenizer
from utils.misc import get_sentence_from_words, align_spans
from enum import Enum
import codecs
import torch
from eval.evaluate_lm import R2D2GenFastEvaluator

class TreeFormat(Enum):
    BRACKET=1,
    PTB=2,
    LATEX=3

def convert_to_bracket(root, org_tokens, atom_spans, indices_mapping):
    if [root.i, root.j] in atom_spans:
        assert indices_mapping[root.i] == indices_mapping[root.j]
        return f'{org_tokens[indices_mapping[root.i]]}'
    else:
        if root.left is not None and root.right is not None:
            left_str = convert_to_bracket(root.left, org_tokens, atom_spans, indices_mapping)
            right_str = convert_to_bracket(root.right, org_tokens, atom_spans, indices_mapping)
            return f'({left_str} {right_str})'
        else:
            return f'{org_tokens[indices_mapping[root.i]]}'

def convert_to_ptb(root, org_tokens, atom_spans, indices_mapping):
    if [root.i, root.j] in atom_spans:
        assert indices_mapping[root.i] == indices_mapping[root.j]
        return f'(T-1 {org_tokens[indices_mapping[root.i]]})'
    else:
        if root.left is not None and root.right is not None:
            left_str = convert_to_ptb(root.left, org_tokens, atom_spans, indices_mapping)
            right_str = convert_to_ptb(root.right, org_tokens, atom_spans, indices_mapping)
            return f'(N-1 {left_str}{right_str})'
        else:
            return f'(T-1 {org_tokens[indices_mapping[root.i]]})'

class GenerativePTBPrinter:
    def __init__(self, beam_searcher, vocab_dir, device, sep_word=' ', tree_format=TreeFormat.BRACKET):
        self._beam_searcher = beam_searcher
        self._tokenizer = AutoTokenizer.from_pretrained(vocab_dir)
        self._sep_word = sep_word
        self._device = device
        self._format = tree_format

    def convert_to_tree(self, tokens):
        sentence, spans = get_sentence_from_words(tokens, self._sep_word)
        outputs = self._tokenizer.encode_plus(sentence,
                                              add_special_tokens=False,
                                              return_offsets_mapping=True)
        input_ids = outputs['input_ids']
        # org_tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
        offset_mapping = outputs['offset_mapping']
        word_starts, word_ends = align_spans(spans, offset_mapping)
        atom_spans = [] # minimal span should be a whole word
        indices_mapping = [0] * len(outputs['input_ids'])
        for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
            if ed > st:
                atom_spans.append([st, ed])
            for idx in range(st, ed + 1):
                indices_mapping[idx] = pos

        input_ids = torch.tensor(input_ids, device=self._device)
        _, states = self._beam_searcher.beam_search(input_ids.unsqueeze(0), atom_spans=[atom_spans])  # for old one
        root = states[0][0].stack_top
        if self._format == TreeFormat.BRACKET:
            return convert_to_bracket(root, tokens, atom_spans, indices_mapping)
        elif self._format == TreeFormat.PTB:
            result =  convert_to_ptb(root, tokens, atom_spans, indices_mapping)
            if result.startswith('(T-1'):
                result = f'(NT-1 {result})'
            return result
        elif self._format == TreeFormat.LATEX:
            pass
        else:
            raise Exception(f'Unsupported format {self._format}')


class NewGenerativePTBPrinter:
    def __init__(self, beam_searcher, vocab_dir, device, sep_word=' ', tree_format=TreeFormat.BRACKET):
        self._beam_searcher = beam_searcher
        self._tokenizer = AutoTokenizer.from_pretrained(vocab_dir)
        self._sep_word = sep_word
        self._device = device
        self._format = tree_format

    def convert_to_tree(self, tokens):
        sentence, spans = get_sentence_from_words(tokens, self._sep_word)
        outputs = self._tokenizer.encode_plus(sentence,
                                              add_special_tokens=False,
                                              return_offsets_mapping=True)
        input_ids = outputs['input_ids']
        # org_tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
        offset_mapping = outputs['offset_mapping']
        word_starts, word_ends = align_spans(spans, offset_mapping)
        atom_spans = [] # minimal span should be a whole word
        indices_mapping = [0] * len(outputs['input_ids'])
        for pos, (st, ed) in enumerate(zip(word_starts, word_ends)):
            if ed > st:
                atom_spans.append([st, ed])
            for idx in range(st, ed + 1):
                indices_mapping[idx] = pos

        input_ids = torch.tensor(input_ids, device=self._device)
        states = self._beam_searcher.beam_search(target_ids=input_ids.unsqueeze(0), 
                                                 target_masks=torch.ones_like(input_ids.unsqueeze(0)), 
                                                 atom_spans=[atom_spans])  # for old one
        root = states[0][0].stack_top
        if self._format == TreeFormat.BRACKET:
            return convert_to_bracket(root, tokens, atom_spans, indices_mapping)
        elif self._format == TreeFormat.PTB:
            result =  convert_to_ptb(root, tokens, atom_spans, indices_mapping)
            if result.startswith('(T-1'):
                result = f'(NT-1 {result})'
            return result
        elif self._format == TreeFormat.LATEX:
            pass
        else:
            raise Exception(f'Unsupported format {self._format}')
        