# coding=utf-8
# Copyright (c) 2021 Ant Group

from data_structure.syntax_tree import BinaryNode
from utils.misc import get_sentence_from_words, _align_spans
from eval.tree_wrapper import TreeDecoderWrapper
import codecs
from data_structure.const_tree import ConstTree, Lexicon


def convert_to_binary_node(root, pos=0):
    if isinstance(root, ConstTree):
        # assert len(root.children) == 1 or len(root.children) == 2
        if len(root.children) >= 2:
            left = root.children[0]
            if len(root.children) > 2:
                right = ConstTree('N-1', root.children[1:])
            else:
                right = root.children[1]
            left_node, l_len = convert_to_binary_node(left, pos)
            right_node, r_len = convert_to_binary_node(right, pos + l_len)
            return BinaryNode(left_node, right_node, None, None), l_len + r_len
        else:
            current = root.children[0]
            while isinstance(current, ConstTree) and len(current.children) == 1:
                current = current.children[0]
            if isinstance(current, Lexicon):
                return BinaryNode(None, None, pos, current.string), 1
            else:
                return convert_to_binary_node(current, pos)
    else:
        return BinaryNode(None, None, pos, root.string), 1

def to_latex_tree(root):
    if isinstance(root, ConstTree):
        if len(root.children) >= 2:
            child_repr = []
            for node in root.children:
                child_repr.append(to_latex_tree(node))
            expr = ''.join(child_repr)
            return f'[{expr}]'
        else:
            current = root.children[0]
            return to_latex_tree(current)
    else:
        return f'[{root.string}]'


class TreeFileWrapper(TreeDecoderWrapper):
    def __init__(self, tree_file):
        self._indexer = {}
        self._sentences = []
        self._entries = self._parse_tree_file(tree_file)

    def _parse_tree_file(self, tree_file):
        table = {}
        with codecs.open(tree_file, mode='r', encoding='utf-8') as f:
            for _line in f:
                s_idx = len(self._sentences)
                tree, lexicons = ConstTree.from_string(_line.strip())
                tokens = [_lexicon.string for _lexicon in lexicons]
                self._sentences.append(tokens)
                key = u' '.join(tokens)
                table[key.lower()] = tree
                for _lexicon in lexicons:
                    self._indexer.setdefault(_lexicon.string, set())
                    self._indexer[_lexicon.string].add(s_idx)
        return table

    def _index(self, tokens):
        s_counter = {}
        for t in tokens:
            if t in self._indexer:
                for s_id in self._indexer[t]:
                    s_counter.setdefault(s_id, 0)
                    s_counter[s_id] += 1
        max_count = -1
        max_s_id = -1
        for s_id, value in s_counter.items():
            if value >= max_count and len(tokens) == len(self._sentences[s_id]):
                max_count = value
                max_s_id = s_id
        assert max_s_id != -1
        key = u' '.join(self._sentences[max_s_id])
        return self._entries[key.lower()], list(range(len(tokens)))

    def __call__(self, tokens):
        key = u' '.join(tokens).lower()
        if key in self._entries:
            return self._entries[key], list(range(len(tokens)))
        else:
            return self._index(tokens)

    def print_binary_ptb(self, tokens) -> str:
        const_tree, _ = self(tokens)
        binary_node = convert_to_binary_node(const_tree)
        return binary_node.to_tree(ptb=True)

    def print_latex_tree(self, tokens) -> str:
        const_tree, _ = self(tokens)
        return to_latex_tree(const_tree)


class WordPieceTreeFileWrapper(TreeFileWrapper):
    def __init__(self, tree_file, tokenizer, sep_word=' '):
        super().__init__(tree_file)
        self._sep_word = sep_word
        self._tokenizer = tokenizer

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
        wp_tokens = self._tokenizer.convert_ids_to_tokens(outputs['input_ids'])
        key = u' '.join(wp_tokens)
        if key in self._entries:
            return self._entries[key], indices_mapping
        return None, None