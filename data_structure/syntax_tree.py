# coding=utf-8
# Copyright (c) 2021 Ant Group

from data_structure.basic_structure import AtomicSpans


class ConllNode:
    def __init__(self, node_id, parent_id, token, postag, rel):
        self.parent_id = parent_id
        self.token = token
        self.node_id = node_id
        self.postag = postag
        self.rel = rel


class ConllTree:
    def __init__(self, nodes, name):
        self.nodes = nodes
        self.name = name
        if self.nodes[0].token != '<root>':
            self.nodes.insert(0, ConllNode(0, 0, '<root>', None, None))

    @property
    def tokens(self):
        return list(map(lambda x: x.token, self.nodes[1:]))

    def __str__(self):
        expr = []
        for n in self.nodes[1:]:
            expr.append('{}\t{}\t{}\t{}\t{}'.format(n.node_id, n.token, n.postag, n.parent_id, n.rel))
        return '\n'.join(expr)


class BinaryNode:
    def __init__(self, left, right, pos, token, **kwargs):
        self.left = left
        self.right = right
        self.pos = pos
        self.token = token
        self.label = kwargs.get('label', None)
        self._start = None
        self._end = None
        self._depth = -1
        self._size = 0

    @property
    def size(self):
        if self._size == 0:
            if self.left is None and self.right is None:
                self._size = 1
            else:
                assert self.left is not None and self.right is not None
                self._size = self.left.size + self.right.size
        return self._size

    @property
    def depth(self):
        if self._depth == -1:
            if self.left is not None and self.right is not None:
                self._depth = max(self.left.depth, self.right.depth) + 1
            else:
                self._depth = 0
        return self._depth

    @property
    def span(self):
        if self._start is None and self._end is None:
            if self.left is not None and self.right is not None:
                left_st, left_ed = self.left.span
                right_st, right_ed = self.right.span
                self._start = left_st
                self._end = right_ed
            else:
                self._start = self.pos
                self._end = self.pos
        return self._start, self._end

    def convert(self, forced_spans: AtomicSpans, current_len=0):
        st, ed = self.span
        for _span, _token in zip(forced_spans.spans, forced_spans.tokens):
            if st == _span.start and ed == _span.end:
                return BinaryNode(None, None, current_len, _token)
        if self.left is None and self.right is None:
            return BinaryNode(None, None, current_len, self.token)
        new_left = self.left.convert(forced_spans, current_len)
        left_st, left_ed = new_left.span
        new_right = self.right.convert(forced_spans, left_ed + 1)
        return BinaryNode(new_left, new_right, None, None)

    def to_tree(self, ptb=False):
        if self.left is not None and self.right is not None:
            left_token_expr = self.left.to_tree(ptb)
            right_token_expr = self.right.to_tree(ptb)
            if not ptb:
                return f'({left_token_expr} {right_token_expr})'
            else:
                return f'(N-1 {left_token_expr}{right_token_expr})'
        else:
            if not ptb:
                return self.token
            else:
                return f'(T-1 {self.token})'

    def to_latex_tree(self):
        if self.left is not None and self.right is not None:
            left_token_expr = self.left.to_latex_tree()
            right_token_expr = self.right.to_latex_tree()
            return f'[{left_token_expr}{right_token_expr}]'
        else:
            expr = self.token.replace('#', '\\#').replace('.', '\\\\.').replace(',', '{,}')
            return f'[{expr}]'

    def to_latex_tree_qTree(self):
        if self.left is not None and self.right is not None:
            label = str(self.label) if self.label is not None else 'none'
            left_token_expr = self.left.to_latex_tree_qTree()
            right_token_expr = self.right.to_latex_tree_qTree()
            return f'[.{label} {left_token_expr} {right_token_expr} ]'
        else:
            label = str(self.label) if self.label is not None else 'none'
            # expr = self.token.replace('#', '\\#').replace('.', '\\\\.').replace(',', '{,}')
            return f'[.{label} {self.token} ]'

    def to_spans(self, offset=0):
        span_set = set()
        span_len = self.size
        if span_len <= 1:
            # span at least longer than 1
            return span_set
        span_set.add((offset, offset + span_len - 1))
        if self.left is not None and self.right is not None:
            span_set |= self.left.to_spans(offset)
            left_size = self.left.size
            right_offset = offset + left_size
            span_set |= self.right.to_spans(right_offset)
        return span_set


class BinaryTree:
    def __init__(self, root):
        self._root = root
