# coding=utf-8
# Copyright (c) 2021 Ant Group

import torch

from utils.math_util import max_neg_value


def _subtree(node, tokens):
    if len(node.candidates) > 0:
        top_pair = node.candidates[0]
        left_span = _subtree(top_pair.left, tokens)
        right_span = _subtree(top_pair.right, tokens)
        return f'({left_span} {right_span})'
    else:
        return tokens[node.pos]


class DotDict(dict):

    def copy(self):
        return DotDict(**self)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, key, value):
        self[key] = value


class Span(object):
    """
    Define a span by the given start and end, inclusive. Indices are 0-based.
    """

    def __init__(self, start, end):
        self._start = start
        self._end = end

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, val):
        self._end = val

    def contains(self, st, ed):
        return self._start <= st <= ed <= self._end

    def is_covered(self, st, ed):
        return st <= self._start <= self._end <= ed

    def intersects(self, st, ed):
        return ed >= self._start and st <= self._end


class AtomicSpans(object):
    def __init__(self, atomic_spans, tokens=None):
        self._spans = []
        self.tokens = tokens
        for st, ed in atomic_spans:
            self._spans.append(Span(st, ed))

    @property
    def spans(self):
        return self._spans

    def is_valid_span(self, height, pos):
        return self.conflict_span(height, pos) is None

    def conflict_span(self, height, pos):
        """
        Given the root position of a span, return the first span that are conflict with the given span.
        Two span are conflict iff there are overlap between them and no one fully contains the other.
        :param height: the height of the given span.
        :param pos: the pos of the given span.
        :return: The first span conflict with the given one.
        """
        start_idx = pos
        end_idx = height + pos
        assert end_idx >= start_idx
        for span in self._spans:
            if not span.contains(start_idx, end_idx) and not span.is_covered(start_idx, end_idx) \
                    and span.intersects(start_idx, end_idx):
                return span
        return None


class ChartNode:
    def __init__(self, height, pos, vec=None, extra_attributes=None):
        self.height = height
        self.pos = pos
        self.candidates = []
        self.e_ij = vec
        self.pruned = False
        self.is_terminal = False
        self.log_p_ij = None
        self._log_p_sum = None

        if extra_attributes is not None:
            for attr_name in extra_attributes:
                self.__setattr__(attr_name, None)

    @property
    def log_p_sum(self):
        """
        log_p_sum: Summation of log_p_ij of all nodes in the tree recovered by self.
        """
        return self._log_p_sum if not self.is_terminal else self._log_p_sum.detach()

    @property
    def is_leaf(self):
        return len(self.candidates) == 0 or self.is_terminal

    @property
    def children(self):
        if len(self.candidates) > 0:
            return self.candidates[0].left, self.candidates[0].right
        else:
            return None, None

    def get_token_tree(self, tokens):
        return _subtree(self, tokens)


class ChartTable:
    def __init__(self, nodes, create_node_func=None):
        self._node_table = [[None] * (len(nodes) - h) for h in range(len(nodes))]
        for i, node in enumerate(nodes):
            self._node_table[0][i] = node
        self._calculate_table = [nodes]
        self._merge_logs = []

        if create_node_func is None:
            self.create_node = lambda height, pos, vec: ChartNode(height, pos, vec)
        else:
            self.create_node = create_node_func

    def table(self, i, j):
        return self._calculate_table[i][j]

    @property
    def table_size(self):
        return [len(nodes) for nodes in self._calculate_table]

    @property
    def node_size(self):
        return [len(nodes) for nodes in self._node_table]

    @property
    def root(self):
        return self._node_table[-1][0]

    @property
    def is_finished(self):
        # assert len(self._calculate_table[-1]) > 0
        return len(self._calculate_table[-1]) <= 1

    @property
    def seq_len(self):
        return len(self._node_table[0])

    def node(self, i, j):
        return self._node_table[i][j]

    def gather_node(self, pos, bos_node, eos_node):
        """
        :param pos: the index of cell in the first row
        :param bos_node:
        :param eos_node:
        :return:
        """
        left_node, right_node = self._gather(pos)
        if left_node is None:
            assert pos == 0
            left_node = bos_node
        if right_node is None:
            assert pos == len(self._node_table[0]) - 1
            right_node = eos_node
        return left_node, right_node

    def gather_tensor(self, pos, left_default, right_default):
        left_node, right_node = self._gather(pos)
        if left_node is None:
            assert pos == 0
            left_tensor = left_default
        else:
            left_tensor = left_node.e_ij
        if right_node is None:
            assert pos == len(self._node_table[0]) - 1
            right_tensor = right_default
        else:
            right_tensor = right_node.e_ij
        return left_tensor, right_tensor

    def _gather(self, pos):
        left_node = None
        right_node = None
        for height in range(len(self._node_table)):
            if pos - 1 - height >= 0 and self._node_table[height][pos - 1 - height] is not None:
                left_node = self._node_table[height][pos - 1 - height]
            if pos + 1 < len(self._node_table[height]) and self._node_table[height][pos + 1] is not None:
                right_node = self._node_table[height][pos + 1]
        return left_node, right_node

    def cross(self, left, right):
        if left == right:
            return left
        left_st = left.pos
        right_end = right.pos + right.height
        # assert left_end < right_st
        if self._node_table[right_end - left_st][left_st] is None:
            self._node_table[right_end - left_st][left_st] = self.create_node(right_end - left_st, left_st, None)
        return self._node_table[right_end - left_st][left_st]

    def merge(self, i):
        j = i + 1
        assert j < len(self._calculate_table[0])
        merge_node = self.cross(self._calculate_table[0][i], self._calculate_table[0][j])
        # remove left
        for height in range(len(self._calculate_table)):
            if len(self._calculate_table[height]) > j:
                node = self._calculate_table[height].pop(j)
                node.pruned = True
            else:
                assert self._calculate_table[height][-1] is None
                self._calculate_table[height].pop(-1)
            if i >= height:
                self._calculate_table[height][i - height].pruned = True
                for h_i in range(height, len(self._calculate_table)):
                    if h_i + 1 == len(self._calculate_table) and i - height < len(self._calculate_table[h_i]):
                        self._calculate_table[h_i][i - height] = None
                    elif i - height < len(self._calculate_table[h_i]) \
                            and i - height < len(self._calculate_table[h_i + 1]):
                        self._calculate_table[h_i][i - height] = self._calculate_table[h_i + 1][i - height]
                        self._calculate_table[h_i + 1][i - height] = None

        for height in range(len(self._calculate_table) - 1):
            assert len(self._calculate_table[height]) - len(self._calculate_table[height + 1]) == 1
        highest = len(self._calculate_table) - 1
        for pos_i in range(len(self._calculate_table[-1])):
            if self._calculate_table[highest][pos_i] is None:
                self._calculate_table[highest][pos_i] = \
                    self.cross(self._calculate_table[0][pos_i], self._calculate_table[0][highest + pos_i])
        merge_node.is_terminal = True
        self._merge_logs.append(merge_node)

    def expand(self):
        prev_node = None
        next_layer = []
        for node in self._calculate_table[-1]:
            if prev_node is not None:
                next_layer.append(self.cross(prev_node, node))
            prev_node = node
        self._calculate_table.append(next_layer)

    def _is_valid_node(self, node, atomic_spans: AtomicSpans):
        if atomic_spans is not None and not atomic_spans.is_valid_span(node.height, node.pos):
            span = atomic_spans.conflict_span(node.height, node.pos)
            assert span is not None
            if self._node_table[span.end - span.start][span.start] is None:
                return False
        return True

    def prepare_best_merge(self, atomic_spans: AtomicSpans = None):
        """
        Please refere to find the best merge point
        :param atomic_spans: Spans not splittable. Only used in evaluating word-level tasks
               Please refer to the Pruning algorithm in the paper.
        :return:
        """
        bigram_nodes = []
        input_len = len(self._node_table[0])
        visited = [False] * (input_len ** 2)
        for node in self._calculate_table[-1]:
            node_stack = [node]
            while len(node_stack) > 0:
                top = node_stack.pop(0)
                if not self._is_valid_node(top, atomic_spans):
                    continue
                if visited[top.height * input_len + top.pos]:
                    continue
                visited[top.height * input_len + top.pos] = True
                if top.is_leaf:
                    continue
                node_stack_len = len(node_stack)
                for pair in top.candidates:
                    if not pair.left.pruned and not pair.right.pruned \
                            and self._is_valid_node(pair.left, atomic_spans) \
                            and self._is_valid_node(pair.right, atomic_spans):
                        left_node = pair.left
                        right_node = pair.right
                        break
                assert left_node is not None
                assert right_node is not None
                if not left_node.is_leaf:
                    node_stack.append(left_node)
                if not right_node.is_leaf:
                    node_stack.append(right_node)
                if len(node_stack) == node_stack_len:
                    bigram_nodes.append(top)

        assert len(bigram_nodes) > 0
        merge_pos_arr = []
        log_p_stack = []
        log_p_ij_stack = []
        template = bigram_nodes[0].log_p_ij
        empty_padding = torch.zeros_like(template).fill_(max_neg_value(template.dtype))
        for bigram_node in bigram_nodes:
            pos = self._calculate_table[1].index(bigram_node)
            merge_pos_arr.append(pos)
            if pos > 0:
                log_p_stack.append(self._calculate_table[1][pos - 1].log_p_ij)
            else:
                log_p_stack.append(empty_padding)
            if pos < len(self._calculate_table[1]) - 1:
                log_p_stack.append(self._calculate_table[1][pos + 1].log_p_ij)
            else:
                log_p_stack.append(empty_padding)
            log_p_ij_stack.append(bigram_node.log_p_ij)
        assert len(log_p_stack) % 2 == 0
        return merge_pos_arr, log_p_stack, log_p_ij_stack

    def get_token_tree(self, tokens):
        assert len(self._node_table[-1]) == 1
        assert self._node_table[-1][0] is not None
        return self._node_table[-1][0].get_token_tree(tokens)

    def get_merge_log(self, tokens):
        logs = []
        for merge_node in self._merge_logs:
            logs.append(_subtree(merge_node, tokens))
        return logs


def find_best_merge_batch(chart_tables, current_step, atomic_spans, window_size, device):
    """
    Return the best merge points if a chart table is not finished.
    """
    lr_log_p_stack = []
    log_p_ij_stack = []
    merge_pos_arr = []
    n_table = 0
    indices_mapping = [0] * len(chart_tables)
    best_merge_points = [-1] * len(chart_tables)
    for table_i, table in enumerate(chart_tables):
        if table.is_finished:
            continue
        if current_step >= window_size:
            merge_pos, lr_log_p, log_p_ij = table.prepare_best_merge(atomic_spans[table_i])
            merge_pos_arr.append(merge_pos)
            lr_log_p_stack.append(lr_log_p)
            log_p_ij_stack.append(log_p_ij)
            indices_mapping[n_table] = table_i
            n_table += 1

    if n_table > 0:
        lr_log_p_max_len = max(map(lambda x: len(x), lr_log_p_stack))
        log_p_ij_max_len = max(map(lambda x: len(x), log_p_ij_stack))
        assert log_p_ij_max_len == lr_log_p_max_len / 2
        lr_log_p_list = []
        log_p_ij_list = []
        template = lr_log_p_stack[0][0]
        pad_zero = torch.zeros(1, device=device)
        pad_maxneg = torch.zeros(1, device=device).fill_(max_neg_value(template.dtype))
        for lr_log_p in lr_log_p_stack:
            lr_log_p_list.extend(lr_log_p)
            lr_log_p_list.extend([pad_zero] * (lr_log_p_max_len - len(lr_log_p)))
        for log_p_ij in log_p_ij_stack:
            log_p_ij_list.extend(log_p_ij)
            log_p_ij_list.extend([pad_maxneg] * (log_p_ij_max_len - len(log_p_ij)))

        # (batch, max_seq_len, 2)
        lr_log_p = torch.stack(lr_log_p_list).view(len(lr_log_p_stack), log_p_ij_max_len, 2)
        lr_log_np = torch.log((1 - torch.exp(lr_log_p)).clamp(min=1e-9)).sum(dim=2)
        log_p_ij = torch.stack(log_p_ij_list).view(len(log_p_ij_stack), log_p_ij_max_len)
        best_merge_indices = (log_p_ij + lr_log_np).argmax(dim=1)
        for batch_i, i in enumerate(best_merge_indices):
            best_merge_points[indices_mapping[batch_i]] = merge_pos_arr[batch_i][i]
    return best_merge_points
