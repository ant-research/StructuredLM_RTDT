from typing import List
import numpy as np
from data_structure.r2d2_tree import PyNode


def get_token_tree(root, tokens):
    if root.left is not None and root.right is not None:
        return '({} {})'.format(get_token_tree(root.left, tokens), get_token_tree(root.right, tokens))
    else:
        return tokens[root.pos]


def get_cache_size(merge_order: List[PyNode], seq_len:int, window_size:int):
    cache_size = 0
    merge_idx = 0
    for layer_idx in range(seq_len):
        if layer_idx <= window_size:
            cache_size += seq_len - layer_idx
        else:
            current_node = merge_order[merge_idx]
            merge_idx += 1
            start_idx = max(current_node.i - window_size, 0)
            end_idx = min(seq_len, current_node.j + window_size) - window_size
            new_cells = end_idx - start_idx - (current_node.j - current_node.i) + 1
            assert new_cells <= window_size + 1
            cache_size += new_cells
    return cache_size


def get_tree_from_merge_trajectory(merge_trajectory: np.array, seq_len, tokens=None, keep_merge_order=False):
    spans_for_splits = [[PyNode(None, None, i, i, -1), PyNode(None, None, i + 1, i + 1, -1)]
                         for i in range(seq_len - 1)]
    latest_span = spans_for_splits[0][0] if seq_len > 1 else None
    if keep_merge_order:
        merge_order = []
    for action_i in range(seq_len - 1):
        merge_pos = merge_trajectory[action_i]
        left, right = spans_for_splits[merge_pos]
        latest_span = PyNode(left, right, left.i, right.j, -1)
        if left.i - 1 >= 0:
            spans_for_splits[left.i - 1][1] = latest_span
        if right.j < len(spans_for_splits):
            spans_for_splits[right.j][0] = latest_span
        if keep_merge_order:
            merge_order.append(latest_span)
    root = latest_span
    results = [root]
    if tokens is not None:
        results.append(get_token_tree(root, tokens))
    if keep_merge_order:
        results.append(merge_order)
    if len(results) == 1:
        return results[0]
    else:
        return results


def get_tree_from_merge_trajectory_in_word(merge_trajectory: np.array, seq_len, atom_spans, indices_mapping, words):
    nodes = [PyNode(None, None, i, i, -1) for i in range(seq_len)]
    span_hit = 0
    for action_i in range(seq_len - 1):
        merge_pos = merge_trajectory[action_i]
        assert merge_pos < len(nodes) - 1
        left = nodes.pop(merge_pos)
        right = nodes.pop(merge_pos)
        if [left.i, right.j] in atom_spans:
            new_node = PyNode(None, None, left.i, right.j, -1)
            span_hit += 1
        else:
            new_node = PyNode(left, right, left.i, right.j, -1)
        nodes.insert(merge_pos, new_node)
        for idx in range(action_i, seq_len - 1):
            if merge_trajectory[idx] > merge_pos:
                merge_trajectory[idx] -= 1
    assert len(nodes) == 1
    assert span_hit == len(atom_spans)
    # adjust indices according to indices_mapping
    to_visit = [nodes[0]]
    while len(to_visit) > 0:
        current = to_visit.pop()
        current.i = indices_mapping[current.i]
        current.j = indices_mapping[current.j]
        if current.right:
            to_visit.append(current.right)
        if current.left:
            to_visit.append(current.left)

    return nodes[0], get_token_tree(nodes[0], words)


def get_nonbinary_spans_label(actions, SHIFT = 0, REDUCE = 1):
    spans = []
    stack = []
    pointer = 0
    binary_actions = []
    num_shift = 0
    num_reduce = 0
    for action in actions:
        # print(action, stack)
        if action == "SHIFT":
            stack.append((pointer, pointer))
            pointer += 1
            binary_actions.append(SHIFT)
            num_shift += 1
        elif action[:3] == 'NT(':
            label = "(" + action.split("(")[1][:-1]
            stack.append(label)
        elif action == "REDUCE":
            right = stack.pop()
            left = right
            n = 1
            while stack[-1][0] is not '(':
                left = stack.pop()
                n += 1
            span = (left[0], right[1], stack[-1][1:])
            if left[0] != right[1]:
                spans.append(span)
            stack.pop()
            stack.append(span)
            while n > 1:
                n -= 1
                binary_actions.append(REDUCE)        
                num_reduce += 1
        else:
            assert False  
    assert(len(stack) == 1)
    assert(num_shift == num_reduce + 1)
    return spans, binary_actions


def get_stats(span1, span2):
    tp = 0
    fp = 0
    fn = 0
    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    for span in span2:
        if span not in span1:
            fn += 1
    return tp, fp, fn

def get_nonbinary_spans(actions, SHIFT = 0, REDUCE = 1):
    spans = []
    stack = []
    pointer = 0
    binary_actions = []
    nonbinary_actions = []
    num_shift = 0
    num_reduce = 0
    for action in actions:
    # print(action, stack)
        if action == "SHIFT":
            nonbinary_actions.append(SHIFT)
            stack.append((pointer, pointer))
            pointer += 1
            binary_actions.append(SHIFT)
            num_shift += 1
        elif action[:3] == 'NT(':
            stack.append('(')            
        elif action == "REDUCE":
            nonbinary_actions.append(REDUCE)
            right = stack.pop()
            left = right
            n = 1
            while stack[-1] is not '(':
                left = stack.pop()
                n += 1
            span = (left[0], right[1])
            if left[0] != right[1]:
                spans.append(span)
            stack.pop()
            stack.append(span)
            while n > 1:
                n -= 1
                binary_actions.append(REDUCE)        
                num_reduce += 1
        else:
            assert False  
    assert(len(stack) == 1)
    assert(num_shift == num_reduce + 1)
    return spans, binary_actions, nonbinary_actions
