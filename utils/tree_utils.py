from collections import deque
from functools import total_ordering
from typing import List
import numpy as np
from data_structure.r2d2_tree import PyNode
from utils.misc import padding
import torch


def get_token_tree(root, tokens):
    if root.left is not None and root.right is not None:
        return '({} {})'.format(get_token_tree(root.left, tokens), get_token_tree(root.right, tokens))
    else:
        return tokens[root.pos]


def get_cache_size(merge_order: List[PyNode], seq_len : int, window_size : int):
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
    if seq_len == 1:
        return PyNode(None, None, 0, 0, -1)
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
    if keep_merge_order:
        assert len(merge_order) == seq_len - 1
    root = latest_span
    results = [root]
    if tokens is not None:
        if root is not None:
            results.append(get_token_tree(root, tokens))
        else:
            results.append(' '.join(tokens))
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
    total_nodes = 0
    while len(to_visit) > 0:
        current = to_visit.pop()
        total_nodes += 1
        current.i = indices_mapping[current.i]
        current.j = indices_mapping[current.j]
        if current.right:
            to_visit.append(current.right)
        if current.left:
            to_visit.append(current.left)
            
    assert total_nodes == 2 * len(words) - 1
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

def _convert_to_tree(splits, start, end, cache_idx, leaves_ids):
    split_idx = splits[0]
    if split_idx == -1:
        assert end - start == 1
        leaves_ids.append(cache_idx)
        return PyNode(None, None, start, end - 1, cache_idx)
    else:
        # right_len = end - split_idx - 1
        left_len = split_idx - start + 1
        # preorder traversal
        left_root = _convert_to_tree(splits[1: 2 * left_len], start, split_idx + 1, 
                                     cache_idx + 1, leaves_ids)
        right_root = _convert_to_tree(splits[2 * left_len:], split_idx + 1, end, 
                                      cache_idx + 2 * left_len, leaves_ids)
        
        root = PyNode(left_root, right_root, start, end - 1, cache_idx)
        return root
    
def _build_composing_order(root, batch):
    if root.left is not None and root.right is not None:
        left_batch_id = _build_composing_order(root.left, batch)
        right_batch_id = _build_composing_order(root.right, batch)
        
        batch_id = max(left_batch_id, right_batch_id) + 1
        batch[batch_id].append([root.left.cache_id, root.right.cache_id, root.cache_id])
        return batch_id
    else:
        return -1

def build_merge_orders(seq_lens_np, rebuild_target_ids_np, group_ids, cache_size, pairwised=True):
    group_roots = []
    if pairwised:
        max_batch_size = max(seq_lens_np[0::2] + seq_lens_np[1::2]) + 1
    else:
        max_batch_size = max(seq_lens_np)
    composing_orders = [[] for _ in range(max_batch_size)]
    total_batch = -1
    leaves_ids = []
    trees = []
    for group_id, group in enumerate(group_ids):
        offset = 0
        start = 0
        
        roots = []
        for batch_id in group:
            ids_len = seq_lens_np[batch_id]
            splits = rebuild_target_ids_np[group_id][offset: offset + 2 * ids_len - 1]
            if pairwised:
                # add extra root node, all cache id + 1
                root = _convert_to_tree(splits, start, start + ids_len, offset + 1 + cache_size * group_id, leaves_ids)
            else:
                root = _convert_to_tree(splits, start, start + ids_len, offset + cache_size * group_id, leaves_ids)
            # build merge orders
            roots.append(root)
            trees.append(root)
            offset += 2 * ids_len - 1
            start += ids_len
        
        if pairwised:
            group_root = PyNode(roots[0], roots[1], 0, start - 1, cache_size * group_id)
        else:
            group_root = roots[0]
        total_batch = max(total_batch, _build_composing_order(group_root, composing_orders))
        
        group_roots.append(group_root)
    
    return composing_orders[:total_batch + 1], leaves_ids, trees

def build_trees(seq_lens_np, split_ids_np, cache_ids_np):
    batch_size = len(seq_lens_np)
    leaves_ids = []
    
    def _convert_to_cache_id_tree(splits, start, end, leaves_ids, cache_indices=None):
        split_idx = splits[0]
        if split_idx == -1:
            assert end - start == 1
            cache_id = -1 if cache_indices is None else cache_indices[0]
            return PyNode(None, None, start, end - 1, cache_id)
        else:
            # right_len = end - split_idx - 1
            left_len = split_idx - start + 1
            # preorder traversal
            if cache_indices is not None:
                left_root = _convert_to_cache_id_tree(splits[1: 2 * left_len], start, split_idx + 1, 
                                                      leaves_ids, cache_indices[1: 2 * left_len])
                right_root = _convert_to_cache_id_tree(splits[2 * left_len:], split_idx + 1, end, 
                                                      leaves_ids, cache_indices[2 * left_len:])
            else:
                left_root = _convert_to_cache_id_tree(splits[1: 2 * left_len], start, split_idx + 1, 
                                                      leaves_ids, None)
                right_root = _convert_to_cache_id_tree(splits[2 * left_len:], split_idx + 1, end, 
                                                      leaves_ids, None)
            
            cache_id = -1 if cache_indices is None else cache_indices[0]
            root = PyNode(left_root, right_root, start, end - 1, cache_id)
            return root
        
    def _on_batch_i(batch_id):
        ids_len = seq_lens_np[batch_id]
        splits = split_ids_np[batch_id][: 2 * ids_len - 1]

        if cache_ids_np is not None:
            return _convert_to_cache_id_tree(splits, 0, ids_len, leaves_ids, cache_ids_np[batch_id])
        else:
            return _convert_to_cache_id_tree(splits, 0, ids_len, leaves_ids, None)

    return list(map(lambda batch_i: _on_batch_i(batch_i), range(batch_size)))
        
def _cover_masked_spans(span, masked_spans):
    for masked_span in masked_spans:
        if span.i <= masked_span.i and span.j >= masked_span.j and span.seq_len > masked_span.seq_len:
            return True
    return False

def _hit_masked_spans(span, masked_spans):
    for masked_span in masked_spans:
        if span.i == masked_span.i and span.j == masked_span.j:
            return True
    return False

def _hit_masked_pos(span, masked_positions):
    for pos in masked_positions:
        if span.i <= pos and span.j >= pos:
            return True
    return False

def _masked_tree_to_sequence(root, masked_poses, cache_ids=[], tgt_ids=[], masks=[], pos_ranges=[]):
    to_visit = [root]
    while len(to_visit) > 0:
        current = to_visit.pop(-1)
        if not _hit_masked_pos(current, masked_poses):
            cache_ids.append(current.cache_id)
            masks.append(1)
            pos_ranges.append([current.i, current.j])
            tgt_ids.append(-1)
        if current.left is not None and current.right is not None:
            to_visit.append(current.right)
            to_visit.append(current.left)
    return cache_ids, tgt_ids, masks, pos_ranges


def _build_masked_tree(tree, input_ids_np, seq_len, mask_id, mask_roller):
    cache_ids = []
    tgt_ids = []
    masks = []
    pos_ranges = []
    masked_pos = list(filter(lambda pos: mask_roller(pos), range(seq_len)))

    for pos_i in masked_pos:
        masks.append(1)
        cache_ids.append(mask_id)
        pos_ranges.append([pos_i, pos_i])
        tgt_ids.append(input_ids_np[pos_i])
        
    return _masked_tree_to_sequence(tree, masked_pos, cache_ids, tgt_ids, masks, pos_ranges)
        

def build_mlm_inputs(splits, cache_ids, input_ids_np=None, seq_lens_np=None, 
                     mask_id=0, sep_id=1, pairwise=False, mask_roller=None):
    cache_ids_batch = []
    masks_batch = []
    pos_range_batch = []
    tgt_ids_batch = []
    trees = build_trees(seq_lens_np, splits, cache_ids)
    
    if not pairwise:
        for batch_i, tree in enumerate(trees):
            cache_ids, tgt_ids, masks, pos_ranges = \
                _build_masked_tree(tree, input_ids_np[batch_i], seq_lens_np[batch_i], mask_id, mask_roller)
            cache_ids_batch.append(cache_ids)
            masks_batch.append(masks)
            pos_range_batch.append(pos_ranges)
            tgt_ids_batch.append(tgt_ids)
    else:
        for batch_i, (tree1, tree2) in enumerate(zip(trees[::2], trees[1::2])):

            cache_ids1, tgt_ids1, masks1, pos_ranges1 = \
                _build_masked_tree(tree1, input_ids_np[batch_i * 2], seq_lens_np[batch_i * 2], mask_id, mask_roller)
            cache_ids2, tgt_ids2, masks2, pos_ranges2 = \
                _build_masked_tree(tree2, input_ids_np[batch_i * 2 + 1], seq_lens_np[batch_i * 2 + 1], mask_id, mask_roller)
            
            cache_ids = cache_ids1 + [sep_id] + cache_ids2
            tgt_ids = tgt_ids1 + [sep_id] + tgt_ids2
            masks = masks1 + [1] + masks2
            
            pos_offset = seq_lens_np[batch_i * 2]
            
            pos_ranges = pos_ranges1 + [[pos_offset, pos_offset]] + [[i + pos_offset + 1, j + pos_offset + 1] for i, j in pos_ranges2]
            
            cache_ids_batch.append(cache_ids)
            masks_batch.append(masks)
            pos_range_batch.append(pos_ranges)
            tgt_ids_batch.append(tgt_ids)
    
    padding(cache_ids_batch, 0)
    padding(masks_batch, 0)
    padding(pos_range_batch, [0, 0])
    padding(tgt_ids_batch, -1)

    return cache_ids_batch, masks_batch, pos_range_batch, tgt_ids_batch, trees

def pairwise_tgt_ids(cache_ids: torch.Tensor, tgt_ids: torch.Tensor):
    seq_lens = (cache_ids != 0).sum(dim=1)  # (N)
    pair_lens = seq_lens[0::2] + seq_lens[1::2]
    max_pair_len = pair_lens.max()
    batch_size = cache_ids.shape[0]
    pairwise_rebuild_tgt_ids = torch.full((batch_size // 2, max_pair_len), -1, dtype=torch.long)
    for pair_i in range(batch_size // 2):
        batch_id1 = pair_i * 2
        batch_id2 = pair_i * 2 + 1
        pairwise_rebuild_tgt_ids[pair_i, :seq_lens[batch_id1]] = tgt_ids[batch_id1, :seq_lens[batch_id1]]
        pairwise_rebuild_tgt_ids[pair_i, seq_lens[batch_id1]: seq_lens[batch_id1] + seq_lens[batch_id2]] = \
            tgt_ids[batch_id2, : seq_lens[batch_id2]]
    return pairwise_rebuild_tgt_ids

def rebuild_tgt_ids(split_ids, tgt_ids: torch.Tensor, pairwise):
    batch_size = split_ids.shape[0]
    seq_len = split_ids.shape[1]
    # if not pairwise:
    rebuild_tgt_ids = torch.full((batch_size, seq_len), -1, dtype=torch.long)
    
    token_indices = (split_ids.flatten() == -1).nonzero().squeeze(1)
    rebuild_tgt_ids = rebuild_tgt_ids.flatten()
    rebuild_tgt_ids.scatter_(0, token_indices, tgt_ids.flatten())
    rebuild_tgt_ids = rebuild_tgt_ids.view(batch_size, seq_len)
    
    if pairwise:
        seq_lens = (split_ids != -100).sum(dim=1)  # (batch_size)
        pair_lens = seq_lens[0::2] + seq_lens[1::2]
        max_pair_len = pair_lens.max()
        pairwise_rebuild_tgt_ids = torch.full((batch_size // 2, max_pair_len), -1, dtype=torch.long)
        
        for pair_i in range(batch_size // 2):
            batch_id1 = pair_i * 2
            batch_id2 = pair_i * 2 + 1
            pairwise_rebuild_tgt_ids[pair_i, :seq_lens[batch_id1]] = rebuild_tgt_ids[batch_id1, :seq_lens[batch_id1]]
            pairwise_rebuild_tgt_ids[pair_i, seq_lens[batch_id1]: seq_lens[batch_id1] + seq_lens[batch_id2]] = \
                rebuild_tgt_ids[batch_id2, : seq_lens[batch_id2]]
        rebuild_tgt_ids = pairwise_rebuild_tgt_ids

    return rebuild_tgt_ids

def flatten_trees(roots):
    flatten_nodes_batch = []
    for root in roots:
        to_visit = deque()
        to_visit.append(root)
        flatten_nodes = []
        while len(to_visit) > 0:
            current = to_visit.pop(-1)
            flatten_nodes.append(current)
            if current.left is not None and current.right is not None:
                to_visit.append(current.right)
                to_visit.append(current.left)
        flatten_nodes_batch.append(flatten_nodes)
    return flatten_nodes_batch

def find_span_in_tree(root, st, ed):
    """

    Using binary search to find a span represented by [st, ed] in tree `root`.

    """
    q = [root]
    while q:
        current = q.pop(0)
        if st == current.i and ed == current.j:
            # hit
            return current
        
        if st <= current.left.j and ed >= current.right.i: 
            # when tgt span is splitted, it doesn't exist in parse tree
            break
        elif ed <= current.left.j:
            # tgt span is inside left subtree
            q.append(current.left)
        else:
            # st >= current.right.i
            # tgt span is inside right subtree
            q.append(current.right)

    return None