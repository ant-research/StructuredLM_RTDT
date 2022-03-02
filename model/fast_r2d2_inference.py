# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

from collections import deque
from typing import List, Tuple
import torch


def create_merge_order(indices, seq_len):
    spans_for_splits = [[(i, i), (i + 1, i + 1)] for i in range(seq_len - 1)]
    merge_order = deque()
    for action_i in range(seq_len - 1):
        merge_pos = indices[action_i]
        left, right = spans_for_splits[merge_pos]
        new_span = (left[0], right[1])
        if left[0] - 1 >= 0:
            spans_for_splits[left[0] - 1][1] = new_span
        if right[1] < len(spans_for_splits):
            spans_for_splits[right[1]][0] = new_span
        merge_order.append((left[0], right[1], merge_pos))
    return merge_order


def build_batch(s_indices, seq_lens):
    s_indices = s_indices.to('cpu').data.numpy()  # (batch_size, L)
    batch_size = s_indices.shape[0]
    merge_orders = []
    for batch_i in range(batch_size):
        merge_orders.append(create_merge_order(s_indices[batch_i], seq_lens[batch_i]))

    max_len = max(seq_lens)
    cell_ids = [[[-1] * max_len for _ in range(max_len)] for _ in range(batch_size)]
    cache_id_offset = 0
    root_ids = [0] * batch_size
    for batch_i in range(batch_size):
        root_ids[batch_i] = cache_id_offset
        for pos in range(seq_lens[batch_i]):
            cell_ids[batch_i][pos][pos] = cache_id_offset
            cache_id_offset += 1
    
    encoding_batchs = []
    cells_to_encode = sum(seq_lens - 1)
    while cells_to_encode > 0:
        current_batch = []
        update_ids = []
        for batch_i in range(batch_size):
            total_items = len(merge_orders[batch_i])
            while total_items > 0:
                total_items -= 1
                left_pos, right_pos, k = merge_orders[batch_i].popleft()
                if cell_ids[batch_i][left_pos][k] != -1 and \
                   cell_ids[batch_i][k + 1][right_pos] != -1:
                    current_batch.append((cell_ids[batch_i][left_pos][k], 
                                          cell_ids[batch_i][k + 1][right_pos]))
                    # cell_ids[batch_i][left_pos][right_pos] = cache_id_offset
                    update_ids.append((batch_i, left_pos, right_pos, cache_id_offset))
                    root_ids[batch_i] = cache_id_offset  # record the last cache_id
                    cache_id_offset += 1
                    cells_to_encode -= 1
                else:
                    merge_orders[batch_i].append((left_pos, right_pos, k))
                    # break
        for batch_i, left_pos, right_pos, cache_id in update_ids:
            cell_ids[batch_i][left_pos][right_pos] = cache_id
        assert len(current_batch) > 0
        encoding_batchs.append(current_batch)
    return encoding_batchs, root_ids


def force_encode(parser, r2d2, input_ids, attention_mask, atom_spans: List[List[Tuple[int]]]):
    # initialize tensor cache
    s_indices = parser(input_ids, attention_mask, atom_spans=atom_spans, add_noise=False)
    seq_lens = torch.sum(attention_mask, dim=1, dtype=torch.int)  # (batch_size, 1)
    seq_lens_np = seq_lens.to('cpu').data.numpy()
    e_ij_cache = torch.full([sum(seq_lens) * 2, r2d2.input_dim], 0.0, device=r2d2.device)
    _, embedding = r2d2.initialize_embeddings(input_ids, seq_lens_np)
    e_ij_cache[0:embedding.shape[0]] = embedding
    encoding_batchs, root_ids = build_batch(s_indices, seq_lens_np)
    # print(encoding_batchs)
    cache_id_offset = sum(seq_lens)
    for current_batch in encoding_batchs:        
        current_batch = torch.tensor(current_batch, device=r2d2.device)
        e_ikj = e_ij_cache[current_batch]
        e_ij, _ = r2d2.encode(e_ikj.unsqueeze(1))  # (?, 1, dim)
        e_ij = e_ij.squeeze(1)
        e_ij_cache[cache_id_offset: cache_id_offset + e_ij.shape[0]] = e_ij
        cache_id_offset += e_ij.shape[0]
    return e_ij_cache[root_ids]