# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

from collections import deque
from functools import reduce
from turtle import left
from typing import List, Tuple
from data_structure.r2d2_tree import PyNode
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy
from math import sqrt


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
    merge_orders = []  # i, j, k
    for batch_i in range(batch_size):
        merge_orders.append(create_merge_order(s_indices[batch_i], seq_lens[batch_i]))

    max_len = max(seq_lens)
    cell_ids = [[[-1] * max_len for _ in range(max_len)] for _ in range(batch_size)]
    cell_nodes = [[[None] * max_len for _ in range(max_len)] for _ in range(batch_size)]
    root_nodes = [None] * batch_size
    cache_id_offset = 0
    root_ids = [0] * batch_size
    for batch_i in range(batch_size):
        root_ids[batch_i] = cache_id_offset
        for pos in range(seq_lens[batch_i]):
            cell_ids[batch_i][pos][pos] = cache_id_offset
            cell_nodes[batch_i][pos][pos] = PyNode(None, None, pos, pos, cache_id_offset)
            cache_id_offset += 1
        root_nodes[batch_i] = cell_nodes[batch_i][0][0]
    
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
                    assert cell_nodes[batch_i][left_pos][k] is not None
                    assert cell_nodes[batch_i][k + 1][right_pos] is not None
                    cell_nodes[batch_i][left_pos][right_pos] = PyNode(cell_nodes[batch_i][left_pos][k], 
                                                                      cell_nodes[batch_i][k + 1][right_pos],
                                                                      left_pos, right_pos, cache_id_offset)
                    root_nodes[batch_i] = cell_nodes[batch_i][left_pos][right_pos]
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
    for n in root_nodes:
        assert n.i == 0
    return encoding_batchs, root_ids, root_nodes


def build_batch_given_trees(trees):
    batch_size = len(trees)
    merge_orders_batch = []  # i, j, k
    seq_lens = []
    for root in trees:
        merge_orders = deque()
        node_stack = deque([root])
        seq_len = root.j - root.i + 1
        seq_lens.append(seq_len)
        visited = [[False for _ in range(seq_len)] for _ in range(seq_len)]
        while len(node_stack) > 0:
            current = node_stack.pop()
            left, right = current.left, current.right
            visited[current.i][current.j] = True
            if left is not None and right is not None:
                if visited[left.i][left.j] and visited[right.i][right.j]:
                    merge_orders.append([current.i, current.j, left.j])
                else:
                    node_stack.append(current)
                    node_stack.append(right)
                    node_stack.append(left)
        merge_orders_batch.append(merge_orders)

    max_len = max(seq_lens)
    cell_ids = [[[-1] * max_len for _ in range(max_len)] for _ in range(batch_size)]
    cell_nodes = [[[None] * max_len for _ in range(max_len)] for _ in range(batch_size)]
    root_nodes = [None] * batch_size
    cache_id_offset = 0
    root_ids = [0] * batch_size
    for batch_i in range(batch_size):
        root_ids[batch_i] = cache_id_offset
        for pos in range(seq_lens[batch_i]):
            cell_ids[batch_i][pos][pos] = cache_id_offset
            cell_nodes[batch_i][pos][pos] = PyNode(None, None, pos, pos, cache_id_offset)
            cache_id_offset += 1
        root_nodes[batch_i] = cell_nodes[batch_i][0][0]
    
    encoding_batchs = []
    cells_to_encode = sum([l - 1 for l in seq_lens])
    while cells_to_encode > 0:
        current_batch = []
        update_ids = []
        for batch_i in range(batch_size):
            total_items = len(merge_orders_batch[batch_i])
            while total_items > 0:
                total_items -= 1
                left_pos, right_pos, k = merge_orders_batch[batch_i].popleft()
                if cell_ids[batch_i][left_pos][k] != -1 and \
                   cell_ids[batch_i][k + 1][right_pos] != -1:
                    current_batch.append((cell_ids[batch_i][left_pos][k], 
                                          cell_ids[batch_i][k + 1][right_pos]))
                    assert cell_nodes[batch_i][left_pos][k] is not None
                    assert cell_nodes[batch_i][k + 1][right_pos] is not None
                    cell_nodes[batch_i][left_pos][right_pos] = PyNode(cell_nodes[batch_i][left_pos][k], 
                                                                      cell_nodes[batch_i][k + 1][right_pos],
                                                                      left_pos, right_pos, cache_id_offset)
                    root_nodes[batch_i] = cell_nodes[batch_i][left_pos][right_pos]
                    # cell_ids[batch_i][left_pos][right_pos] = cache_id_offset
                    update_ids.append((batch_i, left_pos, right_pos, cache_id_offset))
                    root_ids[batch_i] = cache_id_offset  # record the last cache_id
                    cache_id_offset += 1
                    cells_to_encode -= 1
                else:
                    merge_orders_batch[batch_i].append((left_pos, right_pos, k))
                    # break
        for batch_i, left_pos, right_pos, cache_id in update_ids:
            cell_ids[batch_i][left_pos][right_pos] = cache_id
        assert len(current_batch) > 0
        encoding_batchs.append(current_batch)
    for n in root_nodes:
        assert n.i == 0
    return encoding_batchs, root_ids, root_nodes


def force_encode(parser, r2d2, input_ids, attention_mask, atom_spans: List[List[Tuple[int]]]):
    # initialize tensor cache
    with torch.no_grad():
        s_indices = parser(input_ids, attention_mask, atom_spans=atom_spans, add_noise=False)
    return force_encode_by_indices(s_indices, r2d2, input_ids, attention_mask)


def force_encode_by_indices(s_indices, r2d2, input_ids, attention_mask):
    # initialize tensor cache
    seq_lens = torch.sum(attention_mask, dim=1, dtype=torch.int)  # (batch_size, 1)
    seq_lens_np = seq_lens.to('cpu').data.numpy()
    e_ij_cache = torch.full([sum(seq_lens) * 2, r2d2.input_dim], 0.0, device=r2d2.device)
    _, embedding = r2d2.initialize_embeddings(input_ids, seq_lens_np)
    e_ij_cache[0:embedding.shape[0]] = embedding
    encoding_batchs, root_ids, root_nodes = build_batch(s_indices, seq_lens_np)
    cache_id_offset = sum(seq_lens)
    for current_batch in encoding_batchs:        
        current_batch = torch.tensor(current_batch, device=r2d2.device)
        e_ikj = e_ij_cache[current_batch]
        e_ij, _ = r2d2.encode(e_ikj.unsqueeze(1), force_encoding=True)  # (?, 1, dim)
        e_ij = e_ij.squeeze(1)
        e_ij_cache[cache_id_offset: cache_id_offset + e_ij.shape[0]] = e_ij
        cache_id_offset += e_ij.shape[0]
    return e_ij_cache[root_ids], e_ij_cache, root_nodes, s_indices


def force_encode_given_trees(trees, r2d2, input_ids, attention_mask):
    # initialize tensor cache
    seq_lens = torch.sum(attention_mask, dim=1, dtype=torch.int)  # (batch_size, 1)
    seq_lens_np = seq_lens.to('cpu').data.numpy()
    e_ij_cache = torch.full([sum(seq_lens) * 2, r2d2.input_dim], 0.0, device=r2d2.device)
    _, embedding = r2d2.initialize_embeddings(input_ids, seq_lens_np)
    e_ij_cache[0:embedding.shape[0]] = embedding
    encoding_batchs, root_ids, root_nodes = build_batch_given_trees(trees) # build_batch_given_trees(trees, seq_lens_np)
    cache_id_offset = sum(seq_lens)
    for current_batch in encoding_batchs:
        current_batch = torch.tensor(current_batch, device=r2d2.device)
        e_ikj = e_ij_cache[current_batch]
        e_ij, _ = r2d2.encode(e_ikj.unsqueeze(1), force_encoding=True)  # (?, 1, dim)
        e_ij = e_ij.squeeze(1)
        e_ij_cache[cache_id_offset: cache_id_offset + e_ij.shape[0]] = e_ij
        cache_id_offset += e_ij.shape[0]
    return e_ij_cache[root_ids], e_ij_cache, root_nodes


def force_decode(decoder, e_ij_cache, root_nodes, root_role_embedding=None, role_embedding=None):
    # return tensor pool, and (batch, i, j)->pool idx mapping
    
    for idx, node in enumerate(root_nodes):
        node.decode_cache_id = idx
    current_parent_indices = [node.cache_id for node in root_nodes]
    parent_nodes = [_ for _ in root_nodes]
    N = len(root_nodes)
    root_embedding = e_ij_cache[current_parent_indices]  # (N, dim)
    if root_role_embedding is not None:
        # decode root repr
        tensor_to_cat = root_role_embedding.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1)
        input = torch.cat([tensor_to_cat, root_embedding.unsqueeze(1)], dim= 1)
        decode_output = decoder(input)
        root_embedding = decode_output[:, 1, :]
    dim = root_embedding.shape[-1]
    total_len = reduce(lambda a, x: a + x, [n.j + 1 for n in root_nodes])
    device = next(decoder.parameters()).device
    tensor_pool = torch.full([2 * total_len, dim], 0.0, dtype=torch.float, device=device)
    tensor_pool[0: root_embedding.shape[0]] = root_embedding
    pool_id_offset = root_embedding.shape[0]
    while True:
        children_indices = []
        parent_indices = []
        for p_node in parent_nodes:
            if not p_node.is_leaf:
                children_indices.append(p_node.left.cache_id)
                children_indices.append(p_node.right.cache_id)
                assert p_node.decode_cache_id >= 0
                parent_indices.append(p_node.decode_cache_id)
        if len(parent_indices) == 0:
            break
        children_embedding = e_ij_cache[children_indices]  # (2 * N, dim)
        children_embedding = children_embedding.view(-1, 2, dim)
        parent_embedding = tensor_pool[parent_indices]
        decoder_input = torch.cat([parent_embedding.unsqueeze(1), children_embedding], dim=1)
        if role_embedding is not None:
            decoder_input = decoder_input + role_embedding
        decode_output = decoder(decoder_input)  # (N, 3, dim)
        decode_output = decode_output.index_select(dim=1, index=torch.tensor([1, 2], device=device)).view(-1, dim)
        tensor_pool[pool_id_offset: pool_id_offset + decode_output.shape[0]] = decode_output
        children_nodes = []
        cnt = 0
        for _, p_node in enumerate(parent_nodes):
            if not p_node.is_leaf:
                p_node.left.decode_cache_id = pool_id_offset + 2 * cnt
                p_node.right.decode_cache_id = pool_id_offset + 2 * cnt + 1
                children_nodes.append(p_node.left)
                children_nodes.append(p_node.right)
                cnt += 1
        pool_id_offset += decode_output.shape[0]
        parent_nodes = children_nodes
    return tensor_pool, root_nodes


def _convert_cache_to_batch(encoding_cache, root_nodes, device):
    encode_cache_ids_batch = []
    nodes_flatten_batch = []
    for root in root_nodes:
        # expand root to cache id list
        node_queue = deque()
        node_queue.append(root)
        encode_cache_ids = []
        nodes_flatten = []
        while len(node_queue) > 0:
            parent = node_queue.popleft()
            nodes_flatten.append(parent)
            encode_cache_ids.append(parent.cache_id)
            if not parent.is_leaf:
                node_queue.append(parent.left)
                node_queue.append(parent.right)
        encode_cache_ids_batch.append(encode_cache_ids)
        nodes_flatten_batch.append(nodes_flatten)
    # padding and generate mask
    max_ids_len = max(map(lambda x: len(x), encode_cache_ids_batch))
    masks = []
    for encode_cache_ids in encode_cache_ids_batch:
        masks.append([1] * len(encode_cache_ids) + [0] * (max_ids_len - len(encode_cache_ids)))
        encode_cache_ids.extend((max_ids_len - len(encode_cache_ids)) * [0])
    masks = torch.tensor(masks, device=device)  # (N, max_ids_len)
    dim = encoding_cache.shape[-1]
    encode_cache_ids_batch = torch.tensor(encode_cache_ids_batch, device=device)
    encoding_batch = encoding_cache[encode_cache_ids_batch.flatten()]
    encoding_batch = encoding_batch.view(len(root_nodes), max_ids_len, dim)  # (N, max_ids_len, dim)
    return encoding_batch, masks, nodes_flatten_batch

def _convert_cache_to_tensor(encoding_cache, decoding_cache, root_nodes, device):
    decode_cache_ids_batch = []
    encode_cache_ids_batch = []
    nodes_flatten_batch = []
    for root in root_nodes:
        # expand root to cache id list
        node_queue = deque()
        node_queue.append(root)
        decode_cache_ids = []
        encode_cache_ids = []
        nodes_flatten = []
        while len(node_queue) > 0:
            parent = node_queue.popleft()
            nodes_flatten.append(parent)
            assert parent.decode_cache_id >= 0
            decode_cache_ids.append(parent.decode_cache_id)
            encode_cache_ids.append(parent.cache_id)
            if not parent.is_leaf:
                node_queue.append(parent.left)
                node_queue.append(parent.right)
        decode_cache_ids_batch.append(decode_cache_ids)
        encode_cache_ids_batch.append(encode_cache_ids)
        nodes_flatten_batch.append(nodes_flatten)
    # padding and generate mask
    max_ids_len = max(map(lambda x: len(x), decode_cache_ids_batch))
    masks = []
    for decode_cache_ids, encode_cache_ids in zip(decode_cache_ids_batch, encode_cache_ids_batch):
        masks.append([1] * len(decode_cache_ids) + [0] * (max_ids_len - len(decode_cache_ids)))
        decode_cache_ids.extend((max_ids_len - len(decode_cache_ids)) * [0])
        encode_cache_ids.extend((max_ids_len - len(encode_cache_ids)) * [0])
    masks = torch.tensor(masks, device=device)  # (N, max_ids_len)
    dim = decoding_cache.shape[-1]
    decode_cache_ids_batch = torch.tensor(decode_cache_ids_batch, device=device)
    encode_cache_ids_batch = torch.tensor(encode_cache_ids_batch, device=device)
    decoding_batch = decoding_cache[decode_cache_ids_batch.flatten()]  # (N, max_ids_len, dim)
    decoding_batch = decoding_batch.view(len(root_nodes), max_ids_len, dim)
    encoding_batch = encoding_cache[encode_cache_ids_batch.flatten()]
    encoding_batch = encoding_batch.view(len(root_nodes), max_ids_len, dim)  # (N, max_ids_len, dim)
    return encoding_batch, decoding_batch, masks, nodes_flatten_batch


def multi_instance_learning(encoding_cache, decoding_cache, root_nodes, attn_linear, device):
    '''
    encoding_cache: (total_nodes_num, dim), contains representation of all nodes by bottom-up encoding
    decoding_cache: (total_nodes_num, dim), contains representation of all nodes by top-down decoding
    root_nodes: List of roots of trees in the current batch.
    attn_linear: Linear(dim, 1)
    '''
    attn_sqrt_dim = sqrt(decoding_cache.shape[-1])
    encoding_batch, decoding_batch, masks, nodes_flatten_batch = \
        _convert_cache_to_tensor(encoding_cache, decoding_cache, root_nodes, device)
    attn_scores = attn_linear(decoding_batch).squeeze(-1) / attn_sqrt_dim  # (N, max_ids_len)
    attn_scores.masked_fill_(masks == 0, -np.inf)
    attn_weights = F.softmax(attn_scores, dim=-1)  # (N, max_ids_len)
    weighted_logits = attn_weights.unsqueeze(1) @ encoding_batch  # (N, 1, dim)
    weighted_logits = weighted_logits.squeeze(1)
    return weighted_logits, attn_weights, nodes_flatten_batch

def multi_instance_multi_label(encoding_cache, decoding_cache, root_nodes, attn_fn, device):
    '''
    encoding_cache: (total_nodes_num, dim), contains representation of all nodes by bottom-up encoding
    decoding_cache: (total_nodes_num, dim), contains representation of all nodes by top-down decoding
    root_nodes: List of roots of trees in the current batch.
    attn_linear: Linear(dim, 1)
    '''
    attn_sqrt_dim = sqrt(decoding_cache.shape[-1])
    encoding_batch, decoding_batch, _, _ = \
        _convert_cache_to_tensor(encoding_cache, decoding_cache, root_nodes, device)
    return attn_fn(encoding_batch, decoding_batch)

def cross_sentence_attention(encoding_cache, decoding_cache, root_nodes, cross_function, attention_function, device):
    '''
    encoding_cache: (total_nodes_num, dim), contains representation of all nodes by bottom-up encoding
    decoding_cache: (total_nodes_num, dim), contains representation of all nodes by top-down decoding
    root_nodes: List of roots of trees in the current batch.
    attn_linear: Linear(dim, 1)
    '''
    
    encoding_batch, decoding_batch, masks, nodes_flatten_batch = \
        _convert_cache_to_tensor(encoding_cache, decoding_cache, root_nodes, device)
    
    batch_size = encoding_batch.shape[0] // 2
    max_ids_len = encoding_batch.shape[-2]
    dim_size = encoding_batch.shape[-1]
    
    encoding_batch = encoding_batch.reshape(batch_size, 2, max_ids_len, dim_size) # (N, 2, max_ids_len, dim)
    decoding_batch = decoding_batch.reshape(batch_size, 2, max_ids_len, dim_size) # (N, 2, max_ids_len, dim)
    masks = masks.view(masks.shape[0] // 2, 2, max_ids_len) # (N, 2, max_ids_len)

    encoding_batch_A, encoding_batch_B = torch.split(encoding_batch, 1, dim=1) # (N, 1, max_ids_len, dim)
    decoding_batch_A, decoding_batch_B = torch.split(decoding_batch, 1, dim=1) # (N, 1, max_ids_len, dim)

    sim_mat = cross_function(encoding_batch_A, encoding_batch_B, batch_size, max_ids_len)  # (N, max_len_A, max_len_B, dim)
    weight_mat = attention_function(decoding_batch_A, decoding_batch_B, batch_size, max_ids_len)  # (N, max_len_A, max_len_B)
    result = torch.einsum('nid,ni->nd', sim_mat, weight_mat)
    return result, None, nodes_flatten_batch


def multi_head_attention(encoding_cache, decoding_cache, root_nodes, task_embedding, attn, device):
    '''
    encoding_cache: (total_nodes_num, dim), contains representation of all nodes by bottom-up encoding
    decoding_cache: (total_nodes_num, dim), contains representation of all nodes by top-down decoding
    root_nodes: List of roots of trees in the current batch.
    attn_linear: Linear(dim, 1)
    '''
    encoding_batch, decoding_batch, padding_masks, nodes_flatten_batch = \
        _convert_cache_to_tensor(encoding_cache, decoding_cache, root_nodes, device)
    N = len(root_nodes)
    task_embedding = task_embedding.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1)
    attn_outputs, attn_weights = attn(query=task_embedding, key=decoding_batch, value=encoding_batch,
                                      key_padding_mask=(padding_masks != 1))
    return attn_outputs.squeeze(1), attn_weights, nodes_flatten_batch


def multi_head_self_attention(encoding_cache, 
                              decoding_cache,
                              root_nodes, 
                              multihead_attention: nn.MultiheadAttention, 
                              device):
    encoding_batch, decoding_batch, padding_masks, nodes_flatten_batch = \
        _convert_cache_to_tensor(encoding_cache, decoding_cache, root_nodes, device)
    # encoding_batch: (N, max_nodes_len, dim)
    # generate attention mask
    attn_masks = []
    max_nodes_len = encoding_batch.shape[1]
    max_ids_len = (max_nodes_len + 1) // 2
    attn_masks = np.zeros([len(root_nodes), max_ids_len, max_nodes_len])
    query_indices = np.zeros([len(root_nodes), max_ids_len])
    for batch_i, root in enumerate(root_nodes):
        node_queue = deque()
        node_queue.append([root, []])
        current_idx = 0  # order in preorder traversaling
        attn_masks[batch_i, :root.j - root.i + 1, 2 * (root.j - root.i) + 1:] = 1
        while len(node_queue) > 0:
            parent, ancestor_indices = node_queue.popleft()
            assert parent.decode_cache_id >= 0
            mask_indices = ancestor_indices + [current_idx]
            if not parent.is_leaf:
                node_queue.append([parent.left, mask_indices])
                node_queue.append([parent.right, mask_indices])
            else:
                # generate mask
                for masked_idx in mask_indices:
                    assert parent.i == parent.j
                    attn_masks[batch_i, parent.i, masked_idx] = 1
                    query_indices[batch_i, parent.i] = current_idx
            current_idx += 1
    attn_masks = torch.tensor(attn_masks, device=device, dtype=torch.bool)
    S, L = attn_masks.shape[1:]
    attn_masks = attn_masks.unsqueeze(1).repeat(1, multihead_attention.num_heads, 1, 1)
    attn_masks = attn_masks.view(-1, S, L)
    query_indices = torch.tensor(query_indices, device=device, dtype=torch.long)
    dim = decoding_batch.shape[-1]
    query = decoding_batch.gather(dim=1, index=query_indices.unsqueeze(2).repeat(1, 1, dim))
    padding_masks = padding_masks != 1
    attn_output, _ = multihead_attention(query, decoding_batch, encoding_batch, 
                                         key_padding_mask=padding_masks, 
                                         attn_mask=attn_masks)
    return attn_output


def single_head_self_attention(encoding_cache, 
                               decoding_cache,
                               root_nodes, 
                               query_linear,
                               key_linear, 
                               device):
    attn_sqrt_dim = sqrt(decoding_cache.shape[-1])
    encoding_batch, decoding_batch, _, _ = \
        _convert_cache_to_tensor(encoding_cache, decoding_cache, root_nodes, device)
    # encoding_batch: (N, max_nodes_len, dim)
    # generate attention mask
    attn_masks = []
    max_nodes_len = encoding_batch.shape[1]
    max_ids_len = (max_nodes_len + 1) // 2
    attn_masks = np.zeros([len(root_nodes), max_ids_len, max_nodes_len])
    query_indices = np.zeros([len(root_nodes), max_ids_len])
    for batch_i, root in enumerate(root_nodes):
        node_queue = deque()
        node_queue.append([root, []])
        current_idx = 0  # order in preorder traversaling
        attn_masks[batch_i, :root.j - root.i + 1, 2 * (root.j - root.i) + 1:] = 1
        while len(node_queue) > 0:
            parent, ancestor_indices = node_queue.popleft()
            assert parent.decode_cache_id >= 0
            mask_indices = ancestor_indices + [current_idx]
            if not parent.is_leaf:
                node_queue.append([parent.left, mask_indices])
                node_queue.append([parent.right, mask_indices])
            else:
                # generate mask
                for masked_idx in mask_indices:
                    assert parent.i == parent.j
                    attn_masks[batch_i, parent.i, masked_idx] = 1
                    query_indices[batch_i, parent.i] = current_idx
            current_idx += 1
    attn_masks = torch.tensor(attn_masks, device=device, dtype=torch.bool)
    query_indices = torch.tensor(query_indices, device=device, dtype=torch.long)
    dim = decoding_batch.shape[-1]
    query_indices = query_indices.unsqueeze(2).repeat(1, 1, dim)
    query_decoding_batch = decoding_batch.gather(dim=1, index=query_indices)
    # (N, max_len, dim)
    query_batch = query_linear(query_decoding_batch)  # (N, max_len, dim)
    key_batch = key_linear(decoding_batch)  # (N, max_nodes_len, dim)
    attn_weights = query_batch @ key_batch.permute(0, 2, 1) / attn_sqrt_dim
    attn_weights.masked_fill_(attn_masks, -np.inf)
    attn_weights = F.softmax(attn_weights, dim=-1)  # (N, max_len, max_nodes_len)
    attn_output = attn_weights @ encoding_batch  # (N, max_len, dim)
    return attn_output