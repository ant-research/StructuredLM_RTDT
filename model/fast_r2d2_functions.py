# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

from audioop import reverse
from collections import deque
from functools import reduce
from turtle import left
from typing import List, Tuple
from data_structure.r2d2_tree import PyNode
from data_structure.const_tree import SpanTree
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from utils.misc import padding
from math import sqrt



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)

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


def create_trees_by_split_points(split_points_batch, seq_lens):
    assert len(split_points_batch) == len(seq_lens)
    for split_points in split_points_batch:
        pass


def build_batch(s_indices, seq_lens):
    if isinstance(s_indices, torch.Tensor):
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
        s_indices = parser(input_ids, attention_mask, atom_spans=atom_spans, noise_coeff=0.0)
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


def force_decode_causal(e_ij_cache, root_nodes, model, device, task_embedding=None):
    encoding_batch, seq_masks, nodes_flatten_batch = _convert_cache_to_batch(e_ij_cache, root_nodes, device=device)
    # encoding_batch: (N, max_len, dim)
    # root_role_embedding, task_embedding (dim)
    N = encoding_batch.shape[0]
    dim = encoding_batch.shape[-1]
    task_embedding_ = task_embedding.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1)
    inputs = torch.cat([task_embedding_, encoding_batch], dim=1)  # (N, max_len + 1, dim)
    # gather visible indices
    attendable_indices_batch = []
    max_ancestors = 0
    max_nodes_len = 0
    recover_indices = []
    decode_cache_offset = 0
    for batch_i, root in enumerate(root_nodes):
        # expand root to cache id list
        node_queue = deque()
        node_queue.append([root, [0]])
        attendable_ancestors = [[0]]
        idx = 1
        while len(node_queue) > 0:
            parent, ancestors = node_queue.popleft()
            parent.decode_cache_id = decode_cache_offset
            decode_cache_offset += 1
            recover_indices.append([batch_i, idx])
            visible_nodes = [idx] + ancestors
            attendable_ancestors.append(visible_nodes)
            max_ancestors = max(max_ancestors, len(visible_nodes))
            if not parent.is_leaf:
                node_queue.append([parent.left, visible_nodes])
                node_queue.append([parent.right, visible_nodes])
            idx += 1
        max_nodes_len = max(max_nodes_len, len(attendable_ancestors))
        attendable_indices_batch.append(attendable_ancestors)
    # generate gather indices and masks
    # padding indices
    masks_batch = []
    for attendable_ancestors in attendable_indices_batch:
        masks = []
        for ancestors in attendable_ancestors:
            mask = [False] * len(ancestors) + [True] * (max_ancestors - len(ancestors))
            ancestors.extend([0] * (max_ancestors - len(ancestors)))
            masks.append(mask)
        masks.extend([[False] * max_ancestors] * (max_nodes_len - len(masks)))
        attendable_ancestors.extend([[0] * max_ancestors] * (max_nodes_len - len(attendable_ancestors)))
        masks_batch.append(masks)
    gather_indices = torch.tensor(attendable_indices_batch, dtype=torch.long, device=device)
    # (N, max_nodes_len, max_ancestors)
    gather_indices = gather_indices.view(N, -1).unsqueeze(-1).repeat(1, 1, dim)
    
    def get_target_fn(src):
        targets = src.gather(dim=1, index=gather_indices)  # (N, L * aL, dim)
        targets = targets.view(N, max_nodes_len, max_ancestors, dim)
        return targets
    attn_mask = torch.zeros((N, max_nodes_len, max_ancestors), device=device)
    masks_batch = torch.tensor(masks_batch, dtype=torch.bool, device=device)
    attn_mask.masked_fill_(masks_batch == True, -np.inf)
    outputs = model(inputs, get_target_fn, attn_mask)
    recover_indices = torch.tensor(recover_indices, device=device)
    return outputs[recover_indices[:, 0], recover_indices[:, 1], :], root_nodes

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


def pairwise_contextual_inside_outside(root_embeddings, 
                                       inside_enc, 
                                       outside_enc, 
                                       prev_group_embeddings):
    # root_embeddings: (batch_size, dim)
    # prev_group_embeddings: (batch_size // 2, dim)
    assert root_embeddings.shape[0] % 2 == 0
    batch_size = root_embeddings.shape[0]
    dim = root_embeddings.shape[-1]
    pairwise_root_embeddings = root_embeddings.view(batch_size // 2, 2, dim)
    _, group_embeddings = inside_enc(pairwise_root_embeddings.unsqueeze(1), prev_group_embeddings)
    group_embeddings = group_embeddings.squeeze(1)
    _, out_root_embeddings= outside_enc(group_embeddings, pairwise_root_embeddings.unsqueeze(1))
    return out_root_embeddings.view(batch_size, -1), group_embeddings

def group_inside_outside(root_embeddings, 
                         outside_root_embedding,
                         group_ids, 
                         inside_enc, 
                         outside_enc, 
                         device):
    # params: root_embeddings: (batch_size, dim)
    # params: groups_ids: e.g. [[1,2,3], [4,5], [6,7,8], [9]]
    # all root embedding recursively apply composition funtion sequentially
    
    # build composition order according to group_ids
    current_cache_id_offset = root_embeddings.shape[0]
    dim = root_embeddings.shape[-1]
    composition_pairs_list = []
    group_root_ids = []
    
    inside_cache = torch.full([2 * root_embeddings.shape[0], dim], 0.0, device=device)
    outside_cache = torch.full([2 * root_embeddings.shape[0], dim], 0.0, device=device)
    
    while len(group_ids):
        composition_pairs = []
        out_cache_ids = []
        next_group_ids = []
        for group in group_ids:
            if len(group) == 1:
                # 
                group_root_ids.append(group[0])
                continue
            else:
                composition_pairs.append(group[:2])
            next_group_ids.append([current_cache_id_offset] + group[2:])
            out_cache_ids.append(current_cache_id_offset)
            current_cache_id_offset += 1
        
        assert len(out_cache_ids) == len(composition_pairs)
        if len(out_cache_ids) > 0:
            composition_pairs_list.append([torch.tensor(composition_pairs, device=device),
                                        torch.tensor(out_cache_ids, device=device)])
        group_ids = next_group_ids
        
    inside_cache[:root_embeddings.shape[0], :] = root_embeddings
    for composition_ids, out_cache_ids in composition_pairs_list:
        # composition_ids: [current_size, 2]
        e_ij = inside_cache.index_select(0, composition_ids.flatten())
        e_ij = e_ij.view(*composition_ids.shape, dim)

        _, inside_repr = inside_enc(e_ij.unsqueeze(1))  # (current_size, 1, dim)
        inside_repr = inside_repr.squeeze(1)
        # print(f'out cache ids: {out_cache_ids.shape}, inside_repr: {inside_repr.shape}')
        scatter_indices = out_cache_ids.unsqueeze(1).repeat(1, dim)
        inside_cache.scatter(0, scatter_indices, inside_repr)

    group_root_ids = torch.tensor(group_root_ids, device=device)
    group_embedding = inside_cache.index_select(0, group_root_ids)
    outside_root_embeddings = outside_root_embedding.unsqueeze(0).\
        repeat(root_embeddings.shape[0], 1)
    outside_cache.scatter(0, group_root_ids.unsqueeze(1).repeat(1, dim), outside_root_embeddings)
    for composition_ids, out_cache_ids in reversed(composition_pairs):
        outside_parent = outside_cache.index_select(0, out_cache_ids)
        inside_children = inside_cache.index_select(0, composition_ids.flatten())
        inside_children = inside_children.view(*composition_ids.shape, -1)
        outside_children = outside_enc(outside_parent, inside_children)
        scatter_indices = composition_ids.flatten().unsqueeze(1).repeat(1, dim)
        outside_cache.scatter(0, scatter_indices, outside_children.view(-1, dim))
        
    return outside_cache, group_embedding

def rebuild_batch_indices(batch_indices, group_ids, device, empty_cell_id=0, align_batch=True):
    # batch_indices: (batch_size, max_seq_len), 0 for empty cells
    # print(f'batch_indices shape: {batch_indices.shape}')
    cell_num = (batch_indices != empty_cell_id).int().sum(dim=1)
    num_cumsum = cell_num.cumsum(dim=0)  # (batch_size)
    # print(f'cell_num shape: {cell_num.shape}, num_cumsum shape: {num_cumsum.shape}')
    prev_len = 0
    offset = 0
    group_lens = []
    for group in group_ids:
        group_len = num_cumsum[offset + len(group) - 1] - prev_len
        prev_len = num_cumsum[offset + len(group) - 1]
        offset += len(group)
        group_lens.append(group_len)
        
    max_group_len = max(group_lens) 
    rebuild_batch_indices = torch.full([len(group_ids), max_group_len], empty_cell_id, 
                                       dtype=torch.long, device=device)
    for group_id, group in enumerate(group_ids):
        st = 0
        for batch_id in group:
            ed = st + cell_num[batch_id]
            rebuild_batch_indices[group_id, st:ed] = batch_indices[batch_id, : cell_num[batch_id]]
            st = ed
    
    if align_batch:
        group2batch = []
        for group_id, group in enumerate(group_ids):
            group2batch.extend([group_id] * len(group))
            
        group2batch = torch.tensor(group2batch, device=device)
        return rebuild_batch_indices.index_select(dim=0, index=group2batch)
    else:
        return rebuild_batch_indices
    
def rebuild_batch_indices_cpu(batch_indices, group_ids, device, empty_cell_id=0, align_batch=True):
    # batch_indices: (batch_size, max_seq_len), 0 for empty cells
    # print(f'batch_indices shape: {batch_indices.shape}')
    cell_num = (batch_indices != empty_cell_id).sum(axis=1)
    num_cumsum = cell_num.cumsum(axis=0)  # (batch_size)
    # print(f'cell_num shape: {cell_num.shape}, num_cumsum shape: {num_cumsum.shape}')
    prev_len = 0
    offset = 0
    group_lens = []
    for group in group_ids:
        group_len = num_cumsum[offset + len(group) - 1] - prev_len
        prev_len = num_cumsum[offset + len(group) - 1]
        offset += len(group)
        group_lens.append(group_len)
        
    max_group_len = max(group_lens) 
    rebuild_batch_indices = torch.full([len(group_ids), max_group_len], empty_cell_id)
    for group_id, group in enumerate(group_ids):
        st = 0
        for batch_id in group:
            ed = st + cell_num[batch_id]
            rebuild_batch_indices[group_id, st:ed] = batch_indices[batch_id, : cell_num[batch_id]]
            st = ed
    
    if align_batch:
        group2batch = []
        for group_id, group in enumerate(group_ids):
            group2batch.extend([group_id] * len(group))
            
        group2batch = torch.tensor(group2batch, device=device)
        return rebuild_batch_indices[group2batch, :]
    else:
        return rebuild_batch_indices
    
    
def group_contextual_inside_outside(root_embeddings, 
                                    inside_enc, 
                                    outside_enc, 
                                    prev_group_embeddings,
                                    group_ids,
                                    device):
    # reformat representations according to group ids
    max_group_len = max(map(len, group_ids))
    group_num = len(group_ids)
    gather_indices = torch.full([group_num, max_group_len], fill_value=0, dtype=torch.long, device=device)

def force_contextualized_inside_outside(input_ids, masks, parser, model, device, pairwise=False, flatten_nodes_batch=None,
                                        atom_spans=None):
    s_indices = parser(input_ids, attention_mask=masks, atom_spans=atom_spans, noise_coeff=0.0)
    s_indices = s_indices.to('cpu', non_blocking=True)
    seq_lens_np = masks.sum(dim=1).to('cpu', non_blocking=True)
    
    nodes_per_height = [[] for _ in range(max(seq_lens_np))]
    cache_id = -1
    def get_cache_id():
        nonlocal cache_id
        cache_id += 1
        return cache_id
    
    root_ids = []
    batch_size = s_indices.shape[0]
    flatten_cache_ids = [None] * batch_size
    transformer_masks = []
    # flatten_nodes_batch: List[List[PyNode]]
    for batch_i in range(batch_size):
        flatten_nodes = None
        if flatten_nodes_batch is not None:
            flatten_nodes = []
            flatten_nodes_batch.append(flatten_nodes)
        seq_len = seq_lens_np[batch_i]
        indices = s_indices[batch_i].data.numpy()
        terminal_nodes = [PyNode(None, None, i, i, get_cache_id()) for i in range(seq_len)]
        nodes_per_height[0].extend(terminal_nodes)
        flatten_cache_ids[batch_i] = list(map(lambda x: x.cache_id, terminal_nodes))
        if flatten_nodes is not None:
            flatten_nodes.extend(terminal_nodes)
        spans_for_splits = [[terminal_nodes[i], terminal_nodes[i + 1]] for i in range(seq_len - 1)]

        last_span = terminal_nodes[0]
        for action_i in range(seq_len - 1):
            merge_pos = indices[action_i]
            assert merge_pos < seq_len - 1, f"input_ids {input_ids}, s_indices {s_indices}, seq_len {seq_len}, atom_spans {atom_spans}"
            left, right = spans_for_splits[merge_pos]
            assert left is not None
            assert right is not None
            # new_span = (left[0], right[1])
            new_span = PyNode(left, right, left.i, right.j, get_cache_id())
            if left.i - 1 >= 0:
                spans_for_splits[left.i - 1][1] = new_span
            if right.j < len(spans_for_splits):
                spans_for_splits[right.j][0] = new_span
            nodes_per_height[new_span.height].append(new_span)
            last_span = new_span
            flatten_cache_ids[batch_i].append(new_span.cache_id)
            if flatten_nodes is not None:
                flatten_nodes.append(new_span)
        root_ids.append(last_span.cache_id)
        transformer_masks.append([0] * len(flatten_cache_ids[batch_i]))
    
    if pairwise:
        assert batch_size % 2 == 0
    
    terminal_cache_ids = torch.tensor(list(map(lambda n: n.cache_id, nodes_per_height[0])), 
                                      dtype=torch.long).to(device, non_blocking=True)
    root_ids = torch.tensor(root_ids, dtype=torch.long).to(device, non_blocking=True)

    encoding_cache_id_pairs = []
    parent_cache_ids_list = []
    for nodes_batch in nodes_per_height[1:]:
        if len(nodes_batch) == 0:
            break
        cache_ids = []
        parent_cache_ids = []
        for node in nodes_batch:
            cache_ids.append([node.left.cache_id, node.right.cache_id])
            parent_cache_ids.append(node.cache_id)
        pair = torch.tensor(cache_ids, dtype=torch.long).to(device=device, non_blocking=True)
        parent_cache_ids = torch.tensor(parent_cache_ids, dtype=torch.long).to(device, non_blocking=True)
        encoding_cache_id_pairs.append(pair)
        parent_cache_ids_list.append(parent_cache_ids)
    
    _, _, input_embeddings = model.initialize_embeddings(input_ids, seq_lens_np)
    dim = input_embeddings.shape[-1]
    inside_cache = torch.full([cache_id + 1, dim], 0.0, device=device)
    leaf_indices=terminal_cache_ids.unsqueeze(1).repeat(1, dim)
    inside_cache.scatter_(dim=0, index=leaf_indices, src=input_embeddings)
    outside_cache = None
    group_embedding = model.empty_cell.unsqueeze(0).repeat(batch_size // 2, 1)
    
    if pairwise:
        flatten_cache_ids_pair = []
        transformer_masks = []
        for sent_a_ids, sent_b_ids in zip(flatten_cache_ids[0::2], flatten_cache_ids[1::2]):
            flatten_cache_ids_pair.append(sent_a_ids + sent_b_ids)
            transformer_masks.append([0] * (len(sent_a_ids) + len(sent_b_ids)))
        flatten_cache_ids = flatten_cache_ids_pair
    padding(flatten_cache_ids, 0)
    padding(transformer_masks, 1)
    flatten_cache_ids = torch.tensor(flatten_cache_ids, dtype=torch.long).to(device, non_blocking=True)
    transformer_masks = torch.tensor(transformer_masks, dtype=torch.bool).to(device, non_blocking=True)
    
    # force inside outside
    for iter_i in range(model.iter_times):
        for children_ids, parent_ids in zip(encoding_cache_id_pairs, parent_cache_ids_list):
            input_embeddings = inside_cache.index_select(dim=0, index=children_ids.flatten())
            input_embeddings = input_embeddings.view(*children_ids.shape, -1)  # (batch_size, 2, dim)
            if outside_cache is None:
                parent_ij = model.empty_cell.unsqueeze(0).repeat(input_embeddings.shape[0], 1)
            else:
                parent_ij = outside_cache.index_select(dim=0, index=parent_ids.flatten())
            _, c_ijk = model.inside_layers[iter_i](input_embeddings.unsqueeze(1), parent_ij)
            c_ijk = c_ijk.squeeze(1)
            inside_cache.scatter_(dim=0, index=parent_ids.unsqueeze(1).repeat(1, dim), src=c_ijk)
        
        root_embedding = inside_cache.index_select(dim=0, index=root_ids)
        if pairwise:
            # pair inside outside
            root_embedding, group_embedding = \
                pairwise_contextual_inside_outside(root_embedding, model.inside_layers[iter_i], 
                                                   model.outside_layers[iter_i], group_embedding)
        
        outside_cache = torch.full([cache_id + 1, dim], 0.0, device=device)
        outside_cache.scatter_(dim=0, index=root_ids.unsqueeze(1).repeat(1, dim), src=root_embedding)
        for children_ids, parent_ids in zip(reversed(encoding_cache_id_pairs), reversed(parent_cache_ids_list)):
            child_embeddings = inside_cache.index_select(dim=0, index=children_ids.flatten())
            child_embeddings = child_embeddings.view(*children_ids.shape, -1)
            parent_ij = outside_cache.index_select(dim=0, index=parent_ids)
            child_embeddings = child_embeddings.unsqueeze(1)
            _, out_ikj = model.outside_layers[iter_i](parent_ij, child_embeddings)
            out_ikj = out_ikj.squeeze(1)  # (batch_size, dim)
            outside_cache.scatter_(dim=0, index=children_ids.flatten().unsqueeze(1).repeat(1, dim), 
                                   src=out_ikj.view(-1, dim))
            
        inside_cache = torch.full([cache_id + 1, dim], 0.0, device=device)
        # update terminal node representations
        outside_input_embedding = outside_cache.gather(dim=0, index=leaf_indices)
        inside_cache.scatter_(dim=0, index=leaf_indices, src=outside_input_embedding)
        
    # apply self-attention on nodes
    outside_repr = outside_cache.index_select(dim=0, index=flatten_cache_ids.flatten())
    outside_repr = outside_repr.view(*flatten_cache_ids.shape, dim)  # (B, L, d)
    if pairwise:
        outside_repr = torch.cat([group_embedding.unsqueeze(1), outside_repr], dim=1)
        transformer_masks = torch.cat([torch.zeros(transformer_masks.shape[0], 1, device=device, dtype=torch.bool), transformer_masks], 
                                      dim=1)
    assert transformer_masks.dtype == torch.bool
    outputs = model.span_self_attention(outside_repr, src_key_padding_mask=transformer_masks)
    return outputs, s_indices