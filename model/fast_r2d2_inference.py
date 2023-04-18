from base64 import decode
from collections import deque
from functools import reduce
from typing import List, Tuple
from data_structure.r2d2_tree import PyNode
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


def force_encode(parser, r2d2, input_ids, attention_mask, atom_spans: List[List[Tuple[int]]]):
    # initialize tensor cache
    s_indices = parser(input_ids, attention_mask, atom_spans=atom_spans, add_noise=False)
    seq_lens = torch.sum(attention_mask, dim=1, dtype=torch.int)  # (batch_size, 1)
    seq_lens_np = seq_lens.to('cpu').data.numpy()

    # 打印树结构
    # tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # s_indices_merge_trajectories = s_indices.to('cpu').data.numpy()
    # seq_len_test = attention_mask[0].sum()
    # _, tree_str = get_tree_from_merge_trajectory(s_indices_merge_trajectories[0], seq_len_test, tokens)
    # print(tree_str)

    e_ij_cache = torch.full([sum(seq_lens) * 2, r2d2.input_dim], 0.0, device=r2d2.device)
    _, embedding = r2d2.initialize_embeddings(input_ids, seq_lens_np)
    e_ij_cache[0:embedding.shape[0]] = embedding
    encoding_batchs, root_ids, root_nodes = build_batch(s_indices, seq_lens_np)
    cache_id_offset = sum(seq_lens)
    for current_batch in encoding_batchs:        
        current_batch = torch.tensor(current_batch, device=r2d2.device)
        e_ikj = e_ij_cache[current_batch]
        e_ij, _ = r2d2.encode(e_ikj.unsqueeze(1))  # (?, 1, dim)
        e_ij = e_ij.squeeze(1)
        e_ij_cache[cache_id_offset: cache_id_offset + e_ij.shape[0]] = e_ij
        cache_id_offset += e_ij.shape[0]
    return e_ij_cache[root_ids], e_ij_cache, root_nodes


def force_decode(decoder, e_ij_cache, root_nodes, root_role_embedding=None):
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