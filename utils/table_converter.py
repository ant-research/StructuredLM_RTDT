import math
from data_structure.r2d2_tree import PyNode, PyCell, PyTable


def recover_tree_nodes(nodes, i, j, cells, nodes_cache, seq_len):
    pos = i * seq_len + j
    if nodes_cache[pos] is None:
        nodes_cache[pos] = []
    else:
        return
    for node in nodes:
        if node.left_i != -1 and node.right_i != -1 and node.left_j != -1 and node.right_j != -1:
            left_idx = node.left_i * seq_len + node.left_j
            if nodes_cache[left_idx] is None:
                recover_tree_nodes(cells[left_idx].nodes, node.left_i, node.left_j, cells, nodes_cache, seq_len)
            left_nodes = nodes_cache[left_idx]
            assert left_nodes is not None

            right_idx = node.right_i * seq_len + node.right_j
            if nodes_cache[right_idx] is None:
                recover_tree_nodes(cells[right_idx].nodes, node.right_i, node.right_j, cells, nodes_cache, seq_len)
            right_nodes = nodes_cache[right_idx]
            assert right_nodes is not None

            left_node = nodes_cache[left_idx][node.left_idx]
            right_node = nodes_cache[right_idx][node.right_idx]
            nodes_cache[pos].append(PyNode(left_node, right_node, i, j, node.cache_id))
        else:
            assert node.left_i == node.right_i == node.left_j == node.right_j == -1
            if i == j:
                nodes_cache[pos].append(PyNode(None, None, i, j, node.cache_id))


def convert_cuda_tables(cuda_tables, tensor_cache):
    py_tables = []
    for cells in cuda_tables:
        seq_len = int(math.sqrt(len(cells)))
        if seq_len > 0:
            croot = cells[seq_len - 1]
            nodes_cache = [None for _ in range(len(cells))]
            recover_tree_nodes(croot.nodes, 0, seq_len - 1, cells, nodes_cache, seq_len)
            pyroot = PyCell(0, seq_len - 1, croot.best_tree_idx,
                            nodes_cache[seq_len - 1],
                            tensor_cache)
            py_tables.append(PyTable(pyroot, seq_len, len(croot.nodes)))
        else:
            py_tables.append(PyTable(None, seq_len, 0))
    return py_tables
