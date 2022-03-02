import numpy as np
from utils.table_converter import PyNode


def get_token_tree(root, tokens):
    if root.left is not None and root.right is not None:
        return '({} {})'.format(get_token_tree(root.left, tokens), get_token_tree(root.right, tokens))
    else:
        return tokens[root.pos]


def get_tree_from_merge_trajectory(merge_trajectory: np.array, seq_len, tokens):
    nodes = [PyNode(None, None, i, i, -1) for i in range(seq_len)]
    for action_i in range(seq_len - 1):
        merge_pos = merge_trajectory[action_i]
        assert merge_pos < len(nodes) - 1
        left = nodes.pop(merge_pos)
        right = nodes.pop(merge_pos)
        new_node = PyNode(left, right, left.i, right.j, -1)
        nodes.insert(merge_pos, new_node)
        for idx in range(action_i, seq_len - 1):
            if merge_trajectory[idx] > merge_pos:
                merge_trajectory[idx] -= 1
    assert len(nodes) == 1
    return nodes[0], get_token_tree(nodes[0], tokens)


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
