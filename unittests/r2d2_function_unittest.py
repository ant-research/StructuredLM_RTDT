from collections import deque
from curses import noecho
import json
from unittest import TestCase

import torch
from data_structure.r2d2_tree import PyNode
from data_structure.const_tree import SpanTree
from model.fast_r2d2_functions import force_encode_given_trees
from experiments.preprocess import convert_tree_to_node
from model.r2d2_cuda import R2D2Cuda
from unittests.unittest_config import dotdict, mini_r2d2_config


class R2D2FunctionUnittest(TestCase):
    def testR2D2Function(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        mini_config = dotdict(json.loads(mini_r2d2_config))
        r2d2 = R2D2Cuda(mini_config)
        r2d2.eval()
        r2d2.to(device)

        t_11 = PyNode(None, None, 0, 0, -1)
        t_22 = PyNode(None, None, 1, 1, -1)
        t_33 = PyNode(None, None, 2, 2, -1)
        t_12 = PyNode(t_11, t_22, 0, 1, -1)
        root1 = PyNode(t_12, t_33, 0, 2, -1) # ((0, 1) 2)

        t_11 = PyNode(None, None, 0, 0, -1)
        t_22 = PyNode(None, None, 1, 1, -1)
        t_33 = PyNode(None, None, 2, 2, -1)
        t_44 = PyNode(None, None, 3, 3, -1)
        t_12 = PyNode(t_11, t_22, 0, 1, -1)
        t_34 = PyNode(t_33, t_44, 2, 3, -1)
        root2 = PyNode(t_12, t_34, 0, 3, -1)  # ((0,1),(2,3))

        input_ids = [[3, 4, 5, 0],
                     [4, 6, 8, 9]]
        attn_mask = [[1,1,1,0],
                     [1,1,1,1]]
        input_ids = torch.tensor(input_ids, device=device)
        attn_mask = torch.tensor(attn_mask, device=device)
        root_repr1, _, _ = force_encode_given_trees([root1, root2], r2d2, input_ids, attn_mask)

        t_11 = PyNode(None, None, 0, 0, -1)
        t_22 = PyNode(None, None, 1, 1, -1)
        t_33 = PyNode(None, None, 2, 2, -1)
        t_12 = PyNode(t_11, t_22, 0, 1, -1)
        root1 = PyNode(t_12, t_33, 0, 2, -1)

        t_11 = PyNode(None, None, 0, 0, -1)
        t_22 = PyNode(None, None, 1, 1, -1)
        t_33 = PyNode(None, None, 2, 2, -1)
        t_44 = PyNode(None, None, 3, 3, -1)
        t_12 = PyNode(t_11, t_22, 0, 1, -1)
        t_34 = PyNode(t_33, t_44, 2, 3, -1)
        root2 = PyNode(t_12, t_34, 0, 3, -1)
        input_ids = [[4, 6, 8, 9],
                     [3, 4, 5, 0]]
        attn_mask = [[1,1,1,1],
                     [1,1,1,0]]
        input_ids = torch.tensor(input_ids, device=device)
        attn_mask = torch.tensor(attn_mask, device=device)
        root_repr2, _, _ = force_encode_given_trees([root2, root1], r2d2, input_ids, attn_mask)
        self.assertTrue(torch.dist(root_repr1, torch.flip(root_repr2, dims=[0])) < 0.01)

    
    def testSpanTreeToNodes(self):
        def print_tree(node):
            if node.left is None:
                return "{}{}".format(node.i,node.j)
            return "({},{})".format(print_tree(node.left),print_tree(node.right))
        t_00 = SpanTree(0,0,[])
        t_11 = SpanTree(1,1,[])
        t_22 = SpanTree(2,2,[])
        t_33 = SpanTree(3,3,[])
        t_44 = SpanTree(4,4,[])
        t_55 = SpanTree(5,5,[])
        t_24 = SpanTree(2,4,[t_22, t_33, t_44])
        t_14 = SpanTree(1,4,[t_11,t_24])
        root = SpanTree(0,4,[t_00,t_14, t_55])
        node_tree = convert_tree_to_node(root)
        result_true = "((00,(11,((22,33),44))),55)"
        self.assertEqual(print_tree(node_tree),result_true)

        # genearate a random span tree
