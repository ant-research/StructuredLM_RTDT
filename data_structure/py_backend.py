# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

from typing import List
import torch
import cppbackend
import numpy as np
from data_structure.r2d2_tree import PyNode


class LinkedNode:
    def __init__(self) -> None:
        # left_up   \ /  right_up
        # left     -- -- right 
        # left down / \  right down
        self._left = None
        self._right = None
        self._leftup = None
        self._rightup = None
        self._leftdown = None
        self._rightdown = None
        
        self.cell = None
        
    def pruned(self):
        self._left = None
        self._right = None
        self._leftup = None
        self._rightup = None
        self._leftdown = None
        self._rightdown = None
        self.cell.detached = True
        
    @property
    def left(self):
        return self._left
    
    @left.setter
    def left(self, other):
        if self._left != other:
            self._left = other
            if other is not None:
                other.right = self
            
    @property
    def right(self):
        return self._right
    
    @right.setter
    def right(self, other):
        if self._right != other:
            self._right = other
            if other is not None:
                other.left = self
            
    @property
    def leftup(self):
        return self._leftup
    
    @leftup.setter
    def leftup(self, other):
        if self._leftup != other:
            self._leftup = other
            if other is not None:
                other.rightdown = self
            
    @property
    def leftdown(self):
        return self._leftdown
    
    @leftdown.setter
    def leftdown(self, other):
        if self._leftdown != other:
            self._leftdown = other
            if other is not None:
                other.rightup = self
            
    @property
    def rightup(self):
        return self._rightup
    
    @rightup.setter
    def rightup(self, other):
        if self._rightup != other:
            self._rightup = other
            if other is not None:
                other.leftdown = self
            
    @property
    def rightdown(self):
        return self._rightdown
    
    @rightdown.setter
    def rightdown(self, other):
        if self._rightdown != other:
            self._rightdown = other
            if other is not None:
                other.leftup = self


class Cell:
    def __init__(self, i, j) -> None:
        # each cell conresponds to a node
        self.node = None
        self.i = i
        self.j = j
        self.cache_id = -1
        self.splits = []
        self.detached = False
        self.best_split = -1
        
    def get_detached_cache_id(self, detach_offset):
        if self.detached or self.i == self.j:
            return detach_offset + self.cache_id
        else:
            return self.cache_id


class ActiveCells:
    def __init__(self, window_size, seq_len, cell_table) -> None:
        # Assure: elements in cell_table[0:window_size + 1] are all initialized
        self._window_size = window_size
        self._cell_table = cell_table
        for layer_i in range(min(window_size + 1, seq_len)):
            left_previous = None
            for pos_i in range(seq_len - layer_i):
                cell_ij = cell_table[pos_i, pos_i + layer_i]
                if layer_i > 0:
                    cell_ij.splits = [_ for _ in range(pos_i, pos_i + layer_i)]
                node = LinkedNode()
                node.cell = cell_ij
                cell_ij.node = node
                node.left = left_previous
                left_previous = node
                if layer_i > 0:
                    leftdown = cell_table[pos_i, pos_i + layer_i - 1].node
                    rightdown = cell_table[pos_i + 1, pos_i + layer_i].node
                else:
                    leftdown = None
                    rightdown = None
                node.leftdown = leftdown
                node.rightdown = rightdown

    def _create_new_node(self, leftdown, rightdown, left, right, ld_most, rd_most):
        new_node = LinkedNode()
        i = leftdown.cell.i
        j = rightdown.cell.j
        new_cell = self._cell_table[i, j]
        # TODO: update splits
        current = ld_most
        
        while current != rd_most.right:
            # e.g. (i, k), (k + 1, j) k is the split position
            new_cell.splits.append(current.cell.j)
            current = current.right
        
        new_cell.node = new_node
        new_node.cell = new_cell
        new_node.leftdown = leftdown
        new_node.rightdown = rightdown
        new_node.left = left
        new_node.right = right
        return new_node
    
    def prune(self, node):
        node.cell.detached = True
        leftdown = node.leftdown
        rightdown = node.rightdown
        
        left_node = leftdown
        left_steps = 0
        while left_node is not None:
            if left_node.left is not None:
                left_node.left.right = left_node.rightup
            if left_node.rightup is not None:
                left_node.rightup.leftdown = left_node.leftdown
            left_steps += 1
            
            if left_node.leftup is not None:
                # continue pruning
                tmp = left_node.leftup
                left_node.pruned()
                left_node = tmp
            else:
                # meet cky boundary
                if left_node.rightup is not None:
                    tmp = left_node.rightup
                    left_steps += 1
                else:
                    tmp = left_node.right
                left_node.pruned()
                left_node = tmp
                break                
            
        while left_node.rightup is not None:
            left_node = left_node.rightup
            left_steps += 1

        right_node = rightdown
        right_steps = 0
        while right_node is not None:
            if right_node.right is not None:
                right_node.right.left = right_node.leftup
            if right_node.leftup is not None:
                right_node.leftup.rightdown = right_node.rightdown
            right_steps += 1
            
            if right_node.rightup is not None:
                tmp = right_node.rightup
                right_node.pruned()
                right_node = tmp
            else:
                if right_node.leftup is not None:
                    tmp = right_node.leftup
                    right_steps += 1
                else:
                    tmp = right_node.left
                right_node.pruned()
                right_node = tmp
                break
            
        while right_node.leftup is not None:
            right_node = right_node.leftup
            right_steps += 1
            
        assert left_steps == right_steps
        assert left_steps == self._window_size + 1
        
        # Tetris falling
        current = left_node.left if left_node.left is not None else left_node
        end = right_node.right if right_node.right is not None else right_node

        current_ld_most = current
        while current_ld_most.leftdown is not None:
            current_ld_most = current_ld_most.leftdown
        
        current_rd_most = current
        while current_rd_most.rightdown is not None:
            current_rd_most = current_rd_most.rightdown

        nodes_created = []
        while current != end:
            node_left = current.leftup
            node_right = current.right.rightup
            nodes_created.append(self._create_new_node(current, current.right, node_left, node_right, 
                                                       current_ld_most, current_rd_most))
            current = current.right
            current_ld_most = current_ld_most.right
            current_rd_most = current_rd_most.right
            
        return nodes_created
        

class CellTable:
    def __init__(self, seq_len) -> None:
        self.seq_len = seq_len
        self._cells = [[Cell(i,j) for j in range(seq_len)] for i in range(seq_len)]
        
    def __getitem__(self, ij):
        i, j = ij
        return self._cells[i][j]


class ChartTableManager:
    def __init__(self, seq_lens, window_size, merge_orders: List[List[PyNode]], cache_id_offset, detach_id_offset,):
        # merge_order: records the positions of cells to merge
        # format: [[[i11, j11], [i12, j12], ...],, 
        #          [[i21, j21], [i22, j22], ...]
        self._seq_lens = seq_lens
        self._window_size = window_size
        self._merge_orders = merge_orders
        self._cache_id_offset = cache_id_offset
        self._detach_id_offset = detach_id_offset
        self._tables = [CellTable(seq_len) for seq_len in seq_lens]
        self._active_cells = [ActiveCells(window_size, seq_len, self._tables[i]) 
                              for i, seq_len in enumerate(seq_lens)]
        self._cell_encoding_order = []
        self._root_ids = None
        
    @property
    def root_ids(self):
        return self._root_ids
    
    def best_trees(self, best_splits, terminal_only=False):
        for cell_batch, splits_batch in zip(self._cell_encoding_order, best_splits):
            for cell, best_split in zip(cell_batch, splits_batch):
                assert 0 <= best_split < len(cell.splits)
                cell.best_split = best_split
        
        targets = [[] for _ in range(len(self._seq_lens))]
        cache_ids = [[] for _ in range(len(self._seq_lens))]    
        for batch_i, table in enumerate(self._tables):
            root = table[0, table.seq_len - 1]
            to_visit = [root]
            while len(to_visit) > 0:
                current = to_visit.pop(-1)
                if len(current.splits) > 0:
                    k = current.splits[current.best_split]
                    targets[batch_i].append(k)
                    if not terminal_only:
                        cache_ids[batch_i].append(current.cache_id)
                    to_visit.append(table[k + 1, current.j])
                    to_visit.append(table[current.i, k])
                else:
                    targets[batch_i].append(-1)
                    cache_ids[batch_i].append(current.cache_id)
        return targets, cache_ids
    
    def construct_inside_groups(self, device):
        # generate batch_indices, idx2batch, cache_ids
        current_step = 0
        max_steps = max(self._seq_lens)
        
        # initialized cache ids for terminal cells
        current_cache_id = self._cache_id_offset
        detach_id_offset = self._detach_id_offset
        for table in self._tables:
            for i in range(table.seq_len):
                table[i, i].cache_id = current_cache_id
                current_cache_id += 1
        
        target_cache_ids_list = []
        cache_groups_batch_list = []
        detach_cache_groups_batch_list = []
        # batch_indices = [[] for i in range(len(self._seq_lens))]
        # cache_idx = cache_id_offset
        # for i in range(len(self._seq_lens)):
        #     for _ in range(self._seq_lens[i]):
        #         batch_indices[i].append(cache_idx)
        #         cache_idx += 1
        for current_step in range(1, max_steps):
            target_cache_ids = []  # (total_cell)
            cache_groups_batch = []  # (total_cell, split_size, 2)
            detach_cache_groups_batch = []
            cell_encoding_order = []
            for batch_i, seq_len in enumerate(self._seq_lens):
                table = self._tables[batch_i]
                if current_step < seq_len:
                    if current_step <= self._window_size:
                        for i in range(seq_len - current_step):
                            j = i + current_step
                            table[i, j].cache_id = current_cache_id
                            current_cache_id += 1
                            target_cache_ids.append(table[i, j].cache_id)
                            # batch_indices[batch_i].append(table[i, j].cache_id)
                            cache_groups = []
                            detach_cache_groups = []
                            for k in table[i, j].splits:
                                cache_groups.append([table[i, k].cache_id, table[k + 1, j].cache_id])
                                ik_detach = table[i, k].get_detached_cache_id(detach_id_offset)
                                kj_detach = table[k + 1, j].get_detached_cache_id(detach_id_offset)
                                detach_cache_groups.append([ik_detach, kj_detach])
                            cell_encoding_order.append(table[i, j])
                            cache_groups_batch.append(cache_groups)
                            detach_cache_groups_batch.append(detach_cache_groups)
                    else:
                        merge_span = self._merge_orders[batch_i][current_step - self._window_size - 1]
                        i = merge_span.i
                        j = merge_span.j
                        assert table[i, j].node is not None
                        nodes_created = self._active_cells[batch_i].prune(table[i, j].node)
                        
                        for node in nodes_created:
                            i = node.cell.i
                            j = node.cell.j
                            node.cell.cache_id = current_cache_id
                            current_cache_id += 1
                            target_cache_ids.append(table[i, j].cache_id)
                            # batch_indices[batch_i].append(table[i, j].cache_id)
                            cache_groups = []
                            detach_cache_groups = []
                            for k in node.cell.splits:
                                cache_groups.append([table[i, k].cache_id, table[k + 1, j].cache_id])
                                ik_detach = table[i, k].get_detached_cache_id(detach_id_offset)
                                kj_detach = table[k + 1, j].get_detached_cache_id(detach_id_offset)
                                detach_cache_groups.append([ik_detach, kj_detach])
                            cache_groups_batch.append(cache_groups)
                            detach_cache_groups_batch.append(detach_cache_groups)
                            cell_encoding_order.append(node.cell)
            
            self._cell_encoding_order.append(cell_encoding_order)
            target_cache_ids_list.append(torch.tensor(target_cache_ids).to(device, non_blocking=True))
            cache_groups_batch_list.append(torch.tensor(cache_groups_batch).to(device, non_blocking=True))
            detach_cache_groups_batch_list.append(torch.tensor(detach_cache_groups_batch).to(device, non_blocking=True))
        
        # padding batch_indices
        # max_padding_len = max([len(indices) for indices in batch_indices])
        # for indices in batch_indices:
        #     indices.extend([0] * (max_padding_len - len(indices)))
            
        # batch_indices = torch.tensor(batch_indices).to(device, non_blocking=True)
        
        root_ids = []
        for table in self._tables:
            root_ids.append(table[0, table.seq_len - 1].cache_id)
 
        self._root_ids = torch.tensor(root_ids, dtype=torch.long).to(device, non_blocking=True)
        
        return target_cache_ids_list, cache_groups_batch_list, detach_cache_groups_batch_list
    
    
class CPPChartTableManager:
    def __init__(self, seq_lens, window_size, merge_orders, cache_id_offset, detach_cache_id_offset):
        # seq_lens: np array
        # merge_orders: np array
        self.seq_lens = seq_lens
        self.cpp_tbl_mgr = cppbackend.TableManager(seq_lens, merge_orders, window_size, 
                                                   cache_id_offset, detach_cache_id_offset)
        self._root_ids = None
    
    @property
    def root_ids(self):
        return self._root_ids
    
    def construct_inside_groups(self, device):
        target_cache_ids_list = []
        cache_groups_batch_list = []
        detach_cache_groups_batch_list = []
        while not self.cpp_tbl_mgr.is_finished():
            tgt_cache_ids, cache_ids, detach_cache_ids = self.cpp_tbl_mgr.step()
            target_cache_ids_list.append(tgt_cache_ids.to(device, non_blocking=True))
            cache_groups_batch_list.append(cache_ids.to(device, non_blocking=True))
            detach_cache_groups_batch_list.append(detach_cache_ids.to(device, non_blocking=True))
        
        self._root_ids = self.cpp_tbl_mgr.root_ids().to(device, non_blocking=True)
        return target_cache_ids_list, cache_groups_batch_list, detach_cache_groups_batch_list
        # max_steps = max(self.seq_lens)
        # target_cache_ids_list = []
        # cache_groups_batch_list = []
        # detach_cache_groups_batch_list = []
        # for _ in range(1, max_steps):
        #     tgt_cache_ids, cache_ids, detach_cache_ids = self.cpp_tbl_mgr.step()
        #     target_cache_ids_list.append(tgt_cache_ids.to(device, non_blocking=True))
        #     cache_groups_batch_list.append(cache_ids.to(device, non_blocking=True))
        #     detach_cache_groups_batch_list.append(detach_cache_ids.to(device, non_blocking=True))
        
        # self._root_ids = self.cpp_tbl_mgr.root_ids().to(device, non_blocking=True)
        # return target_cache_ids_list, cache_groups_batch_list, detach_cache_groups_batch_list
    
    def best_trees(self, split_orders, atom_spans=None, terminal_only=False):
        if atom_spans is None:
            atom_spans = [np.zeros((0,2))] * self.cpp_tbl_mgr.batch_size()
        else:
            atom_spans = [np.array(spans) if len(spans) > 0 else torch.zeros((0, 2)) for spans in atom_spans]
        assert len(atom_spans) == self.cpp_tbl_mgr.batch_size()
        split_orders = [order.data.numpy() for order in split_orders]
        # np_arr = [t.data.numpy() for t in best_splits]
        # best_splits = np.concatenate(np_arr)
        # return targets, cache_ids
        splits, cache_ids = self.cpp_tbl_mgr.best_trees(split_orders, atom_spans, terminal_only)
        return splits, cache_ids
    
    def prepare_bilm(self, total_len, bos_id, eos_id):
        return self.cpp_tbl_mgr.prepare_bilm(total_len, bos_id, eos_id)