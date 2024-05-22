# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
from typing import List
import torch
import numpy as np

    
class CPPChartTableManager:
    def __init__(self, seq_lens, window_size, merge_orders, cache_id_offset, detach_cache_id_offset, group_ids=None, span_ids=None):
        # seq_lens: np array
        # merge_orders: np array
        self.seq_lens = seq_lens
        import cppbackend
        if group_ids is None:
            group_ids = [i for i in range(len(seq_lens))]
        if span_ids is None:
            span_ids = []
        assert len(group_ids) == len(seq_lens)
        self.cpp_tbl_mgr = cppbackend.TableManager(seq_lens, group_ids, merge_orders, window_size, 
                                                   cache_id_offset, detach_cache_id_offset, span_ids)
        self._root_ids = None
    
    @property
    def root_ids(self):
        return self._root_ids
    
    def construct_inside_groups(self, device):
        target_cache_ids_list = []
        span_ids_batch_list = []
        cache_groups_batch_list = []
        detach_cache_groups_batch_list = []
        total_time = None
        while not self.cpp_tbl_mgr.is_finished():
            tgt_cache_ids, span_ids, cache_ids, detach_cache_ids = self.cpp_tbl_mgr.step()
            target_cache_ids_list.append(tgt_cache_ids.to(device, non_blocking=True))
            span_ids_batch_list.append(span_ids.to(device, non_blocking=True))
            cache_groups_batch_list.append(cache_ids.to(device, non_blocking=True))
            detach_cache_groups_batch_list.append(detach_cache_ids.to(device, non_blocking=True))
        
        self._root_ids = self.cpp_tbl_mgr.root_ids().to(device, non_blocking=True)
        return target_cache_ids_list, span_ids_batch_list, cache_groups_batch_list, detach_cache_groups_batch_list
    
    # def best_trees(self, best_splits, atom_spans=None, terminal_only=False):
    #     if atom_spans is None:
    #         atom_spans = [torch.zeros((0,2))] * self.cpp_tbl_mgr.batch_size()
    #     else:
    #         atom_spans = [torch.tensor(spans) if len(spans) > 0 else torch.zeros((0, 2)) for spans in atom_spans]
    #     assert len(atom_spans) == self.cpp_tbl_mgr.batch_size()
    #     # np_arr = [t.data.numpy() for t in best_splits]
    #     # best_splits = np.concatenate(np_arr)
    #     # return targets, cache_ids
    #     splits, cache_ids = self.cpp_tbl_mgr.best_trees(best_splits, atom_spans, terminal_only)
    #     return splits, cache_ids

    def prepare_generation(self, score_orders, split_orders, atom_spans, input_ids, groups_ids, eos_id, reduce_id, max_input_len,
                           eos_labels=None):
        if atom_spans is None:
            atom_spans = [np.zeros((0,2))] * self.cpp_tbl_mgr.batch_size()
        else:
            atom_spans = [np.array(spans) if len(spans) > 0 else np.zeros((0, 2)) for spans in atom_spans]
        assert len(atom_spans) == self.cpp_tbl_mgr.batch_size()
        score_orders = [order.data.numpy() for order in score_orders]
        split_orders = [order.data.numpy() for order in split_orders]
        if eos_labels is None:
            eos_labels = np.full((groups_ids[-1] + 1), fill_value=eos_id)
        assert len(eos_labels) == groups_ids[-1] + 1
        span_masks, split_targets, ldr_cache_ids, position_ids, tgt_ids, token_indices, ext_ids = \
            self.cpp_tbl_mgr.prepare_generation(score_orders, split_orders, atom_spans, input_ids, groups_ids, 
                                                eos_labels, reduce_id, max_input_len)
        return span_masks, split_targets, ldr_cache_ids, position_ids, tgt_ids, token_indices, ext_ids
    
    def prepare_bilm(self, total_len, bos_id, eos_id):
        return self.cpp_tbl_mgr.prepare_bilm(total_len, bos_id, eos_id)