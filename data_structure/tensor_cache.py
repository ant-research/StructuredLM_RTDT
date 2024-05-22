# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
from enum import Enum
import torch


class CacheType(Enum):
    NORMAL = 0
    DETACH = 1


class TensorCache:
    def __init__(self, max_window,
                 seq_lens,
                 cache_types,
                 dims,
                 placeholder_num,
                 device, 
                 iter_times=1,
                 total_cache_size=-1):
        self.placeholder_num = placeholder_num  #
        self.total_block_size = placeholder_num
        self.iter_times = iter_times

        if total_cache_size == -1:
            self.block_sizes = [0] * len(seq_lens)
            self._init_blocks_size(seq_lens, max_window)
            for i in range(len(seq_lens)):
                self.total_block_size += self.block_sizes[i]
        else:
            self.total_block_size += total_cache_size

        self._max_lengths = [placeholder_num] * len(cache_types)
        self._cache_num = len(cache_types)
        self.cache_types = cache_types
        self.caches = [None] * self._cache_num
        self.dims = dims
        self.device = device
        # dtype = torch.float16 if torch.is_autocast_enabled() else torch.float
        for i, cache_type in enumerate(cache_types):
            if cache_type == CacheType.NORMAL:
                self.caches[i] = torch.full((self.total_block_size, dims[i]), 0.0, dtype=torch.float32, device=device)
            elif cache_type == CacheType.DETACH:
                self.caches[i] = torch.full((self.total_block_size * 2, dims[i]), 0.0, dtype=torch.float32, device=device)

    @property
    def capacity(self):
        return self.total_block_size

    @property
    def detach_offset(self):
        return self.total_block_size

    def init_placeholders(self, cache_ids, values):
        for cache_id, value in zip(cache_ids, values):
            self.caches[cache_id][:self.placeholder_num] = value

    def _init_blocks_size(self, seq_lens, max_window):
        seq_num = len(seq_lens)
        for seq_i in range(seq_num):
            seq_len = seq_lens[seq_i]
            block_max_len = 0
            for layer_i in range(seq_len):
                if layer_i <= max_window:
                    block_max_len += (seq_len - layer_i) * (layer_i + 1)
                else:
                    block_max_len += (max_window + 1) * max_window
            self.block_sizes[seq_i] = block_max_len * self.iter_times

    def gather(self, indices, cache_ids):
        # Gather tensors according to CacheItem pairs
        tensors_gathered = []
        if isinstance(indices, torch.Tensor):
            gather_indices = indices
        else:
            gather_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        for cache_id in cache_ids:
            tensor_block = self.caches[cache_id]
            tensor_gather = tensor_block.index_select(dim=0, index=gather_indices)
            tensors_gathered.append(tensor_gather)
        return tensors_gathered

    def fill(self, cache_id_offset, cache_id_len, cache_ids, values):
        if len(cache_ids) != len(values):
            raise Exception('TensorCache::fill names and values mismatch')
        # cdef PyObject ** tensor_block
        for cache_id, value in zip(cache_ids, values):
            tensor_block = self.caches[cache_id]
            tensor_block[cache_id_offset: cache_id_offset + cache_id_len] = value
            if self.cache_types[cache_id] == CacheType.DETACH:
                detach_offset = self.total_block_size + cache_id_offset
                tensor_block[detach_offset: detach_offset + cache_id_len] = value.detach()

    def get(self, cache_id, idx):
        assert self.caches[cache_id] is not None
        return self.caches[cache_id][idx]

    def get_tensor_cache(self, cache_id):
        return self.caches[cache_id]

    def detach(self, idx):
        detach_idx = self.total_block_size + idx
        for cache_i, cache_type in enumerate(self.cache_types):
            if cache_type == CacheType.DETACH:
                self.caches[cache_i][detach_idx] = self.caches[cache_i][idx].detach()
        return detach_idx

    def scatter(self, indices, cache_ids, values):
        if isinstance(indices, torch.Tensor):
            scatter_indices = indices
        else:
            scatter_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        for cache_id, value in zip(cache_ids, values):
            tensor_block = self.caches[cache_id]
            dim = value.shape[-1]
            scatter_indices_ = scatter_indices.unsqueeze(1).repeat(1, dim)
            if value.dtype != tensor_block.dtype:
                value = value.to(tensor_block.dtype)
            tensor_block.scatter_(dim=0, index=scatter_indices_, src=value)