# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

import torch
import torch.nn.functional as F
from model.r2d2_common import (
    LMLossParam, 
    CacheSlots, 
    BOS_CACHE_ID,
    EOS_CACHE_ID
)
import logging


logger = logging.getLogger(__name__)


def cuda_default_lm_loss(loss_param: LMLossParam):
    model = loss_param.model
    tables = loss_param.chart_tables
    tensor_cache = loss_param.tensor_cache
    flatten_input_ids = loss_param.flatten_input_ids
    window_size = model.window_size
    cache_ids = torch.full([flatten_input_ids.shape[0] * window_size * 2],
                           0,
                           requires_grad=False,
                           dtype=torch.int,
                           device=model.device)
    tables.prepare_bilm(cache_ids, BOS_CACHE_ID, EOS_CACHE_ID)
    context_cache_ids = cache_ids.view(-1, window_size,
                                       2)[:flatten_input_ids.shape[0], :, :]

    e_ij, log_p_ij = tensor_cache.gather(
        context_cache_ids.flatten(),
        [CacheSlots.E_IJ, CacheSlots.LOG_P_IJ_SUM])

    e_ij = e_ij.view(*context_cache_ids.shape, model.input_dim)
    log_p_ij = log_p_ij.view(*context_cache_ids.shape)  # (total_len, comb_size, 2)
    weights = F.softmax(log_p_ij, dim=1)
    context_vec = torch.einsum("ijk,ijk...->ik...", weights, e_ij)  # (total_len, 2, dim)
    logits = model.infer(context_vec)
    return F.cross_entropy(logits, flatten_input_ids)
