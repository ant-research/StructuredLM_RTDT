# coding=utf-8
# Copyright (c) 2023 Ant Group
# Author: Xiang Hu

import torch
import torch.nn.functional as F
from data_structure.tensor_cache import CacheType, TensorCache
from model.r2d2_common import (
    LMLossParam, 
    CacheSlots, 
    BOS_CACHE_ID,
    EOS_CACHE_ID
)
import logging
import numpy as np
from collections import deque
import random
from model.tree_encoder import UniLMEncoder
from utils.table_converter import convert_cuda_tables
from utils.tree_utils import get_tree_from_merge_trajectory
from utils.vocab_builder import convert_tree_to_wordtree


logger = logging.getLogger(__name__)


def cuda_default_lm_loss(loss_param: LMLossParam):
    model = loss_param.model
    tables = loss_param.chart_tables
    tensor_cache = loss_param.tensor_cache
    flatten_input_ids = loss_param.flatten_input_ids
    cache_ids = tables.prepare_bilm(flatten_input_ids.shape[0], BOS_CACHE_ID, EOS_CACHE_ID)
    cache_ids = cache_ids.to(model.device)
    context_cache_ids = cache_ids.view(-1, 2)[:flatten_input_ids.shape[0], :]

    e_ij = tensor_cache.gather(
        context_cache_ids.flatten(),
        [CacheSlots.E_IJ])[0]

    e_ij = e_ij.view(*context_cache_ids.shape, model.input_dim)
    logits = model.infer(e_ij)
    return F.cross_entropy(logits, flatten_input_ids)
