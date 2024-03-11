# coding=utf-8
# Copyright (c) 2021 Ant Group
# Author: Xiang Hu

from collections import namedtuple
import logging
from model.r2d2_base import R2D2Base
import torch.nn as nn
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


InsideGroup = namedtuple("InsideGroup", ["parent_ids", "candidate_e_ij_ids", "candidate_log_p_ids", 
                                         "idx2batch", "span_lens"])


POOLING_MODE = ["per_layer", "final_layer", "no_pooling"]

class PureTransformer(R2D2Base):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.position_embeddings = nn.Embedding(config.max_position_range, config.embedding_dim)
        self.position_embeddings.weight.data.normal_(mean=0, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(config.hidden_size, 
                                               config.num_attention_heads,
                                               config.intermediate_size,
                                               activation=F.gelu,
                                               batch_first=True)
        self.cls_embedding = nn.Parameter(torch.rand(config.hidden_size))
        self.cls_embedding.data.normal_(mean=0, std=0.02)
        self.span_attention = nn.TransformerEncoder(
            enc_layer, config.span_attention_num_layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.cls_dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.attention_probs_dropout_prob)
        )

        if hasattr(config, "max_feature_num"):
            if config.max_feature_num is not None:
                self.feature_embeddings = nn.Embedding(config.max_feature_num,
                                                       config.embedding_dim)
        
    def forward(self, 
                input_ids,
                target_ids=None,
                masks=None,
                add_cls=False,
                **kwargs):
        input_embedding = self.embedding(input_ids)
        batch_size = input_embedding.shape[0]
        pos_range = torch.arange(0, input_embedding.shape[1], device=self.device)  # (L)
        pos_range = pos_range.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, L)
        pos_embedding = self.position_embeddings(pos_range)  # (batch_size, L, dim)
        masks = masks.to(torch.bool)
        input_embedding = input_embedding + pos_embedding
        
        
        if add_cls:
            cls_exp = self.cls_embedding.unsqueeze(0).repeat(batch_size, 1)
            input_embedding = torch.cat([cls_exp.unsqueeze(1), input_embedding], dim=1)
            masks = torch.cat([masks[:, :1], masks], dim=1)
        input_embedding = self.layer_norm(input_embedding)
        input_embedding = self.dropout(input_embedding)
        output = self.span_attention(input_embedding, src_key_padding_mask=~masks)
        
        if add_cls:
            contextualized_token_embeddings = output[:, 1:, :]
            cls_logits = output[:, 0, :]
        else:
            contextualized_token_embeddings = output
            cls_logits = None
        logits = self.classifier(self.cls_dense(contextualized_token_embeddings))
        if target_ids is not None:
            lm_loss = F.cross_entropy(logits.permute(0, 2, 1), target_ids, ignore_index=-1)
        else:
            lm_loss = torch.zeros((1,), dtype=torch.float, device=self.device)
        # estimate cross entropy loss
        results = {}
        results['loss'] = lm_loss
        results['cls_embedding'] = cls_logits
        results['token_embeddings'] = contextualized_token_embeddings
        
        return results