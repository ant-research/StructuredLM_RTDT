# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.model_loader import load_model
from datetime import datetime
from model.modeling_outputs import R2D2GenOutput


class GenerativeR2D2(nn.Module):
    def __init__(self, r2d2, gpt, vocab_size, r2d2_input_dim, embedding_dim, dropout_rate=0.2, ext_vocab_size=0):
        # embedding dim is used to feed to r2d2
        # input dim is sued to feed to GPT
        super().__init__()
        self.embedding_dim = embedding_dim  # embedding_dim > r2d2_input_dim
        self.r2d2_input_dim = r2d2_input_dim
        self.r2d2 = r2d2
        # self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.vocab_size = vocab_size

        self.enable_gpt = False
        if gpt is not None:
            self.enable_gpt = True
            self.gpt = gpt
            self.bos_embedding = nn.Parameter(torch.rand(self.embedding_dim))
            self.up_scale = nn.Linear(self.r2d2_input_dim, self.embedding_dim)
            self.dense = nn.Sequential(nn.Linear(self.embedding_dim, 4 * self.embedding_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout_rate),
                                       nn.Linear(4 * self.embedding_dim, self.embedding_dim))
        
        self.classifier = nn.Linear(self.embedding_dim, vocab_size, bias=False)
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        self.down_scale = nn.Linear(self.embedding_dim, self.r2d2_input_dim)


        self.insideoutside_dense = nn.Sequential(
            nn.Linear(r2d2_input_dim, 4 * r2d2_input_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * r2d2_input_dim, self.embedding_dim)
        )

        
        self._init_weights()
        self._tie_weights()

    def _init_weights(self):
        if self.enable_gpt:
            self.bos_embedding.data.normal_(mean=0, std=0.02)
        self.embeddings.weight.data.normal_(mean=0, std=0.02)

    def _tie_weights(self):
        self.classifier.weight = self.embeddings.weight

    def get_parser(self):
        return self.r2d2.parser
        
    def from_pretrain(self, model_path, strict=True):
        load_model(self, model_path, strict=strict)
        self._tie_weights()

    def forward(self, chunk_input_ids=None, chunk_masks=None, input_ids=None, masks=None, eos_labels=None, group_ids=None, 
                atom_spans=None, span_ids=None, external_vocab_ids=None, 
                coeff=1.0, temperature=1.0, past_key_values=None):
        batch_size = max(group_ids) + 1
        r2d2_input_ids = torch.where(chunk_input_ids != -100, chunk_input_ids, 0)
        input_embeddings = self.embeddings(r2d2_input_ids)
        r2d2_embeddings = self.down_scale(input_embeddings)
        max_input_len = chunk_input_ids.shape[1]
        
        ctx, outside_tgt, ldr_repr, position_ids, tgt_ids, token_indices, ext_ids, split_targets, l_height = \
            self.r2d2(r2d2_input_ids, chunk_masks, input_ids, masks, r2d2_embeddings, group_ids, 
                      max_input_len, atom_spans, coeff=coeff, temperature=temperature, span_ids=span_ids,
                      eos_labels=eos_labels, external_vocab_ids=external_vocab_ids)

        if self.training:
            # with torch.cuda.stream(self.parallel_stream):
            parser_loss = self.r2d2.parser_loss(ctx)
            outside_embeddings = self.r2d2.outside_embeddings(ctx)
            outside_logits = self.classifier(self.insideoutside_dense(outside_embeddings))
            lm_loss = F.cross_entropy(outside_logits, outside_tgt)
        else:
            parser_loss = lm_loss = 0

        loss = 0
        logits = None
        past_kv = None
        hidden_states = None
        if self.enable_gpt:
            # TODO: replace token position with the original token embedding
            gpt_input = self.up_scale(ldr_repr)
            gpt_input.scatter_(1, token_indices.unsqueeze(2).repeat(1, 1, input_embeddings.shape[-1]), 
                               input_embeddings.to(gpt_input.dtype))
            
            bos_emb = self.bos_embedding.unsqueeze(0).repeat(batch_size, 1)
            # position ids already considered <bos>
            cat_input = torch.cat([bos_emb.unsqueeze(1), gpt_input], dim=1)  # (group_size, L + 1, dim)
            # cat_input = self.layer_norm(cat_input)
            outputs = self.gpt(inputs_embeds=cat_input, position_ids=position_ids, past_key_values=past_key_values)
            hidden_states = outputs.last_hidden_state
            logits = self.classifier(self.dense(hidden_states))  # (group_size, L + 1, vocab)
            # print("logits_size: ", logits.shape, "tgt_ids_size: ", tgt_ids.shape)
            loss = F.cross_entropy(logits.permute(0, 2, 1).float(), tgt_ids, ignore_index=-1)
            past_kv = outputs.past_key_values
            
        # return loss + lm_loss + parser_loss, split_targets
        return R2D2GenOutput(struct_loss=lm_loss + l_height,
                             non_struct_loss=2 * loss + parser_loss,
                             logits=logits, 
                             hidden_states=hidden_states, tgt_ids=tgt_ids, 
                             gpt_loss=loss,
                             inside_outside_loss=lm_loss,
                             parser_loss=parser_loss,
                             past_kv=past_kv,
                             splits=split_targets)