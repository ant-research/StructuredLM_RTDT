from multiprocessing import pool
import os
from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
from model.fast_r2d2_iter_attn_share_abl import FastR2D2Plus
from model.topdown_parser import LSTMParser, TransformerParser
from model.fast_r2d2_functions import force_contextualized_inside_outside
from utils.model_loader import load_model
import logging


class FastR2D2IterClassification(nn.Module):
    """

        iterative up-and-down with top-down parser
        base model: fast_r2d2_iter.py

    """
    def __init__(self, config, label_num, transformer_parser=False, pretrain_dir=None, model_loss=False):
        super().__init__()
        self.r2d2 = FastR2D2Plus(config)
        self.parser = LSTMParser(config)
        # load pretrained model
        if pretrain_dir is not None:
            model_path = os.path.join(pretrain_dir, 'model.bin') # to find out
            self.r2d2.from_pretrain(model_path, strict=True)
            parser_path = os.path.join(pretrain_dir, 'parser.bin') # to find out
            load_model(self.parser, parser_path)
            logging.info('FastR2D2IterClassification load pretrained model successfully')

        self.classifier = nn.Linear(config.hidden_size, label_num)
        self.model_loss = model_loss
        
    def from_checkpoint(self, model_path):
        load_model(self, model_path)
        self.r2d2._tie_weights()

    def forward(self, input_ids: torch.Tensor,
                parser_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None,
                tgt_ids: torch.Tensor = None,
                pairwise: bool = False,
                noise_coeff: float = 1.0):
        s_indices = self.parser(parser_ids, attention_mask, noise_coeff=noise_coeff)
        results = self.r2d2(input_ids, tgt_ids=tgt_ids, masks=attention_mask, merge_trajectory=s_indices, 
                            pairwise=pairwise, recover_tree=True) 
        # target=None, pair_targets=None
        mlm_loss = results['loss']
        if torch.all(torch.isnan(mlm_loss)):
            mlm_loss.fill_(0.0)
        pooling_embedding = results['cls_embedding']
        # pooling_embedding = results['group_embeddings']
        logits = self.classifier(pooling_embedding)
        target_tree = results['trees'][-1]
        kl_loss = self.parser(input_ids, attention_mask,
                              split_masks=target_tree['split_masks'],
                              split_points=target_tree['split_points'])

        if self.training:
            loss = F.cross_entropy(logits, labels)
            return {"loss": [loss, mlm_loss, kl_loss]}
        else:
            ret_results = results
            ret_results["predict"] = F.softmax(logits, dim=-1)
            return ret_results # probs + trees for eval 