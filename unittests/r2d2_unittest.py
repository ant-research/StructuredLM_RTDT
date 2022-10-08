from unittest import TestCase
import tqdm
import numpy as np
import json
import torch

from model.r2d2 import R2D2

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def hasattr(self, val):
        return val in self


mini_r2d2_config = '{\
  "architectures": [\
    "Bert"\
  ],\
  "model_type": "bert",\
  "attention_probs_dropout_prob": 0.1,\
  "hidden_act": "gelu",\
  "hidden_dropout_prob": 0.1,\
  "embedding_dim": 64,\
  "hidden_size": 64,\
  "initializer_range": 0.02,\
  "intermediate_size": 256,\
  "max_role_embeddings": 4,\
  "num_attention_heads": 8,\
  "type_vocab_size": 2,\
  "vocab_size": 30522,\
  "encoder_num_hidden_layers": 3,\
  "pad_token_id": 0,\
  "bos_token_id": 4,\
  "eos_token_id": 5,\
  "cls_token_id": 101,\
  "sum_token_id": 7,\
  "mask_token_id": 103,\
  "nsp_token_id": 8,\
  "lr_token_id": 9,\
  "rr_token_id": 10,\
  "eot_token_id": 11,\
  "tree_mask_token_id": 12,\
  "policy_token_id": 6,\
  "window_size": 4\
}'

class TestFastR2D2(TestCase):
    def test_cuda_tables(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        config = dotdict(json.loads(mini_r2d2_config))
        model = R2D2(config)
        model.eval()
        model.to(device)
        
        ids = [[0,1,2,3,4,5],
               [0,1,2,3,4,5]]
        masks = [[1,1,1,1,0,0],
                 [1,1,1,1,1,1]]
        ids = torch.tensor(ids, device=device)
        masks = torch.tensor(masks, device=device)
        loss1, _ = model(ids, masks)
        
        ids = [[0,1,2,3,4,5],
               [0,1,2,3,4,5]]
        masks = [[1,1,1,1,1,1],
                 [1,1,1,1,0,0]]
        ids = torch.tensor(ids, device=device)
        masks = torch.tensor(masks, device=device)
        loss2, _ = model(ids, masks)
        self.assertTrue(torch.abs(loss1 - loss2) < 1e-7)