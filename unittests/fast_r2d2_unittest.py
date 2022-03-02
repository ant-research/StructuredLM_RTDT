from model.fast_r2d2_downstream import FastR2D2Classification
from unittest import TestCase
import tqdm
import numpy as np
import json
import torch

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
  "window_size": 4,\
  "tie_decoder": false,\
  "parser_hidden_dim": 64,\
  "parser_input_dim": 32,\
  "parser_num_layers": 2\
}'

class TestFastR2D2(TestCase):
    def test_cuda_tables(self):
        if not torch.cuda.is_available():
            return
        device = torch.device('cuda:0')
        config = dotdict(json.loads(mini_r2d2_config))
        model = FastR2D2Classification(config, 2)
        model.to(device)
        
        test_times, seq_range, batch_size = 1, 20, 4
        for step in tqdm.tqdm(range(test_times)):
            seq_lens = [max(1, int(np.random.rand() * seq_range)) for _ in range(batch_size)]
            max_len = max(seq_lens)
            masks = [[1] * seq_len + [0] * (max_len - seq_len) for seq_len in seq_lens]
            ids = np.random.randint(0, config.vocab_size, [batch_size, max_len])
            ids = torch.tensor(ids, device=device)
            masks = torch.tensor(masks, device=device)
            model(
                input_ids=ids,
                attention_mask=masks
            )