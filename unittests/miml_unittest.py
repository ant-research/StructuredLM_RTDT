from unittest import TestCase
import json
import torch
from tqdm import tqdm
import numpy as np
from experiments.fast_r2d2_miml import FastR2D2MIL, FastR2D2MIML


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
  "max_positions":20,\
  "type_vocab_size": 2,\
  "max_positions": 20,\
  "vocab_size": 30522,\
  "encoder_num_hidden_layers": 1,\
  "decoder_num_hidden_layers": 1,\
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
  "parser_nhead": 2,\
  "parser_num_layers": 2,\
  "parser_max_len":300 \
}'

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def hasattr(self, val):
        return val in self


class MIMLUnittest(TestCase):
    def testRunnable(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        config = dotdict(json.loads(mini_r2d2_config))
        model = FastR2D2MIML(config, 100)
        model.to(device)
        model.eval()

        ids = [[1,2,3,4,5,6,7],
               [3,4,1,3,5,8,1],
               [5,4,2,7,8,3,2],
               [9,1,2,6,3,1,7]]
        masks = [[1,1,1,1,1,1,1],
                 [1,1,1,1,1,0,0],
                 [1,1,1,1,0,0,0],
                 [1,1,1,1,1,1,0]]
        labels = [[9, 12], [50, 80, 19]]
        ids = torch.tensor(ids, device=device)
        masks = torch.tensor(masks, device=device)
        result1 = model(ids, masks, labels = labels)

        ids = [[5,4,2,7,8,3,2],
               [9,1,2,6,3,1,7],
               [1,2,3,4,5,6,7],
               [3,4,1,3,5,8,1]]
        masks = [[1,1,1,1,0,0,0],
                 [1,1,1,1,1,1,0],
                 [1,1,1,1,1,1,1],
                 [1,1,1,1,1,0,0]]
        labels = [[50, 80, 19], [9, 12]]
        ids = torch.tensor(ids, device=device)
        masks = torch.tensor(masks, device=device)
        result2 = model(ids, masks, labels = labels)
        logits1 = result1['logits']
        logits2 = result2['logits']
        self.assertTrue(torch.dist(logits1[2], logits2[0]) < 0.001)
        self.assertTrue(torch.dist(logits1[3], logits2[1]) < 0.001)
        self.assertTrue(torch.dist(logits1[0], logits2[2]) < 0.001)
        self.assertTrue(torch.dist(logits1[1], logits2[3]) < 0.001)

        test_times, seq_range, batch_size = 10, 200, 30
        for _ in tqdm(range(test_times)):
            seq_lens = [max(1, int(np.random.rand() * seq_range)) for _ in range(batch_size)]
            max_len = max(seq_lens)
            masks = [[1] * seq_len + [0] * (max_len - seq_len) for seq_len in seq_lens]
            ids = np.random.randint(0, config.vocab_size, [batch_size, max_len])
            ids = torch.tensor(ids, device=device)
            masks = torch.tensor(masks, device=device)
            model.train()
            model(
                input_ids=ids,
                attention_mask=masks,
                labels=[[0,1,2] for _ in range(batch_size)],
                bilm_loss=True
            )
            model.eval()
            model(
                input_ids=ids,
                attention_mask=masks,
                labels=[[0,1,2] for _ in range(batch_size)]
            )