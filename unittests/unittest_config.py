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
  "intermediate_size": 128,\
  "max_role_embeddings": 4,\
  "num_attention_heads": 8,\
  "max_positions":20,\
  "type_vocab_size": 2,\
  "max_positions": 20,\
  "vocab_size": 1000,\
  "encoder_num_hidden_layers": 3,\
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
  "parser_max_len":300\
}'

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def hasattr(self, val):
        return val in self