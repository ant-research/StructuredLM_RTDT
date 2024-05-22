import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modeling_outputs import R2D2GenOutput

class XSumWrapper(nn.Module):
    # can be treated as xsum_generator
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def from_pretrain(self, model_path):
        self.model.from_pretrain(model_path)
    
    def forward(self, chunk_input_ids=None, chunk_masks=None, input_ids=None, masks=None, eos_labels=None, group_ids=None, 
                atom_spans=None, span_ids=None, external_vocab_ids=None, 
                coeff=1.0, temperature=1.0, gpt_loss_coeff=1.0):
        result = self.model.forward(chunk_input_ids=chunk_input_ids, chunk_masks=chunk_masks, input_ids=input_ids, masks=masks, eos_labels=eos_labels, group_ids=group_ids, 
                atom_spans=atom_spans, span_ids=span_ids, external_vocab_ids=external_vocab_ids, 
                coeff=coeff, temperature=temperature, gpt_loss_coeff=gpt_loss_coeff)
        return result
