from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class R2D2GenOutput:
    struct_loss: Optional[torch.FloatTensor] = None,
    non_struct_loss: Optional[torch.FloatTensor] = None,
    action_logits: Optional[torch.FloatTensor] = None,
    logits: Optional[torch.FloatTensor] = None,
    hidden_states: Optional[torch.FloatTensor] = None,
    cls_hidden_states: Optional[torch.FloatTensor] = None,
    tgt_ids: Optional[torch.LongTensor] = None, 
    pred: Optional[torch.FloatTensor] = None, 
    splits: Optional[torch.LongTensor] = None,
    gpt_loss: Optional[torch.FloatTensor] = None,
    action_loss: Optional[torch.FloatTensor] = None,
    inside_outside_loss: Optional[torch.FloatTensor] = None,
    parser_loss: Optional[torch.FloatTensor] = None, 
    glue_finetune_loss: Optional[torch.FloatTensor] = None,
    past_kv: Optional[torch.FloatTensor] = None
