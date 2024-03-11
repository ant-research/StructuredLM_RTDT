from imp import load_module
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.pure_transformer import PureTransformer


class TransformerDownstream(nn.Module):
    def __init__(self, config, label_num) -> None:
        super().__init__()
        self.model = PureTransformer(config)
        self.classifier = nn.Linear(config.hidden_size, label_num)
        
    def from_pretrain(self, pretrain_dir):
        self.model.from_pretrain(os.path.join(pretrain_dir, 'model.bin'), strict=True)
        self.model.cls_embedding.data.normal_(mean=0, std=0.02)
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None,
                tgt_ids: torch.Tensor = None):
        results = self.model(input_ids, masks=attention_mask, target_ids=tgt_ids, add_cls=True)
        lm_loss = results['loss']
        if torch.any(torch.isnan(lm_loss)):
            lm_loss.fill_(0.0)
        cls_logits = results['cls_embedding']
        logits = self.classifier(cls_logits)
        
        if self.training:
            cls_loss = F.cross_entropy(logits, labels)
            return {"loss": [cls_loss, lm_loss]}
        else:
            ret_results = results
            ret_results["predict"] = F.softmax(logits, dim=-1)
            return ret_results # probs + trees for eval 