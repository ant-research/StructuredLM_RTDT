import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modeling_outputs import R2D2GenOutput

class GlueWrapper(nn.Module):
    def __init__(self, model, model_type, embed_dim, finetune_class_num=-1):
        # embedding dim is used to feed to r2d2
        # input dim is sued to feed to GPT
        super().__init__()
        self.model = model
        self.model_type = model_type
        self._embed_dim = embed_dim
        self._finetune_class_num = finetune_class_num
        if self._finetune_class_num != -1:
            self.extra_classifier = nn.Linear(self._embed_dim, self._finetune_class_num, bias=False)
    
    def from_pretrain(self, model_path):
        self.model.from_pretrain(model_path)
    
    def forward(self, chunk_input_ids=None, chunk_masks=None, input_ids=None, masks=None, eos_labels=None, group_ids=None, 
                atom_spans=None, span_ids=None, external_vocab_ids=None, 
                coeff=1.0, temperature=1.0, gpt_loss_coeff=1.0, cls_id=50259, min_label_id=50260):
        if self._finetune_class_num == -1:
            # print("--------------get into GENERATIVE successfully---------------")
            result = self.model.forward(chunk_input_ids=chunk_input_ids, chunk_masks=chunk_masks, input_ids=input_ids, masks=masks, eos_labels=eos_labels, group_ids=group_ids, 
                atom_spans=atom_spans, span_ids=span_ids, external_vocab_ids=external_vocab_ids, 
                coeff=coeff, temperature=temperature, gpt_loss_coeff=gpt_loss_coeff)
            return result
        else:
            # print("--------------get into DISCRIMINANT successfully--------------")
            labels = eos_labels - min_label_id  
            if self.model_type != "r2d2-gen-fast":
                eos_labels = None
            result = self.model.forward(chunk_input_ids=chunk_input_ids, chunk_masks=chunk_masks, input_ids=input_ids, masks=masks, eos_labels=eos_labels, group_ids=group_ids, 
                atom_spans=atom_spans, span_ids=span_ids, external_vocab_ids=external_vocab_ids, 
                coeff=coeff, temperature=temperature, gpt_loss_coeff=gpt_loss_coeff)
            hidden_states = result.hidden_states
            tgt_ids = result.tgt_ids
            labels = torch.tensor(labels, device=hidden_states.device)
            
            # print("hidden_states_size: ", hidden_states.shape)
            # print("tgt_ids: ", tgt_ids, "tgt_ids_size: ", tgt_ids.shape)

            # including cls, not including <bos>, origin-input-len=13
            # vanilla_gpt2: 14, r2d2-gen: 26, r2d2-gen-fast: 13
            # input: [10134,   257,   517, 20239,   837,   517, 34264,  8216,   621,   465,   584,  7328, 50259]
            # vanilla gpt tgt_ids: [50258, 10134,   257,   517, 20239,   837,   517, 34264,  8216,   621, 465,   584,  7328, 50259] len:14
            # r2d2-gen tgt_ids: [10134,   257, 50257,   517, 50257, 20239,   837, 50257,   517, 50257, 50257, 34264,  8216,   621, 50257, 50257,   465, 50257,   584, 50257, 7328, 50257, 50257, 50259,    -1,    -1] len: 26
            # r2d2-gen-fast tgt_ids: [10134,   257,   517, 20239,   837,   517, 34264,  8216,   621,   465, 584,  7328, 50259,  -100] len:14

            cls_pos = torch.where(tgt_ids == cls_id)[1]
            # print("cls_pos: ", cls_pos)
            if self.model_type != "gpt":
                cls_pos = cls_pos + 1
            # print("new_cls_pos: ", cls_pos)
            cls_hidden_states = hidden_states[torch.arange(tgt_ids.shape[0]), cls_pos, :]
            # print("--------------------test---------------------")
            # print("cls_hidden_states_size: ", cls_hidden_states.shape)
            # for id in range(cls_hidden_states.shape[0]):
            #     print(cls_hidden_states[id] == hidden_states[id][cls_pos[id]])
            pred = self.extra_classifier(cls_hidden_states)
            # print("pred: ", pred.float(), "pred_size: ", pred.shape, "labels: ", labels)
            glue_finetune_loss = F.cross_entropy(pred.float(), labels)

            return R2D2GenOutput(loss=result.loss + glue_finetune_loss, 
                                 logits=result.logits,
                                 hidden_states=hidden_states, cls_hidden_states=cls_hidden_states, 
                                 tgt_ids=tgt_ids, pred=pred, 
                                 gpt_loss=result.gpt_loss,
                                 action_loss=result.action_loss,
                                 inside_outside_loss=result.inside_outside_loss,
                                 parser_loss=result.parser_loss,
                                 glue_finetune_loss=glue_finetune_loss, 
                                 splits=result.splits)
