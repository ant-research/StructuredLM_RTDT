from model.generative_r2d2 import GenerativeR2D2
import torch.nn.functional as F
import torch
from enum import Enum
from typing import Dict
import codecs
import numpy as np
from model.stack_state import ActionType, Node, State


class VanillaGPTEvaluator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def perplexity(self, input_ids):
        # to validate the correctness of calling transformers interfaces
        # assert input_ids.shape[0] == 1, "currently only support 1 batch size"
        past_key_values=None
        total_len = input_ids.shape[0]
        model_input = torch.zeros((total_len + 1,), dtype=torch.long, device=input_ids.device)
        model_input[0] = self.model.bos_id
        model_input[1:] = input_ids
        ppl_sum = 0
        for step in range(total_len):
            result = self.model.gpt(model_input[step].unsqueeze(0), past_key_values=past_key_values, return_dict=True)
            past_key_values = result.past_key_values
            for kv_layer in past_key_values:
                k, v = kv_layer
            probs = F.softmax(result.logits, dim=-1)
            ppl_sum += -torch.log(probs[0][input_ids[step]])
        return ppl_sum / total_len


class R2D2GenEvaluator:
    def __init__(self, model, config, device, beam_size=1, ext_vocab: Dict[str, int]=None):
        self.model = model
        model.to(device)
        self.model.eval()
        self.hidden_size = model.r2d2_input_dim
        self.gpt_input_size = model.embedding_dim
        self.beam_size = beam_size
        self.layer_num = config.n_layer
        self.head_num = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.reduce_id = model.r2d2.reduce_id
        self.ext_vocab = ext_vocab
        self.device = device

    def tree_enc(self, src, span_embedding=None):
        return self.model.r2d2.inside_enc(src, span_embeds_override=span_embedding)

    def split_past_kv(self, past_key_values, beam_size):
        split_past_kvs = []
        for kv_layer in past_key_values:
            k, v = kv_layer
            assert k.shape[0] % beam_size == 0
            assert v.shape[0] % beam_size == 0

            k = k.view(-1, beam_size, k.shape[-3], k.shape[-2], k.shape[-1])
            v = v.view(-1, beam_size, k.shape[-3], k.shape[-2], k.shape[-1])

            split_past_kvs.append((k, v))
        return split_past_kvs

    def merge_past_kv(self, past_key_values):
        merge_past_kvs = []
        for kv_layer in past_key_values:
            k, v = kv_layer

            k = k.view(-1, k.shape[-3], k.shape[-2], k.shape[-1])
            v = v.view(-1, k.shape[-3], k.shape[-2], k.shape[-1])

            merge_past_kvs.append((k, v))
        return merge_past_kvs

    @torch.no_grad()
    def beam_search(self, input_ids, masks=None, atom_spans=None):
        # input_ids: (batch_size, max_len)
        # masks: (batch_size, max_len)
        assert input_ids.shape[0] == 1, "currently only support 1 batch size"
        if masks is None:
            masks = torch.ones_like(input_ids)
        seq_lens = masks.sum(dim=1)
        seq_lens_np = seq_lens.to(device='cpu').data.numpy()
        max_actions = max(seq_lens_np) * 2 - 1
        batch_size = input_ids.shape[0]
        input_ids_np = input_ids.to('cpu', non_blocking=True)
        # print(f'params: {batch_size}, {max_beam_size}, {max_actions}, {self.hidden_size}')
        encoding_beams = torch.full((batch_size, self.beam_size, 1 + max_actions, self.hidden_size), 
                                    fill_value=0.0, device=self.device)

        gpt_beams = torch.full((batch_size, self.beam_size, 1 + max_actions, self.model.vocab_size), 
                                fill_value=0.0, device=self.device)

        key_beams = [torch.full((batch_size, self.beam_size, self.head_num, 1 + max_actions, self.head_dim), fill_value=0.0, device=self.device)
                     for _ in range(self.layer_num)]
        value_beams = [torch.full((batch_size, self.beam_size, self.head_num, 1 + max_actions, self.head_dim), fill_value=0.0, device=self.device)
                       for _ in range(self.layer_num)]
        # past_kv_beams = None
        
        # encoding_beams[:, 0, :, :] is reserved for emtpy
        per_batch_max_actions = 2 * seq_lens_np - 1
        
        current_step = 0
        batch_ids_per_step = []
        batch_ids_cpu = []
        
        for current_step in range(max_actions):
            not_terminated = current_step < per_batch_max_actions
            batch_ids = np.where(not_terminated)[0]
            assert len(batch_ids) > 0
            batch_ids_cpu.append(batch_ids)
            batch_ids_per_step.append(torch.tensor(batch_ids).to(self.device, non_blocking=True))
        
        input_embs = self.model.embeddings(input_ids)  # (N, L, dim)
        # stack_offsets = torch.ones(batch_size, 1, device=self.device)
        # queue_offsets = torch.zeros(batch_size, 1, device=self.device)  # (batch_size, current_beam_size)
        beam_scores = torch.zeros(batch_size, self.beam_size, 1 + max_actions, device=self.device).fill_(float('-inf'))  # (batch_size, current_beam_size)
        # (batch_size, 2, current_beam_size)

        # beam_action_scores = torch.zeros(batch_size, self.beam_size, 1 + max_actions, device=self.device).fill_(float('-inf'))

        states = []
        for batch_i in range(batch_size):
            init_state = State(None)
            init_state.beam_id = 0
            init_state.total_len = seq_lens_np[batch_i]
            beam_states = [init_state]
            states.append(beam_states)

        # shift twice for all states
        # assume input seq len >= 2
        # no expanding for beams
        # copy token embeddings to tensor_pool

        # initialize
        beam_scores[:, 0, 0] = 0
        gpt_output = self.model.gpt(inputs_embeds=self.model.bos_embedding.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1))
        past_kv_beams = gpt_output.past_key_values
        logits = self.model.classifier(self.model.dense(gpt_output.last_hidden_state))  # (N, 1, vocab_size)
        gpt_beams[:, 0, 0, :] = logits
        beam_scores[:, 0, 0] = 0

        current_cache_offset = 1
        input_ids_np = input_ids_np.data.numpy()
        
        for step in range(2):
            _indices = torch.ones((batch_size, 1, self.hidden_size), dtype=torch.long, device=self.device)
            next_beam_states = []
            for batch_i, beam_states in enumerate(states):
                new_states = []
                for state in beam_states:
                    next_state = state.act(ActionType.SHIFT, input_ids_np[batch_i][states[batch_i][0].input_offset])
                    assert next_state.current_step == current_cache_offset
                    next_state.beam_id = 0
                    new_states.append(next_state)
                next_beam_states.append(new_states)
            _indices = torch.full((batch_size, 1, self.gpt_input_size), fill_value=step, device=self.device)
            # _indices.fill_(step)
            src = input_embs.gather(1, _indices)

            logits = gpt_beams[:, 0, current_cache_offset - 1, :]  # (N, vocab_size)
            logits[:, self.reduce_id] = float('-inf')  # no reduce action
            tgt_ids = input_ids[:, step]  # (N)
            action_log_p = F.softmax(logits, dim=-1).gather(1, tgt_ids.unsqueeze(1)).squeeze(1).log()

            beam_scores[:, 0, current_cache_offset] = beam_scores[:, 0, current_cache_offset - 1] + action_log_p


            current_pos_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
            current_pos_ids.fill_(step + 1)
            gpt_output = self.model.gpt(inputs_embeds=src, position_ids=current_pos_ids, past_key_values=past_kv_beams)
            logits = self.model.classifier(self.model.dense(gpt_output.last_hidden_state))  # (N, 1, vocab_size)
            gpt_beams[:, 0, current_cache_offset, :] = logits
            past_kv_beams = gpt_output.past_key_values

            _indices = torch.full((batch_size, 1, self.hidden_size), fill_value=current_cache_offset, device=self.device)
            current_cache_offset += 1
            src_down_scale = self.model.down_scale(src)
            encoding_beams[:, 0, :, :].scatter_(1, _indices, src_down_scale)
            states = next_beam_states
        del _indices
        past_key_values = gpt_output.past_key_values
        past_kv_len = gpt_output.past_key_values[0][0].shape[2]
        for layer_i, (k, v) in enumerate(past_key_values):
            # k: (N, head_num, 3, head_dim)
            key_beams[layer_i][:, 0, :, :current_cache_offset, :] = k
            value_beams[layer_i][:, 0, :, :current_cache_offset, :] = v
        # print(f'log p: {current_cache_offset}: {F.softmax(logits, dim=-1)}')
        # beam_action_scores[:, 0, current_cache_offset - 1] = 0

        states_cache_offset = [0] * batch_size
        for batch_ids, batch_ids_cpu in zip(batch_ids_per_step[2:], batch_ids_cpu[2:]):
            N = batch_ids.shape[0]

            # (current_batch_size, beam_size)
            # (batch_size, beam_size)
            top_indices_batch = []
            score_mask_batch = []
            token_offsets_batch = []
            ext_vocab_ids_batch = []
            current_beam_size = len(states[batch_ids_cpu[0]])
            for batch_i in batch_ids_cpu:
                top_indices_left = []
                top_indices_right = []
                score_mask = []
                token_offsets = []
                ext_vocab_ids = []
                assert current_beam_size == len(states[batch_i])
                states_cache_offset[batch_i] = current_cache_offset
                for state in states[batch_i]:
                    l_top, r_top, span_ext_id = state.top_states(ext_vocab=self.ext_vocab)
                    if l_top is not None:
                        top_indices_left.append((batch_i, l_top.beam_id, l_top.current_step))
                    else:
                        top_indices_left.append((batch_i, 0, 0))
                    
                    if r_top is not None:
                        top_indices_right.append((batch_i, r_top.beam_id, r_top.current_step))
                    else:
                        top_indices_right.append((batch_i, 0, 0))
                    
                    if atom_spans is not None:
                        score_mask.append(state.action_masks(atom_spans[batch_i]))
                    else:
                        score_mask.append(state.action_masks())
                    token_offsets.append(state.token_offset)
                    if self.ext_vocab is not None:
                        ext_vocab_ids.append(span_ext_id) # +1, 0 is reserved for padding
                ext_vocab_ids_batch.append(ext_vocab_ids)
                top_indices_batch.append((top_indices_left, top_indices_right))
                score_mask_batch.append(score_mask)
                token_offsets_batch.append(token_offsets)
            top_indices_batch = torch.tensor(top_indices_batch).to(device=self.device, non_blocking=True)  # (N, 2, B, 3)
            if self.ext_vocab is not None:
                ext_vocab_ids_batch = torch.tensor(ext_vocab_ids_batch, device=self.device)  # (N, B)
            current_beam_scores = beam_scores[batch_ids, :, current_cache_offset - 1][:, :current_beam_size]  # (N, B)
            # current_action_scores = beam_action_scores[batch_ids, :, current_cache_offset - 1][:, :current_beam_size]

            top_indices_batch = top_indices_batch.permute(1, 0, 2, 3)  # (2, N, B, 3)
            top_indices_batch = top_indices_batch.reshape(2, N * current_beam_size, 3)
            top_left = encoding_beams[top_indices_batch[0, :, 0], top_indices_batch[0, :, 1], top_indices_batch[0, :, 2], :]
            top_left = top_left.view(N, current_beam_size, -1)
            top_right = encoding_beams[top_indices_batch[1, :, 0], top_indices_batch[1, :, 1], top_indices_batch[1, :, 2], :]  # (N * B, dim)
            top_right = top_right.view(N, current_beam_size, -1)

            top_tensors = torch.stack([top_left, top_right], dim=2)
            # (current_batch_size, beam_size, 2, dim)
            
            score_mask_batch = torch.tensor(score_mask_batch, dtype=torch.bool).to(device=self.device, non_blocking=True)  # (N, B, 2)
            # next token offset
            token_offsets_batch = torch.tensor(token_offsets_batch).to(device=self.device, non_blocking=True)  # (N, B)

            # TODO: replace with GPT
            last_logits = gpt_beams[:, :current_beam_size, current_cache_offset - 1]  # (N, B, vocab_size)
            # for shift only, set reduce logits to -inf
            last_logits[:, :, self.reduce_id].masked_fill_(~score_mask_batch[:, :, 1], float('-inf'))
            probs = F.softmax(last_logits, dim=-1)

            next_token_id = input_ids.gather(1, token_offsets_batch)  # (N, B)
            shift_scores = probs.gather(2, next_token_id.unsqueeze(-1)).log()  # probability of shifting and predicting the next token
            shift_scores = shift_scores.squeeze(-1)
            reduce_scores = probs[:, :, self.reduce_id].log()  # (N, B), probability of reducing

            # for reduce only , set probs to 1, corresponding scores set to 0
            # reduce_scores.masked_fill_(~score_mask_batch[:, :, 0], 0)
            action_scores = torch.stack([shift_scores, reduce_scores], dim=2)

            pure_shift_scores = (1 - reduce_scores.exp()).log()
            pure_action_scores = torch.stack([pure_shift_scores, reduce_scores], dim=2)
            
            # print(current_beam_scores)
            # print(action_scores)
            # print('--' * 10)
            current_beam_scores = current_beam_scores.unsqueeze(2) + action_scores  # (N, B, 2)
            # pure_action_scores = current_action_scores.unsqueeze(2) + pure_action_scores
            
            current_beam_scores.masked_fill_(~score_mask_batch, float('-inf'))
            # pure_action_scores.masked_fill_(~score_mask_batch, float('-inf'))

            # rank actions
            sorted_scores, top_beam_indices = current_beam_scores.view(N, 2 * current_beam_size).sort(dim=1, descending=True)
            top_beam_indices = top_beam_indices[:, :self.beam_size] # (N, B)
            sorted_scores = sorted_scores[:, :self.beam_size]
            beam_scores[batch_ids, :sorted_scores.shape[1], current_cache_offset] = sorted_scores
            # beam_action_scores[batch_ids, :sorted_scores.shape[1], current_cache_offset] = pure_action_scores.view(N, -1)[:, top_beam_indices]

            # print(beam_ppl)
            # (N, B)
            actions = top_beam_indices % 2
            beam_ids = top_beam_indices // 2
            assert torch.all(actions <= 1)
            actions_np = actions.to('cpu', non_blocking=True)  # (N, max_beam)
            beam_ids_np = beam_ids.to('cpu', non_blocking=True)  # (N, max_beam)
            if self.ext_vocab is None:
                _, reduce_repr = self.tree_enc(top_tensors)  # (N, B, dim)
            else:
                top_tensors = top_tensors.view(N * current_beam_size, 1, 2, self.hidden_size)  # (N * B, 1, 2, dim)
                # print(ext_vocab_ids_batch)
                span_embeddings = self.model.r2d2.ext_embeds(ext_vocab_ids_batch).view(-1, self.hidden_size)  # (N * B, dim)
                _, reduce_repr = self.tree_enc(top_tensors, span_embeddings)
                reduce_repr = reduce_repr.view(N, current_beam_size, self.hidden_size)

            token_offsets_batch_ = token_offsets_batch.unsqueeze(-1).repeat(1, 1, self.gpt_input_size)

            shift_repr = input_embs.gather(1, token_offsets_batch_)  # (N, B, dim)
            concat_repr = torch.stack([self.model.down_scale(shift_repr), reduce_repr], dim=2)  # (N, B, 2, dim)
            concat_repr = concat_repr.view(N, 2 * current_beam_size, -1)  # (N, 2 * B, dim)

            new_repr = concat_repr.gather(1, top_beam_indices.unsqueeze(-1).repeat(1, 1, self.hidden_size))  # (N, B, dim)
            
            encoding_beams[batch_ids, :new_repr.shape[1], current_cache_offset, :] = new_repr

            # adjust past_key_values
            past_kv_inputs = []
            for k, v in zip(key_beams, value_beams):
                k_ = k[batch_ids, beam_ids, :, :current_cache_offset, :]
                v_ = v[batch_ids, beam_ids, :, :current_cache_offset, :]
                past_kv_inputs.append((k_.view(-1, self.head_num, current_cache_offset, self.head_dim),
                                       v_.view(-1, self.head_num, current_cache_offset, self.head_dim)))
            # apply GPT
            new_beam_size = new_repr.shape[1]
            pos_ids = torch.stack([token_offsets_batch + 1, token_offsets_batch], dim=2)   # (N, B, 2)
            pos_ids = pos_ids.view(N, -1).gather(1, top_beam_indices)  # (N, B)

            upscale_reduce_repr = self.model.up_scale(reduce_repr)
            gpt_inputs = torch.stack([shift_repr, upscale_reduce_repr], dim=2)  # (N, B, 2, gpt_dim)
            gpt_inputs = gpt_inputs.view(N, 2 * current_beam_size, -1)

            gpt_inputs = gpt_inputs.gather(1, top_beam_indices.unsqueeze(-1).repeat(1, 1, self.gpt_input_size))  # (N, B, gpt_dim)

            assert past_kv_inputs[0][0].shape[2] == past_kv_len
            result = self.model.gpt(inputs_embeds=gpt_inputs.view(-1, 1, self.gpt_input_size), position_ids=pos_ids, 
                                    past_key_values=past_kv_inputs)
            past_kv_len = result.past_key_values[0][0].shape[2]
            past_kv_beams = self.split_past_kv(result.past_key_values, new_beam_size)
            for layer_i, (k, v) in enumerate(past_kv_beams):
                key_beams[layer_i][batch_ids, :new_beam_size, :, :current_cache_offset + 1, :] = k.view(-1, new_beam_size, k.shape[-3], k.shape[-2], k.shape[-1])
                value_beams[layer_i][batch_ids, :new_beam_size, :, :current_cache_offset + 1, :] = v.view(-1, new_beam_size, v.shape[-3], v.shape[-2], v.shape[-1])
            logits = self.model.classifier(self.model.dense(result.last_hidden_state)) # (N * B, vocab_size)
            gpt_beams[batch_ids, :new_beam_size, current_cache_offset, :] = logits.view(N, new_beam_size, -1)

            actions_np = actions_np.data.numpy()
            beam_ids_np = beam_ids_np.data.numpy()

            assert len(actions_np) == len(beam_ids_np)
            for idx, (action_ids, beam_ids) in enumerate(zip(actions_np, beam_ids_np)):
                batch_i = batch_ids_cpu[idx]
                new_states = []
                assert len(action_ids) == len(beam_ids)
                for new_beam_id, (action_id, beam_idx) in enumerate(zip(action_ids, beam_ids)):
                    if action_id == 0:
                        if states[batch_i][beam_idx].input_offset < input_ids_np.shape[1]:
                            next_state = states[batch_i][beam_idx].act(ActionType.SHIFT, token_id=input_ids_np[batch_i, states[batch_i][beam_idx].input_offset])
                        else:
                            next_state = states[batch_i][beam_idx].act(ActionType.SHIFT)
                    elif action_id == 1:
                        next_state = states[batch_i][beam_idx].act(ActionType.REDUCE)
                    else:
                        raise Exception(f'Unexisted action id: {action_id}, step: {step}, actions_np: {actions_np}, action_ids: {action_ids}')
                    next_state.beam_id = new_beam_id
                    assert next_state.current_step == current_cache_offset, \
                            f'new step: {next_state.current_step} vs {current_cache_offset}'
                    new_states.append(next_state)
                states[batch_i] = new_states
            
            current_cache_offset += 1
        # print(beam_action_scores[:, :, -1].exp().sum(dim=-1))
        # print(-beam_scores[:, :, -1].logsumexp(dim=-1) / seq_lens)
        # return -beam_scores[:, :, -1].logsumexp(dim=-1) / seq_lens
        # return (-beam_scores[:, :, -1].logsumexp(dim=-1) + beam_action_scores[:, :, -1].logsumexp(dim=-1))  / seq_lens
        
        # TODO: replace -1 with action lens to support multi-batch
        return -beam_scores[:, :, -1].logsumexp(dim=-1) / seq_lens, states

class R2D2GenFastEvaluator:
    def __init__(self, model, config, device, beam_size=1, ext_vocab: Dict[str, int]=None, structonly=False):
        self.model = model
        model.to(device)
        self.model.eval()
        self.hidden_size = model.r2d2_input_dim
        self.gpt_input_size = model.embedding_dim
        self.beam_size = beam_size
        # self.layer_num = config.n_layer
        self.action_layer_num = config.action_layer_num
        self.generation_layer_num = config.n_layer - config.action_layer_num
        self.head_num = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.reduce_id = model.r2d2.reduce_id
        self.ext_vocab = ext_vocab
        self.device = device
        self.structonly = structonly

    def tree_enc(self, src, span_embeddings=None):
        return self.model.r2d2.inside_enc(src, span_embeddings)

    def split_past_kv(self, past_key_values, beam_size):
        split_past_kvs = []
        for kv_layer in past_key_values:
            k, v = kv_layer
            assert k.shape[0] % beam_size == 0
            assert v.shape[0] % beam_size == 0

            k = k.view(-1, beam_size, k.shape[-3], k.shape[-2], k.shape[-1])
            v = v.view(-1, beam_size, k.shape[-3], k.shape[-2], k.shape[-1])

            split_past_kvs.append((k, v))
        return split_past_kvs

    def merge_past_kv(self, past_key_values):
        merge_past_kvs = []
        for kv_layer in past_key_values:
            k, v = kv_layer

            k = k.view(-1, k.shape[-3], k.shape[-2], k.shape[-1])
            v = v.view(-1, k.shape[-3], k.shape[-2], k.shape[-1])

            merge_past_kvs.append((k, v))
        return merge_past_kvs


    @torch.no_grad()
    def beam_search(self, input_ids, masks=None, atom_spans=None):
        # input_ids: (batch_size, max_len)
        # masks: (batch_size, max_len)
        assert input_ids.shape[0] == 1, "currently only support 1 batch size"
        if masks is None:
            masks = torch.ones_like(input_ids)
        # print(masks)
        seq_lens = masks.sum(dim=1)
        input_ids_np = input_ids.to('cpu', non_blocking=True)
        seq_lens_np = seq_lens.to(device='cpu').data.numpy()
        max_actions = max(seq_lens_np) * 2 - 1
        batch_size = input_ids.shape[0]
        # print(f'params: {batch_size}, {max_beam_size}, {max_actions}, {self.hidden_size}')
        encoding_beams = torch.full((batch_size, self.beam_size, 1 + max_actions, self.hidden_size), 
                                     fill_value=0.0, device=self.device)

        gpt_beams = torch.full((batch_size, self.beam_size, 1 + max_actions, self.model.vocab_size), 
                                fill_value=0.0, device=self.device)
        action_beams = torch.full((batch_size, self.beam_size, 1 + max_actions, 2), fill_value=0.0, device=self.device)

        action_key_beams = [torch.full((batch_size, self.beam_size, self.head_num, 1 + max_actions, self.head_dim), fill_value=0.0, device=self.device)
                            for _ in range(self.action_layer_num)]
        action_value_beams = [torch.full((batch_size, self.beam_size, self.head_num, 1 + max_actions, self.head_dim), fill_value=0.0, device=self.device)
                              for _ in range(self.action_layer_num)]

        token_key_beams = [torch.full((batch_size, self.beam_size, self.head_num, 1 + max_actions, self.head_dim), fill_value=0.0, device=self.device)
                           for _ in range(self.generation_layer_num)]
        token_value_beams = [torch.full((batch_size, self.beam_size, self.head_num, 1 + max_actions, self.head_dim), fill_value=0.0, device=self.device)
                             for _ in range(self.generation_layer_num)]
        token_mask = torch.full((batch_size, self.beam_size, 1 + max_actions), dtype=torch.bool, fill_value=1, device=self.device)
        # past_kv_beams = None

        def step(input, current_cache_offset, new_beam_size, position_ids=None, 
                 action_past_kv=None, generation_past_kv=None, attn_mask=None, batch_ids=None):
            N = input.shape[0] // new_beam_size
            action_results = self.model.action_layers(inputs_embeds=input,
                                                      position_ids=position_ids,
                                                      past_key_values=action_past_kv)
            token_results = self.model.generation_layers(inputs_embeds=action_results.last_hidden_state, past_key_values=generation_past_kv,
                                                         attention_mask=attn_mask)
            nonlocal action_key_beams
            nonlocal action_value_beams
            nonlocal token_key_beams
            nonlocal token_value_beams
            # Update kv beams, update attn_mask
            past_kv_len = action_results.past_key_values[0][0].shape[2]
            past_kv_beams = self.split_past_kv(action_results.past_key_values, new_beam_size)

            for layer_i, (k, v) in enumerate(past_kv_beams):
                _k = k.view(-1, new_beam_size, k.shape[-3], k.shape[-2], k.shape[-1])
                _v = v.view(-1, new_beam_size, v.shape[-3], v.shape[-2], v.shape[-1])
                if batch_ids is None:
                    action_key_beams[layer_i][:, :new_beam_size, :, :current_cache_offset + 1, :] = _k
                    action_value_beams[layer_i][:, :new_beam_size, :, :current_cache_offset + 1, :] = _v
                else:
                    action_key_beams[layer_i][batch_ids, :new_beam_size, :, :current_cache_offset + 1, :] = _k
                    action_value_beams[layer_i][batch_ids, :new_beam_size, :, :current_cache_offset + 1, :] = _v

            past_kv_len = token_results.past_key_values[0][0].shape[2]
            past_kv_beams = self.split_past_kv(token_results.past_key_values, new_beam_size)

            for layer_i, (k, v) in enumerate(past_kv_beams):
                _k = k.view(-1, new_beam_size, k.shape[-3], k.shape[-2], k.shape[-1])
                _v = v.view(-1, new_beam_size, v.shape[-3], v.shape[-2], v.shape[-1])
                if batch_ids is None:
                    token_key_beams[layer_i][:, :new_beam_size, :, :current_cache_offset + 1, :] = _k
                    token_value_beams[layer_i][:, :new_beam_size, :, :current_cache_offset + 1, :] = _v
                else:
                    token_key_beams[layer_i][batch_ids, :new_beam_size, :, :current_cache_offset + 1, :] = _k
                    token_value_beams[layer_i][batch_ids, :new_beam_size, :, :current_cache_offset + 1, :] = _v
            
            # Update token/action beams
            nonlocal gpt_beams
            nonlocal action_beams
            action_logits = self.model.action_mlp(action_results.last_hidden_state)  # (N * B, 2)
            token_logits = self.model.classifier(self.model.dense(token_results.last_hidden_state))

            if batch_ids is not None:
                gpt_beams[batch_ids, :new_beam_size, current_cache_offset, :] = token_logits.view(N, new_beam_size, -1)
                action_beams[batch_ids, :new_beam_size, current_cache_offset, :] = action_logits.view(N, new_beam_size, -1)
            else:
                gpt_beams[:, :new_beam_size, current_cache_offset, :] = token_logits.view(N, new_beam_size, -1)
                action_beams[:, :new_beam_size, current_cache_offset, :] = action_logits.view(N, new_beam_size, -1)
    
        def prepare_kv(batch_ids, beam_ids, current_cache_offset):
            past_action_kv = []
            # nonlocal action_key_beams
            # nonlocal action_value_beams
            for k, v in zip(action_key_beams, action_value_beams):
                if batch_ids is not None:
                    k_ = k[batch_ids, beam_ids, :, :current_cache_offset, :]
                    v_ = v[batch_ids, beam_ids, :, :current_cache_offset, :]
                else:
                    k_ = k[:, beam_ids, :, :current_cache_offset, :]
                    v_ = v[:, beam_ids, :, :current_cache_offset, :]
                past_action_kv.append((k_.view(-1, self.head_num, current_cache_offset, self.head_dim),
                                       v_.view(-1, self.head_num, current_cache_offset, self.head_dim)))

            past_token_kv = []
            # nonlocal token_key_beams
            # nonlocal token_value_beams
            for k, v in zip(token_key_beams, token_value_beams):
                if batch_ids is not None:
                    k_ = k[batch_ids, beam_ids, :, :current_cache_offset, :]
                    v_ = v[batch_ids, beam_ids, :, :current_cache_offset, :]
                else:
                    k_ = k[:, beam_ids, :, :current_cache_offset, :]
                    v_ = v[:, beam_ids, :, :current_cache_offset, :]
                past_token_kv.append((k_.view(-1, self.head_num, current_cache_offset, self.head_dim),
                                      v_.view(-1, self.head_num, current_cache_offset, self.head_dim)))
            return past_action_kv, past_token_kv
        
        # encoding_beams[:, 0, :, :] is reserved for emtpy
        per_batch_max_actions = 2 * seq_lens_np - 1
        
        current_step = 0
        batch_ids_per_step = []
        batch_ids_cpu = []
        
        for current_step in range(max_actions):
            not_terminated = current_step < per_batch_max_actions
            batch_ids = np.where(not_terminated)[0]
            assert len(batch_ids) > 0
            batch_ids_cpu.append(batch_ids)
            batch_ids_per_step.append(torch.tensor(batch_ids).to(self.device, non_blocking=True))
        
        input_embs = self.model.embeddings(input_ids)  # (N, L, dim)
        # stack_offsets = torch.ones(batch_size, 1, device=self.device)
        # queue_offsets = torch.zeros(batch_size, 1, device=self.device)  # (batch_size, current_beam_size)
        beam_scores = torch.zeros(batch_size, self.beam_size, 1 + max_actions, device=self.device).fill_(float('-inf'))  # (batch_size, current_beam_size)
        # (batch_size, 2, current_beam_size)

        # beam_action_scores = torch.zeros(batch_size, self.beam_size, 1 + max_actions, device=self.device).fill_(float('-inf'))

        states = []
        for batch_i in range(batch_size):
            init_state = State(None)
            init_state.beam_id = 0
            init_state.total_len = seq_lens_np[batch_i]
            beam_states = [init_state]
            states.append(beam_states)

        # shift twice for all states
        # assume input seq len >= 2
        # no expanding for beams
        # copy token embeddings to tensor_pool

        # initialize
        beam_scores[:, 0, 0] = 0
        # gpt_output = self.model.action_layer(inputs_embeds=self.model.bos_embedding.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1))
        # past_kv_beams = gpt_output.past_key_values
        # logits = self.model.classifier(self.model.dense(gpt_output.last_hidden_state))  # (N, 1, vocab_size)
        # gpt_beams[:, 0, 0, :] = logits
        current_pos_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        step(self.model.bos_embedding.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1), 0, 1, position_ids=current_pos_ids)

        current_cache_offset = 1
        input_ids_np = input_ids_np.data.numpy()
        
        for step_i in range(2):
            _indices = torch.ones((batch_size, 1, self.hidden_size), dtype=torch.long, device=self.device)
            next_beam_states = []
            for batch_i, beam_states in enumerate(states):
                new_states = []
                for state in beam_states:
                    next_state = state.act(ActionType.SHIFT, token_id=input_ids_np[batch_i][states[batch_i][0].input_offset])
                    assert next_state.current_step == current_cache_offset
                    next_state.beam_id = 0
                    new_states.append(next_state)
                next_beam_states.append(new_states)
            _indices = torch.full((batch_size, 1, self.gpt_input_size), fill_value=step_i, device=self.device)
            # _indices.fill_(step)
            src = input_embs.gather(1, _indices)

            action_logits = action_beams[:, 0, current_cache_offset - 1, :]  # (N, 2)
            token_logits = gpt_beams[:, 0, current_cache_offset - 1, :]  # (N, vocab_size)
            action_logits[:, ActionType.REDUCE.value] = float('-inf')  # no reduce action
            tgt_ids = input_ids[:, step_i]  # (N)
            log_p = F.log_softmax(action_logits, dim=-1)[:, ActionType.SHIFT.value] + F.log_softmax(token_logits).gather(1, tgt_ids.unsqueeze(1))
            beam_scores[:, 0, current_cache_offset] = beam_scores[:, 0, current_cache_offset - 1] + log_p
            # token_mask[:, 0, current_cache_offset - 1] = True  # set the previous token mask to True for a shift action, and False for a reduce action

            current_pos_ids.fill_(step_i + 1)
            # gpt_output = self.model.gpt(inputs_embeds=src, position_ids=current_pos_ids, past_key_values=past_kv_beams)
            # logits = self.model.classifier(self.model.dense(gpt_output.last_hidden_state))  # (N, 1, vocab_size)
            # gpt_beams[:, 0, current_cache_offset, :] = logits
            action_past_kv, token_past_kv = prepare_kv(None, [0], current_cache_offset)
            step(src, current_cache_offset, 1, position_ids=current_pos_ids, action_past_kv=action_past_kv, generation_past_kv=token_past_kv)
            # past_kv_beams = gpt_output.past_key_values

            _indices = torch.full((batch_size, 1, self.hidden_size), fill_value=current_cache_offset, device=self.device)
            current_cache_offset += 1
            src_down_scale = self.model.down_scale(src)
            encoding_beams[:, 0, :, :].scatter_(1, _indices, src_down_scale)
            states = next_beam_states
        del _indices
        
        # print(f'log p: {current_cache_offset}: {F.softmax(logits, dim=-1)}')
        # beam_action_scores[:, 0, current_cache_offset - 1] = 0

        states_cache_offset = [0] * batch_size
        for batch_ids, batch_ids_cpu in zip(batch_ids_per_step[2:], batch_ids_cpu[2:]):
            N = batch_ids.shape[0]

            # (current_batch_size, beam_size)
            # (batch_size, beam_size)
            top_indices_batch = []
            score_mask_batch = []
            token_offsets_batch = []
            ext_vocab_ids_batch = []
            current_beam_size = len(states[batch_ids_cpu[0]])
            for batch_i in batch_ids_cpu:
                top_indices_left = []
                top_indices_right = []
                score_mask = []
                token_offsets = []
                ext_vocab_ids = []
                assert current_beam_size == len(states[batch_i])
                states_cache_offset[batch_i] = current_cache_offset
                for state in states[batch_i]:
                    l_top, r_top, ext_vocab_id = state.top_states(ext_vocab=self.ext_vocab)
                    
                    if l_top is not None:
                        top_indices_left.append((batch_i, l_top.beam_id, l_top.current_step))
                    else:
                        top_indices_left.append((batch_i, 0, 0))
                    
                    if r_top is not None:
                        top_indices_right.append((batch_i, r_top.beam_id, r_top.current_step))
                    else:
                        top_indices_right.append((batch_i, 0, 0))
                    
                    if atom_spans is not None:
                        score_mask.append(state.action_masks(atom_spans[batch_i]))
                    else:
                        score_mask.append(state.action_masks())
                    token_offsets.append(min(state.token_offset, state.total_len - 1))
                    ext_vocab_ids.append(ext_vocab_id)
                ext_vocab_ids_batch.append(ext_vocab_ids)
                top_indices_batch.append((top_indices_left, top_indices_right))
                score_mask_batch.append(score_mask)
                token_offsets_batch.append(token_offsets)
            top_indices_batch = torch.tensor(top_indices_batch).to(device=self.device, non_blocking=True)  # (N, 2, B, 3)
            current_beam_scores = beam_scores[batch_ids, :, current_cache_offset - 1][:, :current_beam_size]  # (N, B)
            # current_action_scores = beam_action_scores[batch_ids, :, current_cache_offset - 1][:, :current_beam_size]
            if self.ext_vocab is not None:
                ext_vocab_ids_batch = torch.tensor(ext_vocab_ids_batch, device=self.device)
            top_indices_batch = top_indices_batch.permute(1, 0, 2, 3)  # (2, N, B, 3)
            # 3: batch_id, beam_id, position_id
            top_indices_batch = top_indices_batch.reshape(2, N * current_beam_size, 3)
            top_left = encoding_beams[top_indices_batch[0, :, 0], top_indices_batch[0, :, 1], top_indices_batch[0, :, 2], :]
            top_left = top_left.view(N, current_beam_size, -1)
            top_right = encoding_beams[top_indices_batch[1, :, 0], top_indices_batch[1, :, 1], top_indices_batch[1, :, 2], :]  # (N * B, dim)
            top_right = top_right.view(N, current_beam_size, -1)

            top_tensors = torch.stack([top_left, top_right], dim=2)
            # (current_batch_size, beam_size, 2, dim)
            
            score_mask_batch = torch.tensor(score_mask_batch, dtype=torch.bool).to(device=self.device, non_blocking=True)  # (N, B, 2)
            # next token offset
            token_offsets_batch = torch.tensor(token_offsets_batch).to(device=self.device, non_blocking=True)  # (N, B)

            token_logits = gpt_beams[:, :current_beam_size, current_cache_offset - 1]  # (N, B, vocab_size)
            action_logits = action_beams[:, :current_beam_size, current_cache_offset - 1]  # (N, B, vocab_size)
            # for shift only, set reduce logits to -inf
            action_logits[:, :, ActionType.REDUCE.value].masked_fill_(~score_mask_batch[:, :, 1], float('-inf'))
            # probs = F.softmax(last_logits, dim=-1)
            action_probs = F.log_softmax(action_logits, dim=-1) # (N, B, 2)
            token_probs = F.log_softmax(token_logits, dim=-1)  # (N, B, vocab_size)

            next_token_id = input_ids.gather(1, token_offsets_batch)  # (N, B)
            gen_scores = action_probs[:, :, ActionType.SHIFT.value] + token_probs.gather(2, next_token_id.unsqueeze(-1)).squeeze(-1) # probability of shifting and predicting the next token
            # gen_scores: (N, B)
            compose_scores = action_probs[:, :, ActionType.REDUCE.value]  # (N, B), probability of reducing

            action_scores = torch.stack([gen_scores, compose_scores], dim=2)

            # pure_shift_scores = (1 - reduce_scores.exp()).log()
            # pure_action_scores = torch.stack([pure_shift_scores, reduce_scores], dim=2)
            
            # print(current_beam_scores)
            # print(action_scores)
            # print('--' * 10)
            current_beam_scores = current_beam_scores.unsqueeze(2) + action_scores  # (N, B, 2)
            # pure_action_scores = current_action_scores.unsqueeze(2) + pure_action_scores
            
            current_beam_scores.masked_fill_(~score_mask_batch, float('-inf'))
            # pure_action_scores.masked_fill_(~score_mask_batch, float('-inf'))

            # rank actions
            sorted_scores, top_beam_indices = current_beam_scores.view(N, 2 * current_beam_size).sort(dim=1, descending=True)
            top_beam_indices = top_beam_indices[:, :self.beam_size] # (N, B)
            sorted_scores = sorted_scores[:, :self.beam_size]
            beam_scores[batch_ids, :sorted_scores.shape[1], current_cache_offset] = sorted_scores
            # beam_action_scores[batch_ids, :sorted_scores.shape[1], current_cache_offset] = pure_action_scores.view(N, -1)[:, top_beam_indices]

            # print(beam_ppl)
            # (N, B)
            actions = top_beam_indices % 2
            beam_ids = top_beam_indices // 2
            assert torch.all(actions <= 1)
            actions_np = actions.to('cpu', non_blocking=True)  # (N, max_beam)
            beam_ids_np = beam_ids.to('cpu', non_blocking=True)  # (N, max_beam)
            if self.ext_vocab is None:
                _, reduce_repr = self.tree_enc(top_tensors)  # (N, B, dim)
            else:
                top_tensors = top_tensors.view(N * current_beam_size, 1, 2, self.hidden_size)  # (N * B, 1, 2, dim)
                # print(ext_vocab_ids_batch)
                span_embeddings = self.model.r2d2.ext_embeds(ext_vocab_ids_batch).view(-1, self.hidden_size)  # (N * B, dim)
                _, reduce_repr = self.tree_enc(top_tensors, span_embeddings)
                reduce_repr = reduce_repr.view(N, current_beam_size, self.hidden_size)

            token_offsets_batch_ = token_offsets_batch.unsqueeze(-1).repeat(1, 1, self.gpt_input_size)

            shift_repr = input_embs.gather(1, token_offsets_batch_)  # (N, B, dim)
            concat_repr = torch.stack([self.model.down_scale(shift_repr), reduce_repr], dim=2)  # (N, B, 2, dim)
            concat_repr = concat_repr.view(N, 2 * current_beam_size, -1)  # (N, 2 * B, dim)

            new_repr = concat_repr.gather(1, top_beam_indices.unsqueeze(-1).repeat(1, 1, self.hidden_size))  # (N, B, dim)
            
            encoding_beams[batch_ids, :new_repr.shape[1], current_cache_offset, :] = new_repr

            # adjust past_key_values
            # past_kv_inputs = []
            # for k, v in zip(key_beams, value_beams):
            #     k_ = k[batch_ids, beam_ids, :, :current_cache_offset, :]
            #     v_ = v[batch_ids, beam_ids, :, :current_cache_offset, :]
            #     past_kv_inputs.append((k_.view(-1, self.head_num, current_cache_offset, self.head_dim),
            #                            v_.view(-1, self.head_num, current_cache_offset, self.head_dim)))
            # apply GPT
            new_beam_size = new_repr.shape[1]
            pos_ids = torch.stack([token_offsets_batch + 1, token_offsets_batch], dim=2)   # (N, B, 2)
            pos_ids = pos_ids.view(N, -1).gather(1, top_beam_indices)  # (N, B)

            if not self.structonly:
                upscale_reduce_repr = self.model.up_scale(reduce_repr)
            else:
                reduce_ids = torch.ones(*reduce_repr.shape[:-1], dtype=torch.long, device=self.device).fill_(self.model.r2d2.reduce_id)
                upscale_reduce_repr = self.model.embeddings(reduce_ids)
            gpt_inputs = torch.stack([shift_repr, upscale_reduce_repr], dim=2)  # (N, B, 2, gpt_dim)
            gpt_inputs = gpt_inputs.view(N, 2 * current_beam_size, -1)
            gpt_inputs = gpt_inputs.gather(1, top_beam_indices.unsqueeze(-1).repeat(1, 1, self.gpt_input_size))  # (N, B, gpt_dim)
            # print(f'next gpt inputs shape: {gpt_inputs.shape}')

            gpt_mask = token_mask[batch_ids, :current_beam_size, :current_cache_offset + 1].unsqueeze(2).repeat(1, 1, 2, 1)
            assert torch.all(gpt_mask[:,:, :, current_cache_offset - 1])
            # gpt_mask[:,:, ActionType.SHIFT.value, current_cache_offset - 1] = True # corresponding to shift
            gpt_mask[:,:, ActionType.REDUCE.value, current_cache_offset - 1] = False # corresponding to shift
            gpt_mask = gpt_mask.view(N, 2 * current_beam_size, current_cache_offset + 1)  # (N, 2 * B, current_cache_offset)
            gpt_mask = gpt_mask.gather(1, top_beam_indices.unsqueeze(-1).repeat(1, 1, current_cache_offset + 1))
            # token_offsets_batch: [N, B]
            # print(f'gpt_mask_sum: {gpt_mask.sum(dim=-1)}, token offset batch: {pos_ids}')
            # assert gpt_mask.sum(dim=-1) == token_offsets_batch.gather(1, top_beam_indices)
            # print(f'gpt mask: {gpt_mask.shape}, current cache_offset : {current_cache_offset}')
            action_past_kv, generation_past_kv = prepare_kv(batch_ids, beam_ids, current_cache_offset)
            # print(f'action past k: {action_past_kv[0][0].shape}')
            step(gpt_inputs.view(-1, 1, self.gpt_input_size), current_cache_offset, new_beam_size, position_ids=pos_ids,
                 action_past_kv=action_past_kv, generation_past_kv=generation_past_kv,
                 attn_mask=gpt_mask, batch_ids=batch_ids)
            # update token_mask
            token_mask[batch_ids, :new_beam_size, :current_cache_offset + 1] = gpt_mask

            actions_np = actions_np.data.numpy()
            beam_ids_np = beam_ids_np.data.numpy()

            assert len(actions_np) == len(beam_ids_np)
            for idx, (action_ids, beam_ids) in enumerate(zip(actions_np, beam_ids_np)):
                batch_i = batch_ids_cpu[idx]
                new_states = []
                assert len(action_ids) == len(beam_ids)
                for new_beam_id, (action_id, beam_idx) in enumerate(zip(action_ids, beam_ids)):
                    if action_id == 0:
                        if states[batch_i][beam_idx].input_offset < input_ids_np.shape[1]:
                            next_state = states[batch_i][beam_idx].act(ActionType.SHIFT, token_id=input_ids_np[batch_i, states[batch_i][beam_idx].input_offset])
                        else:
                            next_state = states[batch_i][beam_idx].act(ActionType.SHIFT)
                    elif action_id == 1:
                        next_state = states[batch_i][beam_idx].act(ActionType.REDUCE)
                    else:
                        raise Exception(f'Unexisted action id: {action_id}, step: {step}, actions_np: {actions_np}, action_ids: {action_ids}')
                    next_state.beam_id = new_beam_id
                    assert next_state.current_step == current_cache_offset, \
                            f'new step: {next_state.current_step} vs {current_cache_offset}'
                    new_states.append(next_state)
                states[batch_i] = new_states
            
            current_cache_offset += 1
        # print(beam_action_scores[:, :, -1].exp().sum(dim=-1))
        # print(-beam_scores[:, :, -1].logsumexp(dim=-1) / seq_lens)
        # return -beam_scores[:, :, -1].logsumexp(dim=-1) / seq_lens
        # return (-beam_scores[:, :, -1].logsumexp(dim=-1) + beam_action_scores[:, :, -1].logsumexp(dim=-1))  / seq_lens
        
        # TODO: replace -1 with action lens to support multi-batch
        return -beam_scores[:, :, -1].logsumexp(dim=-1) / seq_lens, states