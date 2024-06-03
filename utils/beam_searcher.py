# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
from model.stack_state import ActionType, Node, State, BeamContext
import numpy as np
import torch
import torch.nn.functional as F


class R2D2GenFastBeamSearcher:
    def __init__(self, model, config, device, beam_size=10, sampling=False):
        self.model = model
        model.to(device)
        self.model.eval()
        self.hidden_size = model.r2d2_input_dim
        self.gpt_input_size = model.embedding_dim
        self.gpt_config = config
        self.beam_size = beam_size
        # self.layer_num = config.n_layer
        self.eos_id = config.eos_token_id
        self.action_layer_num = config.action_layer_num
        self.generation_layer_num = config.n_layer - config.action_layer_num
        self.head_num = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.reduce_id = model.r2d2.reduce_id
        self.sampling = sampling
        self.device = device

    def tree_enc(self, src):
        return self.model.r2d2.inside_enc(src)

    # def step(self, input, position_ids, beam_context, active_states):
    def step(self, states_batch, beam_context, sync_steps):
        # RETURN action_logits, token_logits, action_kv, token_kv
        gpt_input, position_ids, action_kv, token_kv, action_masks, token_masks = beam_context.prepare_gpt_input(states_batch, sync_steps)
        if gpt_input is not None:
            assert torch.all(token_masks.sum(dim=1) == position_ids + 1), f'{token_masks.sum(dim=1)} / {position_ids + 1}'
            # print(action_kv[0][0][0, 0, :5, :5])
            # print(action_masks)
            # for k, v in action_kv:
            #     # k (?, n_head, L, n_dim)
            #     assert (k * (1.0 - action_masks[:, :-1].float()).unsqueeze(1).unsqueeze(-1)).sum() == 0
            #     assert (v * (1.0 - action_masks[:, :-1].float()).unsqueeze(1).unsqueeze(-1)).sum() == 0
            #     assert torch.all((k.sum(dim=-1).sum(dim=1) != 0) == action_masks[:, :-1]), f'{k.sum(dim=-1).sum(dim=1) != 0} vs {action_masks[:, :-1]}'
            #     assert torch.all((v.sum(dim=-1).sum(dim=1) != 0) == action_masks[:, :-1]), f'{v.sum(dim=-1).sum(dim=1) != 0} vs {action_masks[:, :-1]}'

            # for k, v in token_kv:
            #     # k (?, n_head, L, n_dim)
            #     assert (k * (1.0 - token_masks[:, :-1].float()).unsqueeze(1).unsqueeze(-1)).sum() == 0
            #     assert (v * (1.0 - token_masks[:, :-1].float()).unsqueeze(1).unsqueeze(-1)).sum() == 0
            #     assert torch.all((k.sum(dim=-1).sum(dim=1) != 0) == token_masks[:, :-1]), f'{k.sum(dim=-1).sum(dim=1) != 0} vs {token_masks[:, :-1]}'
            #     assert torch.all((v.sum(dim=-1).sum(dim=1) != 0) == token_masks[:, :-1]), f'{v.sum(dim=-1).sum(dim=1) != 0} vs {token_masks[:, :-1]}'
            action_result = self.model.action_layers(inputs_embeds=gpt_input.unsqueeze(1), position_ids=position_ids, 
                                                     past_key_values=action_kv, attention_mask=action_masks)
            token_result = self.model.generation_layers(inputs_embeds=action_result.last_hidden_state, 
                                                        past_key_values=token_kv, 
                                                        attention_mask=token_masks)
            action_logits = self.model.action_mlp(action_result.last_hidden_state)  # (N * B, 2)
            token_logits = self.model.classifier(self.model.dense(token_result.last_hidden_state))

            action_kv_last_slice = []
            token_kv_last_slice = []
            for k, v in action_result.past_key_values:
                action_kv_last_slice.append((k[:, :, -1:, :], v[:, :, -1:, :]))
            for k, v in token_result.past_key_values:
                token_kv_last_slice.append((k[:, :, -1:, :], v[:, :, -1:, :]))
            return action_logits, token_logits, action_kv_last_slice, token_kv_last_slice
        else:
            return None, None, None, None

    def bos_step(self, N):
        # gpt_input = self.model.layer_norm(self.model.bos_embedding.unsqueeze(0).unsqueeze(1).repeat(N, 1, 1))
        gpt_input = self.model.bos_embedding.unsqueeze(0).unsqueeze(1).repeat(N, 1, 1)
        position_ids = torch.zeros((N, 1), dtype=torch.long, device=self.device)
        action_result = self.model.action_layers(inputs_embeds=gpt_input, position_ids=position_ids)
        token_result = self.model.generation_layers(inputs_embeds=action_result.last_hidden_state)
        action_logits = self.model.action_mlp(action_result.last_hidden_state)  # (N * B, 2)
        token_logits = self.model.classifier(self.model.dense(token_result.last_hidden_state))
        return action_logits, token_logits, action_result.past_key_values, token_result.past_key_values
        
    def next_actions(self, action_logits, token_logits, active_states, sync_steps, input_ids=None, atom_spans=None):
        # action_logits: (?, 1, 2), token_logits(?, 1, vocab_size)
        action_logits = action_logits.squeeze(1)
        token_logits = token_logits.squeeze(1)
        flatten_states = []
        # padding action_logits & token_logits
        max_beam_size = 0
        score_masks = []
        gather_indices = []
        token_ids = []
        state_idx = 0
        base_scores = []

        for batch_i, states in enumerate(active_states):
            for state_i, state in enumerate(states):
                state.batch_idx = -1  # set pending state batch idx to -1
                if state.token_offset == sync_steps[batch_i] and not state.is_finished:
                    state.batch_idx = len(flatten_states)
                    base_scores.append(state.score)
                    flatten_states.append(state)
                    if atom_spans is not None:
                        score_masks.append(state.action_masks(atom_spans[batch_i]))
                    else:
                        score_masks.append(state.action_masks())
                    gather_indices.append((batch_i, state_i))
                    if input_ids is not None:
                        if state.token_offset < state.total_len:
                            token_ids.append((state_idx, input_ids[batch_i][state.token_offset]))
                        else:
                            token_ids.append((state_idx, 0))
                    state_idx += 1
            max_beam_size = max(max_beam_size, len(states))

        assert len(flatten_states) == action_logits.shape[0], f'{len(flatten_states)} vs {action_logits.shape[0]}'
        action_space = token_logits.shape[-1] + 1
        base_scores = torch.tensor(base_scores, device=self.device)
        padded_scores = torch.full((len(active_states), max_beam_size, action_space), fill_value=float('-inf'), device=self.device)
        gather_indices = torch.tensor(gather_indices, device=self.device)
        score_masks = torch.tensor(score_masks, device=self.device)

        # action_logits.masked_fill_(~score_mask_batch, float('-inf'))
        # probs = F.softmax(last_logits, dim=-1)
        action_scores = F.log_softmax(action_logits, dim=-1) # (1, B, 2)
        action_scores.masked_fill_(~score_masks, float('-inf'))  # set log_p of invalid actions to -inf

        token_scores = F.log_softmax(token_logits, dim=-1)  # (?, vocab_size)
        vocab_size = token_scores.shape[-1]
        if input_ids is not None:
            masked_scores = torch.zeros_like(token_scores).fill_(float('-inf'))
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            masked_scores[token_ids[:, 0], token_ids[:, 1]] = token_scores[token_ids[:, 0], token_ids[:, 1]]
            token_scores = masked_scores
        gen_scores = action_scores[:, ActionType.SHIFT.value].unsqueeze(-1) + token_scores  
        # gen_scores: (?, vocab_size)
        compose_scores = action_scores[:, ActionType.REDUCE.value]  # (?), probability of reducing
        action_scores = torch.cat([gen_scores, compose_scores.unsqueeze(-1)], dim=1)  # (?, vocab_size + 1)
        action_scores = action_scores + base_scores.unsqueeze(1)
        assert action_scores.shape[-1] == vocab_size + 1
        # print(f'gather_indices: {gather_indices}, padded_scores shape: {padded_scores.shape}')
        padded_scores[gather_indices[:, 0], gather_indices[:, 1], :] = action_scores
        # sort
        sorted_val, indices = torch.sort(padded_scores.view(len(active_states), -1), dim=1, descending=True, stable=True)
        top_scores = sorted_val[:, :self.beam_size].cpu().data.numpy()  # (batch_sz, beam_size)
        top_indices = indices[:, :self.beam_size].cpu().data.numpy()  # (batch_sz, beam_size)
        new_active_states = []

        for batch_i, indices in enumerate(top_indices):
            new_states = []
            for score_idx, sorted_idx in enumerate(indices):
                beam_id = sorted_idx // (vocab_size + 1)
                action_id = sorted_idx % (vocab_size + 1)
                if top_scores[batch_i, score_idx] > float('-inf'):
                    # ensure not invalid action
                    assert active_states[batch_i][beam_id].batch_idx != -1
                    if action_id == vocab_size:
                        state = active_states[batch_i][beam_id].act(ActionType.REDUCE)
                    else:
                        state = active_states[batch_i][beam_id].act(ActionType.SHIFT, token_id=action_id)
                    state.beam_id = active_states[batch_i][beam_id].beam_id
                    # assert top_scores[batch_i][score_idx] < 0
                    assert top_scores[batch_i][score_idx] <= active_states[batch_i][beam_id].score
                    state.score = top_scores[batch_i][score_idx]
                    state.batch_idx = active_states[batch_i][beam_id].batch_idx
                    new_states.append(state)
            assert len(active_states) > 0
            for state in active_states[batch_i]:
                if state.token_offset == sync_steps[batch_i] + 1 or state.is_finished:
                    # already step into the next token
                    new_states.append(state)
            new_active_states.append(new_states)

        return new_active_states

    def compose(self, states, beam_context):
        top_stack_reprs = beam_context.prepare_compositions(states, beam_context)  # (?, 2, dim)
        
        if top_stack_reprs is not None:
            # print(top_stack_reprs[:, :, :5])
            assert not torch.any(top_stack_reprs.abs().sum(dim=-1) == 0)
            _, reduce_reprs = self.tree_enc(top_stack_reprs)
            return reduce_reprs
        else:
            return None

    def prepare_position_ids(self, states_batch):
        position_ids = []
        for states in states_batch:
            for state in states:
                position_ids.append(state.token_offset)
        return torch.tensor(position_ids, dtype=torch.long, device=self.device)

    def all_finished(self, states_batch, input_ids):
        for states in states_batch:
            for state in states:
                if not state.is_finished:
                    return False
        return True

    def update_sync_steps(self, states_batch, sync_steps, marginal_logp=None):
        for batch_i, states in enumerate(states_batch):
            all_pass = True
            for state in states:
                if state.token_offset == sync_steps[batch_i]:
                    all_pass = False
            if all_pass:
                
                if marginal_logp != None:
                    log_p = np.array([state.score for state in states])
                    log_p_torch = torch.from_numpy(log_p).to(device=self.device)
                    marginal_logp[batch_i][sync_steps[batch_i]] = log_p_torch.logsumexp(dim=-1)
                    # print("token_offset: ", [state.token_offset for state in states])
                    # print("sync_steps: ", sync_steps)
                    # print("marginal_logp: ", marginal_logp)

                if self.sampling:
                    # randomly select one
                    # log_p_mean = np.array([state.score / (sync_steps[batch_i] + 1) for state in states])
                    log_p = np.array([state.score for state in states])
                    p = np.exp(log_p)
                    p /= p.sum()
                    selected_idx = np.random.choice(a=len(states), size=1, p=p)
                    states[selected_idx[0]].score = 0
                    states_batch[batch_i] = [states[selected_idx[0]]]
                sync_steps[batch_i] += 1
        return sync_steps, marginal_logp

    @torch.no_grad()
    def beam_search(self, 
                    chunk_input_ids=None, 
                    chunk_masks=None, 
                    input_ids=None, 
                    masks=None, 
                    group_ids=None, 
                    past_kv=None, 
                    max_steps=100, 
                    target_ids=None, 
                    target_masks=None, 
                    atom_spans=None,
                    tag=None):
        # input_ids: (N, seq_len)
        assert chunk_input_ids is not None or target_ids is not None
        assert chunk_input_ids is None or target_ids is None

        if target_ids is not None:
            N = target_ids.shape[0]
            seq_lens = target_masks.sum(dim=1)
            seq_lens_np = seq_lens.cpu().data.numpy()
            max_steps = max(seq_lens_np)
        else:
            N = chunk_input_ids.shape[0]
            seq_lens_np = [max_steps] * N

        marginal_logp = None
        if tag != None:
            marginal_logp = torch.zeros(N, max_steps, device=self.device).fill_(float('-inf')) # (N, L)

        states_batch = []
        if chunk_input_ids is not None:
            outputs = self.model(chunk_input_ids=chunk_input_ids, 
                                 chunk_masks=chunk_masks, 
                                 input_ids=input_ids,
                                 group_ids=group_ids,
                                 masks=masks,
                                 span_ids=None,
                                 external_vocab_ids=None,
                                 past_key_values=past_kv)
            
            past_action_kvs, past_token_kvs = outputs.past_kv
            action_kv_len = past_action_kvs[0][0].shape[2]
            token_kv_len = past_token_kvs[0][0].shape[2]
            # TODO: set history kv, update beam context
            beam_context = BeamContext(self.model, N, self.beam_size, max_steps, 
                                       self.hidden_size, self.gpt_config, 
                                       self.device, action_kv_history_len=action_kv_len, 
                                       token_kv_history_len=token_kv_len)
            seq_lens = masks.sum(dim=-1).cpu().data.numpy()
            action_len_bucket, token_len_bucket = beam_context.init_history_kv(past_action_kvs, past_token_kvs, seq_lens, group_ids)
            action_len_bucket_t = torch.tensor(action_len_bucket, device=self.device)
            token_len_bucket_t = torch.tensor(token_len_bucket, device=self.device)
            action_logits = outputs.action_logits.gather(dim=1, index=(action_len_bucket_t - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, 2))
            token_logits = outputs.logits.gather(dim=1, index=(token_len_bucket_t - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, outputs.logits.shape[-1]))
            last_action_kv = last_token_kv = None
        else:
            beam_context = BeamContext(self.model, N, self.beam_size, max_steps, 
                                       self.hidden_size, self.gpt_config, self.device)
            action_logits, token_logits, last_action_kv, last_token_kv = self.bos_step(N)

        for batch_i in range(N):
            init_state = State(None)
            init_state.beam_id = 0
            init_state.total_len = seq_lens_np[batch_i]
            states_batch.append([init_state])


        sync_steps = [0] * N
        while True:
            candidate_states = self.next_actions(action_logits, token_logits, states_batch, sync_steps, target_ids, atom_spans)
            # update beam context
            for batch_i, states in enumerate(candidate_states):
                states.sort(key=lambda x: x.score, reverse=True)
                top_k_states = states[:self.beam_size]
                states_batch[batch_i] = top_k_states
            # compose representation
            reduce_repr = self.compose(states_batch, beam_context)

            # update context
            beam_context.update(states_batch, sync_steps, reduce_repr, last_action_kv, last_token_kv)

            # for those beams all synchronized on token, pass next token to gpt
            if tag != None:
                sync_steps, marginal_logp = self.update_sync_steps(states_batch, sync_steps, marginal_logp)
            else:
                sync_steps, _ = self.update_sync_steps(states_batch, sync_steps)                        
            action_logits, token_logits, last_action_kv, last_token_kv = self.step(states_batch, beam_context, sync_steps)
            if action_logits is None:
                break
        for states in states_batch:
            states.sort(key=lambda x: x.score, reverse=True)
        if tag != None:
            # print("marginal_logp: ", marginal_logp)
            # print("tag: ", tag)
            assert marginal_logp != None
            difference = marginal_logp[:, 1:] - marginal_logp[:, :-1]
            tag = tag[:, 1:]
            # print("difference: ", difference)
            surprisal = difference * tag
            # print("surprisal_all: ", surprisal)
            surprisal = surprisal.sum(dim=1)
            # print("surprisal_new: ", surprisal)
            return -surprisal, states_batch
        else:
            return states_batch
