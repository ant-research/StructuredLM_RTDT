import torch
import torch.nn.functional as F
from enum import Enum
from model.stack_state import ActionType, Node, State
from collections import OrderedDict
from utils.r2d2_span_tokenizer import SpanTokenizingSession
import numpy as np
import sys


class GenerationMode(Enum):
    TOPK = 0,
    NUCLEUS = 1,


def split_past_kv(past_key_values, beam_size):
    split_past_kvs = []
    for kv_layer in past_key_values:
        k, v = kv_layer
        assert k.shape[0] % beam_size == 0
        assert v.shape[0] % beam_size == 0

        k = k.view(-1, beam_size, k.shape[-3], k.shape[-2], k.shape[-1])
        v = v.view(-1, beam_size, k.shape[-3], k.shape[-2], k.shape[-1])

        split_past_kvs.append((k, v))
    return split_past_kvs


class GPTGenerationUtil:
    def __init__(self, model, device):
        self.device = device
        model.to(device)
        self.model = model
        self.embedding_dim = self.model.embedding_dim
        self.eos_id = self.model.eos_id
        self.model.eval()
    
    @torch.no_grad()
    def random_sampling(self, prefix_ids, past_kv=None, max_steps=1024, mode=GenerationMode.TOPK, mode_arg=5):
        # probslist = []
        prefix_len = prefix_ids.shape[0]
        model_input = torch.zeros((prefix_len + 1,), dtype=torch.long, device=prefix_ids.device)
        model_input[0] = self.model.bos_id
        model_input[1:] = prefix_ids

        result = self.model.gpt(model_input.unsqueeze(0), past_key_values=past_kv, return_dict=True)
        past_key_values = result.past_key_values
        
        current_step = 0
        probs = F.softmax(result.logits[:, -1, :], dim=-1)
        # probslist.append(probs)
        return_ids = []

        for current_step in range(max_steps):
            if mode == GenerationMode.TOPK:
                values, indices = probs.topk(mode_arg, dim=1)  # (1, K)
                prob = values.flatten()
                sampled_idx = torch.multinomial(prob, num_samples=1, replacement=False) # (1)
                next_token_id_t = indices[:, sampled_idx[0]]
                # print("next_token_id_t: ", next_token_id_t.shape)
                next_token_id = next_token_id_t.cpu().data.numpy()[0]
            elif mode == GenerationMode.NUCLEUS:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > mode_arg
                # ensure at least one index is selected
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                prob = probs.flatten()
                next_token_id_t = torch.multinomial(prob, num_samples=1, replacement=False) # (1)
                # print("next_token_id_t: ", next_token_id_t.shape)
                next_token_id = next_token_id_t.cpu().data.numpy()[0]
            else:
                raise Exception('unknow mode')
            
            # if next_token_id_t == torch.tensor([1429]):
            #     return past_key_values
            result = self.model.gpt(next_token_id_t, past_key_values=past_key_values, return_dict=True)
            # print("logits_shape: ", result.logits.shape)     
            probs = F.softmax(result.logits, dim=-1)
            # probslist.append(probs)
            past_key_values = result.past_key_values
            
            if next_token_id == self.eos_id:
                return return_ids
            else:
                return_ids.append(next_token_id)
        return return_ids

    @torch.no_grad()
    def batch_random_sampling(self, prefix_ids, prefix_masks, past_kv=None, max_steps=1024, mode=GenerationMode.TOPK, mode_arg=5):
        # probslist = []
        batch_size, max_prefix_len = prefix_ids.shape
        prefix_lens = (prefix_masks != 0).sum(dim=1)
        model_input = torch.zeros((batch_size, max_prefix_len + 1), dtype=torch.long, device=prefix_ids.device)
        model_input[:, 0] = self.model.bos_id
        model_input[:, 1:] = prefix_ids

        model_input = torch.where(model_input != -100, model_input, 0)
        attention_mask = torch.where(prefix_masks != 0, 1, 0)

        aux = torch.ones((batch_size, 1), device=prefix_ids.device)
        attention_mask = torch.concat((aux, attention_mask), dim=1)
        aux2 = torch.ones((batch_size, max_steps), device=prefix_ids.device)
        attention_mask = torch.concat((attention_mask, aux2), dim=1).to(dtype=float)

        result = self.model.gpt(model_input, past_key_values=past_kv, return_dict=True)
        past_key_values = result.past_key_values
        # print("past_key_values: ", len(past_key_values))
        logits = result.logits[torch.arange(batch_size), prefix_lens, :]
        # print("logits: ", logits.shape)
        # print(logits[0] == result.logits[0][prefix_lens[0]], logits[1] == result.logits[1][prefix_lens[1]])
        probs = F.softmax(logits, dim=-1)
        # probslist.append(probs)
        
        current_step = 0
        # probs = F.softmax(result.logits[:, -1, :], dim=-1)
        return_ids = torch.zeros((batch_size, max_steps), dtype=torch.long, device=prefix_ids.device)

        for current_step in range(max_steps):
            if mode == GenerationMode.TOPK:
                values, indices = probs.topk(mode_arg, dim=1)  # (N, K)
                # print("values: ", values)
                sampled_idx = torch.multinomial(values, num_samples=1, replacement=False) # (N, 1)
                # print("sampled_idx: ", sampled_idx)
                next_token_id_t = indices[torch.arange(batch_size), sampled_idx.squeeze(1)]
                # print("next_token_id_t: ", next_token_id_t)
                next_token_id = next_token_id_t.cpu().data.numpy()
            elif mode == GenerationMode.NUCLEUS:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > mode_arg
                # ensure at least one index is selected
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                # print("sorted_indices_to_remove:", sorted_indices_to_remove.shape)
                indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                # print("indices_to_remove: ", indices_to_remove.shape, indices_to_remove)
                probs[indices_to_remove] = 0
                next_token_id_t = torch.multinomial(probs, num_samples=1, replacement=False) # (N, 1)
                next_token_id_t = next_token_id_t.squeeze(1)
                # print("next_token_id_t: ", next_token_id_t.shape, next_token_id_t)
                next_token_id = next_token_id_t.cpu().data.numpy()
            else:
                raise Exception('unknow mode')

            position_ids = attention_mask[:, :max_prefix_len+1+current_step+1].long().cumsum(-1) - 1
            # print("position_ids: ", position_ids)
            # position_ids.masked_fill_(attention_mask[:, :max_prefix_len+1+current_step+1] == 1, 1)
            # print("position_ids2: ", position_ids)
            position_ids = position_ids[:, -1].unsqueeze(-1)
            # print("model_input: ", next_token_id_t.unsqueeze(1))
            # print("position_ids: ", position_ids.shape, position_ids)
            # print("attention_mask: ", attention_mask[:, :max_prefix_len+1+current_step+1].shape, attention_mask[:, :max_prefix_len+1+current_step+1])
            result = self.model.gpt(next_token_id_t.unsqueeze(1), past_key_values=past_key_values, position_ids=position_ids, attention_mask=attention_mask[:, :max_prefix_len+1+current_step+1], return_dict=True)
            probs = F.softmax(result.logits[:, 0, :], dim=-1)
            # probslist.append(probs)
            past_key_values = result.past_key_values
            
            return_ids[:, current_step] = next_token_id_t
        return return_ids


class GenerationUtil:
    def __init__(self, model, device, config, ext_vocab=None, span_tokenizer=None):
        self.model = model
        model.to(device)
        self.hidden_size = model.r2d2_input_dim
        self.head_num = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.max_seq_len = config.n_positions - 1
        self.action_layer_num = config.action_layer_num
        self.generation_layer_num = config.n_layer - config.action_layer_num
        self.device = device
        self.span_tokenizer = span_tokenizer
        self.eos_id = config.eos_token_id
        self.ext_vocab = ext_vocab
        self.embedding_dim = model.embedding_dim
        self.temperature = temperature
        self.model.eval()

    def tree_enc(self, src, span_emb=None):
        return self.model.r2d2.inside_enc(src, span_emb)

    @torch.no_grad()
    def batch_random_sampling(self, 
                              chunk_input_ids=None, 
                              chunk_masks=None, 
                              input_ids=None, 
                              masks=None, 
                              group_ids=None, 
                              past_kv=None, 
                              max_steps=1024, 
                              mode=GenerationMode.TOPK, 
                              mode_arg=2):
        # there are two generation mode: topk sampling or nuclear sampling, indicated by the parameter mode
        # mode_arg is the argument to corresponding mode
        # e.g. for TOPK mode, mode_arg=2 means selecting the top 2 as candidate tokens for random sampling
        # for NUCLEUS mode, mode_args=0.8, means selecting top 80% tokens as candidates
        # ARGUMENTS: prefix_ids: [batch_size, seq_len]

        # given prefix, generate
        vocab_size = self.model.vocab_size
        N = chunk_input_ids.shape[0]
        span_indices = []
        sent_lens_t = masks.sum(dim=1)
        sent_lens = sent_lens_t.cpu().data.numpy()
        if self.span_tokenizer is not None:
            sent_ids = input_ids.cpu().data.numpy()
            tokenizing_sess = SpanTokenizingSession(self.span_tokenizer)

            for sent_i, sent_len in enumerate(sent_lens):
                span_indices.append(tokenizing_sess.tokenize(sent_ids[sent_i, :sent_len]))
            external_ids = tokenizing_sess.span_indices.to(self.device, non_blocking=True)
        else:
            external_ids = None

        # generate 
        outputs = self.model(chunk_input_ids=chunk_input_ids, 
                             chunk_masks=chunk_masks, 
                             input_ids=input_ids,
                             group_ids=group_ids,
                             masks=masks,
                             span_ids=span_indices,
                             external_vocab_ids=external_ids,
                             past_key_values=past_kv,
                             temperature=self.temperature)

        past_action_kvs, past_token_kvs = outputs.past_kv
        action_kv_len = past_action_kvs[0][0].shape[2]
        token_kv_len = past_token_kvs[0][0].shape[2]

        encoding_beams = torch.full((N, 2 * max_steps, self.hidden_size), 
                                     fill_value=0.0, device=self.device)

        action_mask_np = np.ones((N, action_kv_len + 2 * max_steps))
        token_mask_np = np.ones((N, token_kv_len + 2 * max_steps))
        action_mask_np[:, :action_kv_len] = 0
        token_mask_np[:, :token_kv_len] = 0

        action_len_bucket = [1] * (max(group_ids) + 1)
        token_len_bucket = [1] * (max(group_ids) + 1)
        for sent_i, group_id in enumerate(group_ids):
            action_len_bucket[group_id] += 2 * sent_lens[sent_i] - 1
            token_len_bucket[group_id] += sent_lens[sent_i]

        assert action_kv_len >= max(action_len_bucket), f'{action_kv_len}/{max(action_len_bucket)}'
        assert token_kv_len == max(token_len_bucket), f'{token_kv_len}/{max(token_len_bucket)}'

        for batch_i in range(N):
            action_mask_np[batch_i, :action_len_bucket[batch_i]] = 1
            token_mask_np[batch_i, :token_len_bucket[batch_i]] = 1

        action_attn_mask = torch.tensor(action_mask_np).to(self.device, non_blocking=True)
        gen_attn_mask = torch.tensor(token_mask_np).to(self.device, non_blocking=True)

        def step(input, current_step, position_ids=None, 
                 action_past_kv=None, generation_past_kv=None):
            new_beam_size = input.shape[0]
            action_results = self.model.action_layers(inputs_embeds=input,
                                                      position_ids=position_ids,
                                                      past_key_values=action_past_kv,
                                                      attention_mask=action_attn_mask[:, :action_kv_len + current_step + 1])
            token_results = self.model.generation_layers(inputs_embeds=action_results.last_hidden_state, past_key_values=generation_past_kv,
                                                         attention_mask=gen_attn_mask[:, :token_kv_len + current_step + 1])

            action_logits = self.model.action_mlp(action_results.last_hidden_state)  # (1, 1, 2)
            token_logits = self.model.classifier(self.model.dense(token_results.last_hidden_state))

            return action_logits, token_logits, action_results.past_key_values, token_results.past_key_values
        
        current_step = 0

        states = []
        for _ in range(N):
            init_state = State(None)
            init_state.beam_id = 0
            init_state.total_len = sys.maxsize # set to max steps
            states.append(init_state)

        # initialize
        action_len_bucket_t = torch.tensor(action_len_bucket, device=self.device)
        token_len_bucket_t = torch.tensor(token_len_bucket, device=self.device)
        action_logits = outputs.action_logits.gather(dim=1, index=(action_len_bucket_t - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, 2))
        token_logits = outputs.logits.gather(dim=1, index=(token_len_bucket_t - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, outputs.logits.shape[-1]))

        # get the last action_logits and token_logits of r2d2-gen
        current_cache_offset = 0

        # for current_step in range(2 * max_steps - 1):
        while True:
            # (current_batch_size, beam_size)
            # (batch_size, beam_size)
            # print(f'action logits: {action_logits[:, :, :5]}')
            # print(f'token logits: {token_logits[:, :, :5]}')
            top_indices_batch = []
            score_mask_batch = []
            token_offsets_batch = []
            ext_vocab_ids_batch = []

            for batch_i, state in enumerate(states):
                top_indices = []
                l_top, r_top, ext_vocab_id = state.top_states(self.ext_vocab)
                if l_top is not None:
                    top_indices.append(l_top.current_step - 1)
                else:
                    top_indices.append(0)
                
                if r_top is not None:
                    top_indices.append(r_top.current_step - 1)
                else:
                    top_indices.append(0)
                
                token_offsets_batch.append(min(state.token_offset + token_len_bucket[batch_i], self.max_seq_len))
                ext_vocab_ids_batch.append(ext_vocab_id)
                top_indices_batch.append(top_indices)
                score_mask_batch.append(state.action_masks())

            top_indices_batch = torch.tensor(top_indices_batch).to(device=self.device, non_blocking=True)  # (N, 2)
            if self.ext_vocab is not None:
                ext_vocab_ids_batch = torch.tensor(ext_vocab_ids_batch, device=self.device)  # (N)
            # (1, beam_size, 2, dim)
            
            score_mask_batch = torch.tensor(score_mask_batch, dtype=torch.bool).to(device=self.device, non_blocking=True)  # (N, 2)
            # next token offset
            token_offsets_batch = torch.tensor(token_offsets_batch).to(device=self.device, non_blocking=True)  # (N)

            action_logits[:, :, ActionType.REDUCE.value].masked_fill_(~score_mask_batch[:, 1].unsqueeze(1), float('-inf'))
            # probs = F.softmax(last_logits, dim=-1)
            action_probs = F.log_softmax(action_logits, dim=-1) # (N, 1, 2)
            action_probs.masked_fill_(~score_mask_batch.unsqueeze(1), float('-inf'))  # set log_p of invalid actions to -inf
            token_probs = F.log_softmax(token_logits, dim=-1)  # (N, 1, vocab_size)
            gen_scores = action_probs[:, :, ActionType.SHIFT.value].unsqueeze(2) + token_probs  # probability of shifting and predicting the next token
            # gen_scores: (N, 1, vocab_size)
            compose_scores = action_probs[:, :, ActionType.REDUCE.value]  # (N, 1), probability of reducing
            action_scores = torch.cat([gen_scores, compose_scores.unsqueeze(2)], dim=2)  # (N, 1, vocab_size + 1)

            if mode == GenerationMode.TOPK:
                values, indices = action_scores.topk(mode_arg, dim=2)  # (N, 1, K)
                prob = values.exp().squeeze(1)
                sampled_idx = torch.multinomial(prob, num_samples=1, replacement=False) # (N, 1)
                next_token_id_t = indices.squeeze(1).gather(dim=1, index=sampled_idx)  # (N, 1)
                next_token_id = next_token_id_t.cpu().data.numpy()  # (N, 1)
            elif mode == GenerationModel.NUCLEUS:
                probs = action_scores.squeeze(1).exp()  # (N, K)
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)  # (N, K)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (N, K)
                sorted_indices_to_remove = cumulative_probs > mode_arg  # # (N, K)
                # ensure at least one index is selected
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                next_token_id_t = torch.multinomial(probs, num_samples=1, replacement=False) # (N, 1)
                next_token_id = next_token_id_t.cpu().data.numpy()
            else:
                raise Exception('unknow mode')

            is_reduce = next_token_id_t.squeeze(1) == vocab_size  # (N)
            true_indices = torch.nonzero(is_reduce) # (?, 2)
            # true_indices is in format [[batch_i, 0], ...]
            gpt_input = torch.full((N, 1, self.embedding_dim), fill_value=0.0, device=self.device)
            pos_ids = torch.full((N, 1), fill_value=0, dtype=torch.long, device=self.device)
            if true_indices.shape[0] > 0:
                top_indices = top_indices_batch[true_indices[:, 0], :]  # (?, 2)
                top_left = encoding_beams[true_indices[:, 0], top_indices[:, 0], :]  # (?, dim)
                top_right = encoding_beams[true_indices[:, 0], top_indices[:, 1], :]  # (?, dim)
                top_tensors_to_reduce = torch.stack([top_left, top_right], dim=1).unsqueeze(1)  # (?, 1, 2, dim)
                if self.ext_vocab is None:
                    _, reduce_repr = self.tree_enc(top_tensors_to_reduce)
                else:
                    ext_vocab_ids_batch = ext_vocab_ids_batch[true_indices[:, 0]]
                    span_embeddings = self.model.r2d2.ext_embeds(ext_vocab_ids_batch)  # (N, dim)
                    _, reduce_repr = self.tree_enc(top_tensors_to_reduce, span_embeddings)  # (?, 1, dim)
                # reduce_repr: (1, ?, dim)
                # scatter to corresponding encoding beams
                encoding_beams[:, current_cache_offset, :].scatter_(dim=0, 
                    index=true_indices[:, 0].unsqueeze(1).repeat(1, self.hidden_size), src=reduce_repr.squeeze(1))
                gpt_input.scatter_(dim=0, index=true_indices[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_dim),
                                   src=self.model.up_scale(reduce_repr))
                selected_pos_ids = token_offsets_batch.index_select(dim=0, index=true_indices[:, 0]) # (?)
                pos_ids.scatter_(dim=0, index=true_indices[:, 0].unsqueeze(1), src=selected_pos_ids.unsqueeze(1) - 1)
                # mask out previous token logits
                gen_attn_mask[true_indices[:, 0], token_kv_len + current_cache_offset - 1] = False

            is_shift = next_token_id_t.squeeze(1) != vocab_size
            true_indices = torch.nonzero(is_shift)
            if true_indices.shape[0] > 0:
                next_tokens = next_token_id_t[true_indices[:,0], :]  # (?, 1)
                next_token_embedding = self.model.embeddings(next_tokens)  # (?, 1, dim)
                next_token_embedding_ = self.model.down_scale(next_token_embedding)  # (?, 1, r2d2_dim)
                encoding_beams[:, current_cache_offset, :].scatter_(dim=0, 
                    index=true_indices[:, 0].unsqueeze(1).repeat(1, self.hidden_size), src=next_token_embedding_.squeeze(1))
                gpt_input.scatter_(dim=0, index=true_indices[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_dim), 
                                   src=next_token_embedding)
                selected_pos_ids = token_offsets_batch.index_select(dim=0, index=true_indices[:, 0]) # (?)
                # (1, B)
                pos_ids.scatter_(dim=0, index=true_indices[:, 0].unsqueeze(1), src=selected_pos_ids.unsqueeze(1))

            action_logits, token_logits, past_action_kvs, past_token_kvs = step(
                gpt_input.view(-1, 1, self.embedding_dim), current_cache_offset, position_ids=pos_ids,
                action_past_kv=past_action_kvs, generation_past_kv=past_token_kvs)
            # print(f'token offsets: {token_offsets_batch}, action hist len: {past_action_kvs[0][0].shape[2]}, token hist len: {past_token_kvs[0][0].shape[2]}')
            # print(action_logits.shape)
            # update token_mask
            # print(pos_ids)
            next_states = []
            for batch_i, state in enumerate(states):
                if next_token_id[batch_i, 0] != vocab_size:
                    state = state.act(ActionType.SHIFT, token_id=next_token_id[batch_i, 0])
                elif next_token_id[batch_i, 0] == vocab_size:
                    state = state.act(ActionType.REDUCE)
                else:
                    raise Exception(f'Unexisted action id: {action_id}, step: {step}, actions_np: {actions_np}, action_ids: {action_ids}')
                assert state is not None
                assert not state.invalid_state
                next_states.append(state)
            states = next_states
            if min(map(lambda x: x.token_offset, states)) == max_steps:
                break

            current_cache_offset += 1

        return states

    @torch.no_grad()
    def random_sampling(self,
                        chunk_input_ids=None, 
                        chunk_masks=None, 
                        input_ids=None, 
                        masks=None, 
                        past_kv=None, 
                        max_steps=1024, 
                        mode=GenerationMode.TOPK, 
                        mode_arg=2):
        # there are two generation mode: topk sampling or nuclear sampling, indicated by the parameter mode
        # mode_arg is the argument to corresponding mode
        # e.g. for TOPK mode, mode_arg=2 means selecting the top 2 as candidate tokens for random sampling
        # for NUCLEUS mode, mode_args=0.8, means selecting top 80% tokens as candidates

        # given prefix, generate
        vocab_size = self.model.vocab_size
        assert chunk_input_ids.shape[0] == 1
        
        span_indices = []
        if self.span_tokenizer is not None:
            sent_ids = input_ids.cpu().data.numpy()
            tokenizing_sess = SpanTokenizingSession(self.span_tokenizer)

            span_indices.append(tokenizing_sess.tokenize(sent_ids[0]))
            external_ids = tokenizing_sess.span_indices.to(self.device, non_blocking=True)
        else:
            external_ids = None

        # generate 
        outputs = self.model(chunk_input_ids=chunk_input_ids, 
                             chunk_masks=chunk_masks, 
                             input_ids=input_ids,
                             group_ids=[0] * input_ids.shape[0],
                             masks=masks,
                             span_ids=span_indices,
                             external_vocab_ids=external_ids,
                             past_key_values=past_kv,
                             temperature=self.temperature)
                             
        encoding_beams = torch.full((1, 2 * max_steps, self.hidden_size), 
                                     fill_value=0.0, device=self.device)
        
        current_step = 0

        state = State(None)
        state.beam_id = 0
        state.total_len = sys.maxsize # set to max steps

        # initialize
        sent_lens = masks.sum(dim=1).cpu().data.numpy()
        action_last_idx = sum([2 * x - 1 for x in sent_lens])
        token_last_idx = sum(sent_lens)
        action_logits = outputs.action_logits[:, action_last_idx, :].unsqueeze(1)  #(1, 1, 2)
        token_logits = outputs.logits[:, token_last_idx, :].unsqueeze(1)  # (1, 1, vocab_size)
        past_action_kvs, past_token_kvs = outputs.past_kv
        past_action_kvs = [(k[:, :, :action_last_idx + 1, :], v[:, :, :action_last_idx+1, :]) for k,v in past_action_kvs]
        position_offset = past_token_kvs[0][0].shape[2]
        assert position_offset == token_last_idx + 1

        def step(input, position_ids=None, 
                 action_past_kv=None, generation_past_kv=None):
            new_beam_size = input.shape[0]
            action_results = self.model.action_layers(inputs_embeds=input,
                                                      position_ids=position_ids,
                                                      past_key_values=action_past_kv)
            token_results = self.model.generation_layers(inputs_embeds=action_results.last_hidden_state, past_key_values=generation_past_kv)

            action_logits = self.model.action_mlp(action_results.last_hidden_state)  # (1, 1, 2)
            token_logits = self.model.classifier(self.model.dense(token_results.last_hidden_state))
            return action_logits, token_logits, action_results.past_key_values, token_results.past_key_values


        # get the last action_logits and token_logits of r2d2-gen-fast
        current_cache_offset = 0

        # TODO: 根据生成token的数量更新current_step
        while current_step < max_steps:
            # (current_batch_size, beam_size)
            # (batch_size, beam_size)
            # print(f'action logits: {action_logits[:, :, :5]}')
            # print(f'token logits: {token_logits[:, :, :5]}')
            top_indices_batch = []
            score_mask_batch = []
            token_offsets_batch = []
            ext_vocab_ids_batch = []

            score_mask = []
            token_offsets = []
            ext_vocab_ids = []
            
            l_top, r_top, ext_vocab_id = state.top_states(self.ext_vocab)
            if l_top is not None:
                top_indices_batch.append(l_top.current_step - 1)
            else:
                top_indices_batch.append(0)
            
            if r_top is not None:
                top_indices_batch.append(r_top.current_step - 1)
            else:
                top_indices_batch.append(0)
            
            score_mask.append(state.action_masks())
            token_offsets.append(state.token_offset + position_offset)
            ext_vocab_ids.append(ext_vocab_id)

            ext_vocab_ids_batch.append(ext_vocab_ids)
            top_indices_batch = [top_indices_batch]
            score_mask_batch.append(score_mask)
            token_offsets_batch.append(token_offsets)
            top_indices_batch = torch.tensor(top_indices_batch).to(device=self.device, non_blocking=True)  # (1, 2)
            if self.ext_vocab is not None:
                ext_vocab_ids_batch = torch.tensor(ext_vocab_ids_batch, device=self.device)  # (1, beam_size, 2, dim)
            
            score_mask_batch = torch.tensor(score_mask_batch, dtype=torch.bool).to(device=self.device, non_blocking=True)  # (N, B, 2)
            # next token offset
            token_offsets_batch = torch.tensor(token_offsets_batch).to(device=self.device, non_blocking=True)  # (N, B)

            action_logits[:, :, ActionType.REDUCE.value].masked_fill_(~score_mask_batch[:, :, 1], float('-inf'))
            # probs = F.softmax(last_logits, dim=-1)
            action_probs = F.log_softmax(action_logits, dim=-1) # (1, B, 2)
            action_probs.masked_fill_(~score_mask_batch, float('-inf'))  # set log_p of invalid actions to -inf
            token_probs = F.log_softmax(token_logits, dim=-1)  # (1, B, vocab_size)
            gen_scores = action_probs[:, :, ActionType.SHIFT.value].unsqueeze(2) + token_probs  # probability of shifting and predicting the next token
            # gen_scores: (N, B, vocab_size)
            compose_scores = action_probs[:, :, ActionType.REDUCE.value]  # (1, 1), probability of reducing
            action_scores = torch.cat([gen_scores, compose_scores.unsqueeze(2)], dim=2)  # (1, 1, vocab_size + 1)

            if mode == GenerationMode.TOPK:
                values, indices = action_scores.topk(mode_arg, dim=2)  # (1, 1, K)
                # print("indices: ", indices)
                prob = values.flatten().exp()
                sampled_idx = torch.multinomial(prob, num_samples=1, replacement=False) # (1)
                next_token_id_t = indices[:, :, sampled_idx[0]]
                next_token_id = next_token_id_t.cpu().data.numpy()[0][0]
            elif mode == GenerationMode.NUCLEUS:
                probs = action_scores.exp()
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > mode_arg
                # ensure at least one index is selected
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                next_token_id_t = torch.multinomial(probs, 1) # (1)
                next_token_id = next_token_id_t.cpu().data.numpy()[0][0]
            else:
                raise Exception('unknown mode')

            is_reduce = (next_token_id == vocab_size)  # (1, beam_size)
            # print(f'is reduce: {is_reduce}, current step: {current_step}')
            gpt_input = torch.full((1, 1, self.embedding_dim), fill_value=0.0, device=self.device)
            if is_reduce:
                # print(f'fetch {top_indices_batch[0, 0]} and {top_indices_batch[0, 1]}')
                top_left = encoding_beams[0, top_indices_batch[0, 0], :]  # (dim)
                top_right = encoding_beams[0, top_indices_batch[0, 1], :]  # (dim)
                top_tensors = torch.stack([top_left, top_right], dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, 2, dim)
                if self.ext_vocab is not None:
                    # TODO: check if ext_vocab_ids_batch shape is correct
                    span_embeddings = self.model.r2d2.ext_embeds(ext_vocab_ids_batch)
                    span_embeddings = span_embeddings.squeeze(1)
                    _, reduce_repr = self.tree_enc(top_tensors, span_embeddings)  # (1, 1, dim)
                else:
                    _, reduce_repr = self.tree_enc(top_tensors)  # (1, 1, dim)
                # reduce_repr: (1, ?, dim)
                # scatter to corresponding encoding beams
                # print(f'reduce write to {current_cache_offset}')
                encoding_beams[0, current_cache_offset, :]= reduce_repr.squeeze(0).squeeze(0)
                # if reduce, remove the previous kv
                past_token_kvs = [(k[:, :, :-1, :], v[:, :, :-1, :]) for k, v in past_token_kvs]
                token_offsets_batch = token_offsets_batch - 1
                gpt_input = self.model.up_scale(reduce_repr)
            else:
                next_token_embedding = self.model.embeddings(next_token_id_t)
                next_token_embedding_ = self.model.down_scale(next_token_embedding)
                # print(f'shift write to {current_cache_offset}')
                encoding_beams[0, current_cache_offset, :] = next_token_embedding_.squeeze(0).squeeze(0)
                gpt_input = next_token_embedding.unsqueeze(0)

            # print(token_offsets_batch)
            # print(token_offsets_batch)
            action_logits, token_logits, past_action_kvs, past_token_kvs = step(
                gpt_input.view(-1, 1, self.embedding_dim), position_ids=token_offsets_batch,
                action_past_kv=past_action_kvs, generation_past_kv=past_token_kvs)
            # print(f'token offsets: {token_offsets_batch}, action hist len: {past_action_kvs[0][0].shape[2]}, token hist len: {past_token_kvs[0][0].shape[2]}')
            # print(action_logits.shape)
            # update token_mask
            
            if next_token_id == self.eos_id:
                # over
                return state
            elif next_token_id != vocab_size:
                current_step += 1
                state = state.act(ActionType.SHIFT, token_id=next_token_id)
            elif next_token_id == vocab_size:
                state = state.act(ActionType.REDUCE)
            else:
                raise Exception(f'Unexisted action id: {action_id}, step: {step}, actions_np: {actions_np}, action_ids: {action_ids}')
            current_cache_offset += 1
        
        # TODO: return the best state
        return state

    @torch.no_grad()
    def beam_gen(self, prefix_ids, past_kv=None, beam_size=10, max_steps=1024):
        # given prefix, generate
        vocab_size = self.model.vocab_size
        assert beam_size <= vocab_size
        prefix_ids_np = prefix_ids.cpu().data.numpy()

        span_indices = []
        if self.span_tokenizer is not None:
            external_dict = OrderedDict()
            results = self.span_tokenizer.tokenize(prefix_ids_np)
            span_idx = np.zeros((len(results),))
            if len(results) > 0:
                for group_id in range(len(results) // 3):
                    idx, span_len, span_id = results[group_id * 3: group_id * 3 + 3]
                    span_idx[group_id * 3] = idx - span_len + 1
                    span_idx[group_id * 3 + 1] = idx
                    if span_id + 1 not in external_dict:
                        external_dict[span_id + 1] = external_vocab_idx
                        external_vocab_idx += 1
                    span_idx[group_id * 3 + 2] = external_dict[span_id + 1]
            span_indices.append(span_idx)
            external_ids_np = torch.tensor(np.array([0] + list(external_dict.keys())))
        else:
            external_ids_np = None

        prefix_ids = prefix_ids.unsqueeze(0)  # (1, L) 

        # generate 
        outputs = self.model(chunk_input_ids=prefix_ids, 
                             chunk_masks=torch.ones_like(prefix_ids), 
                             input_ids=prefix_ids,
                             group_ids=[0],
                             masks=torch.ones_like(prefix_ids),
                             span_ids=span_indices,
                             external_vocab_ids=external_ids_np,
                             past_key_values=past_kv)


        past_action_kvs, past_token_kvs = outputs.past_kv
        position_offset = past_token_kvs[0][0].shape[2]
        action_offset = past_action_kvs[0][0].shape[2]

        encoding_beams = torch.full((1, beam_size, max_steps, self.hidden_size), 
                                     fill_value=0.0, device=self.device)

        action_key_beams = [torch.full((1, beam_size, self.head_num, action_offset + max_steps, self.head_dim), fill_value=0.0, device=self.device)
                            for _ in range(self.action_layer_num)]
        action_value_beams = [torch.full((1, beam_size, self.head_num, action_offset + max_steps, self.head_dim), fill_value=0.0, device=self.device)
                              for _ in range(self.action_layer_num)]

        token_key_beams = [torch.full((1, beam_size, self.head_num, position_offset + max_steps // 2 + 1, self.head_dim), fill_value=0.0, device=self.device)
                           for _ in range(self.generation_layer_num)]
        token_value_beams = [torch.full((1, beam_size, self.head_num, position_offset + max_steps // 2 + 1, self.head_dim), fill_value=0.0, device=self.device)
                             for _ in range(self.generation_layer_num)]
        token_mask = torch.full((1, beam_size, max_steps // 2 + 1 + position_offset), dtype=torch.bool, fill_value=1, device=self.device)
        beam_scores = torch.zeros(1, beam_size, max_steps + 1, device=self.device).fill_(float('-inf'))  # (batch_size, current_beam_size)

        for layer_i in range(self.action_layer_num):
            action_key_beams[layer_i][:, 0, :, :action_offset, :] = past_action_kvs[layer_i][0]
            action_value_beams[layer_i][:, 0, :, :action_offset, :] = past_action_kvs[layer_i][1]

        for layer_i in range(self.generation_layer_num):
            token_key_beams[layer_i][:, 0, :, :position_offset, :] = past_token_kvs[layer_i][0]
            token_value_beams[layer_i][:, 0, :, :position_offset, :] = past_token_kvs[layer_i][1]

        finished_beams = []

        def step(input, current_cache_offset, position_ids=None, 
                 action_past_kv=None, generation_past_kv=None, attn_mask=None):
            # print(input.shape)
            # print(action_past_kv[0][0].shape)
            # print(position_ids.shape)
            # print(position_ids)
            new_beam_size = input.shape[0]
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
            past_kv_beams = split_past_kv(action_results.past_key_values, new_beam_size)

            for layer_i, (k, v) in enumerate(past_kv_beams):
                _k = k.view(-1, new_beam_size, k.shape[-3], k.shape[-2], k.shape[-1])
                _v = v.view(-1, new_beam_size, v.shape[-3], v.shape[-2], v.shape[-1])
                # print(_k.shape)
                # print(new_beam_size)
                # print(current_cache_offset + 1 + action_offset)
                action_key_beams[layer_i][:, :new_beam_size, :, :current_cache_offset + 1 + action_offset, :] = _k
                action_value_beams[layer_i][:, :new_beam_size, :, :current_cache_offset + 1 + action_offset, :] = _v

            past_kv_len = token_results.past_key_values[0][0].shape[2]
            past_kv_beams = split_past_kv(token_results.past_key_values, new_beam_size)

            for layer_i, (k, v) in enumerate(past_kv_beams):
                _k = k.view(-1, new_beam_size, k.shape[-3], k.shape[-2], k.shape[-1])
                _v = v.view(-1, new_beam_size, v.shape[-3], v.shape[-2], v.shape[-1])
                # print(_k.shape)
                # print(current_cache_offset + 1 + position_offset)
                token_key_beams[layer_i][:, :new_beam_size, :, :current_cache_offset + 1 + position_offset, :] = _k
                token_value_beams[layer_i][:, :new_beam_size, :, :current_cache_offset + 1 + position_offset, :] = _v
            
            # Update token/action beams
            action_logits = self.model.action_mlp(action_results.last_hidden_state)  # (N * B, 2)
            token_logits = self.model.classifier(model.dense(token_results.last_hidden_state))
            return action_logits, token_logits
    
        def prepare_kv(beam_ids, current_cache_offset):
            past_action_kv = []
            # nonlocal action_key_beams
            # nonlocal action_value_beams
            for k, v in zip(action_key_beams, action_value_beams):
                k_ = k[:, beam_ids, :, :current_cache_offset + action_offset, :]
                v_ = v[:, beam_ids, :, :current_cache_offset + action_offset, :]
                past_action_kv.append((k_.view(-1, self.head_num, current_cache_offset + action_offset, self.head_dim),
                                       v_.view(-1, self.head_num, current_cache_offset + action_offset, self.head_dim)))

            past_token_kv = []
            # nonlocal token_key_beams
            # nonlocal token_value_beams
            for k, v in zip(token_key_beams, token_value_beams):
                k_ = k[:, beam_ids, :, :current_cache_offset + position_offset, :]
                v_ = v[:, beam_ids, :, :current_cache_offset + position_offset, :]
                past_token_kv.append((k_.view(-1, self.head_num, current_cache_offset + position_offset, self.head_dim),
                                      v_.view(-1, self.head_num, current_cache_offset + position_offset, self.head_dim)))
            return past_action_kv, past_token_kv
        
        current_step = 0

        states = []
        init_state = State(None)
        init_state.beam_id = 0
        init_state.total_len = max_steps # set to max steps
        states.append(init_state)

        # initialize
        beam_scores[:, 0, 0] = 0
        action_logits = outputs.action_logits[:, -1, :].unsqueeze(1)  #(1, 1, 2)
        token_logits = outputs.logits[:, -1, :].unsqueeze(1)  # (1, 1, vocab_size)

        # get the last action_logits and token_logits of r2d2-gen
        current_cache_offset = 0

        finished_states = []
        for current_step in range(2 * max_steps - 1):
            # (current_batch_size, beam_size)
            # (batch_size, beam_size)
            top_indices_batch = []
            score_mask_batch = []
            token_offsets_batch = []
            ext_vocab_ids_batch = []
            current_beam_size = len(states)

            top_indices_left = []
            top_indices_right = []
            score_mask = []
            token_offsets = []
            ext_vocab_ids = []
            for state in states:
                l_top, r_top, ext_vocab_id = state.top_states(self.ext_vocab)
                if l_top is not None:
                    top_indices_left.append((0, l_top.beam_id, l_top.current_step))
                else:
                    top_indices_left.append((0, 0, 0))
                
                if r_top is not None:
                    top_indices_right.append((0, r_top.beam_id, r_top.current_step))
                else:
                    top_indices_right.append((0, 0, 0))
                
                score_mask.append(state.action_masks())
                token_offsets.append(state.token_offset + position_offset)
                ext_vocab_ids.append(ext_vocab_id)

            ext_vocab_ids_batch.append(ext_vocab_ids)
            top_indices_batch.append((top_indices_left, top_indices_right))
            score_mask_batch.append(score_mask)
            token_offsets_batch.append(token_offsets)
            top_indices_batch = torch.tensor(top_indices_batch).to(device=self.device, non_blocking=True)  # (N, 2, B, 3)
            if self.ext_vocab is not None:
                ext_vocab_ids_batch = torch.tensor(ext_vocab_ids_batch, device=self.device)
            current_beam_scores = beam_scores[:, :current_beam_size, current_cache_offset]  # (N, B)
            # current_action_scores = beam_action_scores[batch_ids, :, current_cache_offset - 1][:, :current_beam_size]

            top_indices_batch = top_indices_batch.permute(1, 0, 2, 3)  # (2, N, B, 3)
            top_indices_batch = top_indices_batch.reshape(2, current_beam_size, 3)

            top_left = encoding_beams[top_indices_batch[0, :, 0], top_indices_batch[0, :, 1], top_indices_batch[0, :, 2], :]
            top_left = top_left.view(1, current_beam_size, -1)
            top_right = encoding_beams[top_indices_batch[1, :, 0], top_indices_batch[1, :, 1], top_indices_batch[1, :, 2], :]  # (N * B, dim)
            top_right = top_right.view(1, current_beam_size, -1)

            top_tensors = torch.stack([top_left, top_right], dim=2)
            # (1, beam_size, 2, dim)
            
            score_mask_batch = torch.tensor(score_mask_batch, dtype=torch.bool).to(device=self.device, non_blocking=True)  # (N, B, 2)
            # next token offset
            token_offsets_batch = torch.tensor(token_offsets_batch).to(device=self.device, non_blocking=True)  # (N, B)

            # TODO: replace with GPT
            # for shift only, set reduce logits to -inf
            # print(f'action logits: {action_logits[:, :, ActionType.REDUCE.value].shape}')
            # print(f'score mask batch: {score_mask_batch[:, :, 1].shape}')
            action_logits[:, :, ActionType.REDUCE.value].masked_fill_(~score_mask_batch[:, :, 1], float('-inf'))
            # probs = F.softmax(last_logits, dim=-1)
            action_probs = F.log_softmax(action_logits, dim=-1) # (1, B, 2)
            action_probs.masked_fill_(~score_mask_batch, float('-inf'))  # set log_p of invalid actions to -inf
            token_probs = F.log_softmax(token_logits, dim=-1)  # (1, B, vocab_size)

            gen_scores = action_probs[:, :, ActionType.SHIFT.value].unsqueeze(2) + token_probs  # probability of shifting and predicting the next token
            # gen_scores: (N, B, vocab_size)
            compose_scores = action_probs[:, :, ActionType.REDUCE.value]  # (N, B), probability of reducing
            action_scores = torch.cat([gen_scores, compose_scores.unsqueeze(2)], dim=2)  # (N, B, vocab_size + 1)

            # pure_shift_scores = (1 - reduce_scores.exp()).log()
            # pure_action_scores = torch.stack([pure_shift_scores, reduce_scores], dim=2)
            
            # print(current_beam_scores)
            # print(action_scores)
            # print('--' * 10)
            current_beam_scores = current_beam_scores.unsqueeze(2) + action_scores  # (1, B, vocab_size + 1)

            # rank actions
            sorted_scores, top_beam_indices = current_beam_scores.view(1, -1).sort(dim=1, descending=True)
            top_beam_indices = top_beam_indices[:, :beam_size] # (N, B)
            sorted_scores = sorted_scores[:, :beam_size]

            beam_scores[:, :sorted_scores.shape[1], current_cache_offset + 1] = sorted_scores
            # beam_action_scores[batch_ids, :sorted_scores.shape[1], current_cache_offset] = pure_action_scores.view(N, -1)[:, top_beam_indices]

            # print(beam_ppl)
            # (N, B)
            # actions = top_beam_indices % vocab_size == 0
            token_ids = top_beam_indices % (vocab_size + 1)
            beam_ids = top_beam_indices // (vocab_size + 1)  # (N, B)

            token_ids_np = token_ids.to('cpu', non_blocking=True)  # (1, max_beam)
            beam_ids_np = beam_ids.to('cpu', non_blocking=True)  # (1, max_beam)

            # TODO: for those to predict next tokens, prepare embedding of the predicted tokens
            # for those to reduce, prepare compositional representations
            is_reduce = token_ids == vocab_size  # (1, beam_size)
            true_indices = torch.nonzero(is_reduce) # (?, 2)
            gpt_input = torch.full((1, beam_size, self.embedding_dim), fill_value=0.0, device=self.device)
            pos_ids = torch.full((1, beam_size), fill_value=0, dtype=torch.long, device=self.device)
            if true_indices.shape[0] > 0:
                reduce_beam_ids = beam_ids[true_indices[:,0], true_indices[:, 1]]  # (?)
                # top_tensors: (1, beam_size, 2, dim)
                top_tensors_to_reduce = top_tensors.index_select(dim=1, index=reduce_beam_ids)  # (1, ?, 2, dim)
                _, reduce_repr = self.tree_enc(top_tensors_to_reduce)
                # reduce_repr: (1, ?, dim)
                # scatter to corresponding encoding beams
                encoding_beams[:, :, current_cache_offset, :].scatter_(dim=1, 
                    index=true_indices[:, 1].unsqueeze(0).unsqueeze(2).repeat(1, 1, self.hidden_size), src=reduce_repr)
                gpt_input.scatter_(dim=1, index=true_indices[:, 1].unsqueeze(0).unsqueeze(2).repeat(1, 1, self.embedding_dim),
                                   src=self.model.up_scale(reduce_repr))
                selected_pos_ids = token_offsets_batch.index_select(dim=1, index=reduce_beam_ids) # (1, ?)
                pos_ids.scatter_(dim=1, index=true_indices[:, 1].unsqueeze(0), src=selected_pos_ids)
                # update token mask
                selected_mask = token_mask[:, :, :current_cache_offset + position_offset + 1].index_select(dim=1, index=reduce_beam_ids)
                # (B, beam_size, L)
                selected_mask[:, :, current_cache_offset + position_offset - 1] = False  # mask, not attend to
                # token_mask[:, :, :current_cache_offset + position_offset + 1].scatter_(dim=1, index=true_indices[:, 1], src=selected_mask)
                token_mask[:, :, :current_cache_offset + position_offset + 1].scatter_(dim=1, 
                    index=true_indices[:, 1].unsqueeze(0).unsqueeze(2).repeat(1, 1, current_cache_offset + position_offset + 1), src=selected_mask)
            is_shift = token_ids != vocab_size
            true_indices = torch.nonzero(is_shift)
            if true_indices.shape[0] > 0:
                shift_beam_ids = beam_ids[true_indices[:,0], true_indices[:, 1]]  # (?)
                next_tokens = token_ids[true_indices[:,0], true_indices[:, 1]]  # (?)
                next_token_embedding = self.model.embeddings(next_tokens)
                next_token_embedding_ = self.model.down_scale(next_token_embedding)
                encoding_beams[:, :, current_cache_offset, :].scatter_(dim=1, 
                    index=true_indices[:, 1].unsqueeze(0).unsqueeze(2).repeat(1, 1, self.hidden_size), src=next_token_embedding_.unsqueeze(0))
                gpt_input.scatter_(dim=1, index=true_indices[:, 1].unsqueeze(0).unsqueeze(2).repeat(1, 1, self.embedding_dim), 
                                   src=next_token_embedding.unsqueeze(0))
                selected_pos_ids = token_offsets_batch.index_select(dim=1, index=shift_beam_ids) # (?)
                # (1, B)
                pos_ids.scatter_(dim=1, index=true_indices[:, 1].unsqueeze(0), src=selected_pos_ids + 1)
                # update token mask
                selected_mask = token_mask[:, :, :current_cache_offset + position_offset + 1].index_select(dim=1, index=shift_beam_ids)
                # (B, beam_size, L)
                token_mask[:, :, :current_cache_offset + position_offset + 1].scatter_(dim=1, 
                    index=true_indices[:, 1].unsqueeze(0).unsqueeze(2).repeat(1, 1, current_cache_offset + position_offset + 1), src=selected_mask)
            gpt_mask = token_mask[:, :, :current_cache_offset + position_offset + 1]
            # print(f'next gpt inputs shape: {gpt_inputs.shape}')
            # token_offsets_batch: [N, B]
            # print(f'gpt_mask_sum: {gpt_mask.sum(dim=-1)}, token offset batch: {pos_ids}')
            # assert gpt_mask.sum(dim=-1) == token_offsets_batch.gather(1, top_beam_indices)
            # print(f'gpt mask: {gpt_mask.shape}, current cache_offset : {current_cache_offset}')
            action_past_kv, generation_past_kv = prepare_kv(beam_ids, current_cache_offset)
            # print(f'action past k: {action_past_kv[0][0].shape}')
            action_logits, token_logits = step(
                gpt_input.view(-1, 1, self.embedding_dim), current_cache_offset, position_ids=pos_ids,
                action_past_kv=action_past_kv, generation_past_kv=generation_past_kv,
                attn_mask=gpt_mask)
            # ? why transpose?
            action_logits = action_logits.transpose(0, 1)
            token_logits = token_logits.transpose(0, 1)
            # update token_mask

            token_ids_np = token_ids_np.data.numpy()
            beam_ids_np = beam_ids_np.data.numpy()

            assert len(token_ids_np) == len(beam_ids_np)
            for idx, (token_ids, beam_ids) in enumerate(zip(token_ids_np, beam_ids_np)):
                new_states = []
                assert len(token_ids) == len(beam_ids)
                for new_beam_id, (token_id, beam_idx) in enumerate(zip(token_ids, beam_ids)):
                    # TODO: judge if is end of text
                    if token_id == self.eos_id:
                        # over
                        finished_states.append(states[beam_idx])
                        if finished_states >= topK:
                            return finished_states
                    elif token_id != vocab_size:
                        next_state = states[beam_idx].act(ActionType.SHIFT, token_id=token_id)
                    elif token_id == vocab_size:
                        next_state = states[beam_idx].act(ActionType.REDUCE)
                    else:
                        raise Exception(f'Unexisted action id: {action_id}, step: {step}, actions_np: {actions_np}, action_ids: {action_ids}')
                    next_state.beam_id = new_beam_id
                    assert next_state.current_step == current_cache_offset + 1, \
                            f'new step: {next_state.current_step} vs {current_cache_offset}'
                    new_states.append(next_state)
                states = new_states
            
            current_cache_offset += 1
        
        # TODO: return the best state
        return finished_states

