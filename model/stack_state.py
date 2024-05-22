from enum import Enum
from typing import List
import torch


class ActionType(Enum):
    SHIFT = 0
    REDUCE= 1

class Node:
    def __init__(self, left, right, owner, terminal_id=-1, token_id=-1) -> None:
        self.left = left
        self.right = right
        self.terminal_id = terminal_id
        self.token_id = token_id
        self.owner = owner

        if self.left is not None and self.right is not None:
            self.i = self.left.i
            self.j = self.right.j
        else:
            self.i = terminal_id
            self.j = terminal_id

        if token_id != -1:
            self.expr = str(token_id)
        elif self.left is not None and self.right is not None:
            self.expr = f'{self.left.expr},{self.right.expr}'
        else:
            self.expr = ''
    
    def __repr__(self):
        if self.terminal_id != -1:
            return f'{self.terminal_id}'
        else:
            assert self.left is not None and self.right is not None
            return f'({self.left}, {self.right})'

    def to_ids(self):
        if self.left is None and self.right is None:
            return f'{self.token_id}'
        else:
            return f'({self.left.to_ids()} {self.right.to_ids()})'

    def to_tokens(self, vocab):
        if self.left is None and self.right is None:
            return f'{vocab[self.token_id]}'
        else:
            return f'({self.left.to_tokens(vocab)} {self.right.to_tokens(vocab)})'

def shift_conflict_with_atom_spans(i, j, atom_spans):
    for atom_i, atom_j in atom_spans:
        if j == atom_j and i > atom_i:
            # shift before completing reduce action for an atom span is not allowed
            return True
    return False

def reduce_conflict_with_atom_spans(i, j, atom_spans):
    for atom_i, atom_j in atom_spans:
        if i < atom_i <= j < atom_j or atom_i < i <= atom_j < j:
            # reduce break any atom span is not allowed
            return True
    return False

INIT_VAL = 0.0

class State:
    def __init__(self, prev_state = None) -> None:
        self.prev_state = prev_state
        self.prev_shift = None
        self.total_len = prev_state.total_len if prev_state is not None else 0
        self.current_action = ActionType.SHIFT
        self.current_step = 0 if prev_state is None else prev_state.current_step + 1
        self.stack_top = None  # reference to the node on the top of the stack
        self.stack_depth = 0 # num of un-reduced nodes
        self.input_offset = 0 if prev_state is None else prev_state.input_offset  # the index to the next token to shift
        self.invalid_state = False
        self.updated = False
        self.score = 0
        self.beam_id = -1
        self.batch_idx = -1
    
    def top_states(self, ext_vocab=None):
        if self.invalid_state:
            return None, None, 0
        ext_vocab_id = -1
        if ext_vocab is not None:
            if self.prev_shift is not None and self.prev_shift.stack_top is not None:
                assert self.stack_top is not None
                new_expr = f'{self.prev_shift.stack_top.expr},{self.stack_top.expr}'
                ext_vocab_id = ext_vocab.get(new_expr, -1)
        return self.prev_shift, self, ext_vocab_id + 1
    
    @property
    def token_offset(self):
        return min(self.input_offset, self.total_len)

    @property
    def is_finished(self):
        if self.token_offset == self.total_len and self.stack_top.i == 0:
            return True
        return False
    
    def action_masks(self, atom_spans=None):
        if self.invalid_state:
            return [False, False]
        # is shift valid?
        shift_valid = False
        reduce_valid = False
        if self.input_offset < self.total_len:
            shift_valid = True
            if atom_spans is not None and self.stack_top is not None:
                span_i, span_j = self.stack_top.i, self.stack_top.j
                if shift_conflict_with_atom_spans(span_i, span_j, atom_spans):
                    shift_valid = False
            
        if self.stack_depth >= 2:
            reduce_valid = True
            if atom_spans is not None:
                span_i, span_j = self.prev_shift.stack_top.i, self.stack_top.j
                if reduce_conflict_with_atom_spans(span_i, span_j, atom_spans):
                    reduce_valid = False
        return [shift_valid, reduce_valid]
            
    def act(self, action: ActionType, token_id = -1):
        if action == ActionType.SHIFT:
            # assert self.input_offset < self.total_len, f'input offset : {self.input_offset}, total len: {self.total_len}'
            if self.input_offset >= self.total_len or self.invalid_state:
                # nothing to shift, invalid state
                next_state = State(self)
                next_state.invalid_state = True
                next_state.input_offset = self.total_len - 1
                return next_state
            next_state = State(self)
            next_state.prev_shift = self
            next_state.current_action = ActionType.SHIFT
            next_state.stack_top = Node(None, None, next_state, 
                                        terminal_id=self.input_offset, 
                                        token_id=token_id)
            next_state.input_offset = self.input_offset + 1
            next_state.stack_depth = self.stack_depth + 1
            next_state.token_id = token_id
            return next_state
        elif action == ActionType.REDUCE:
            # assert self.stack_depth >= 2
            if self.stack_depth < 2 or self.invalid_state:
                # nothing to reduce, invalid state
                next_state = State(self)
                next_state.invalid_state = True
                return next_state
            next_state = State(self)
            next_state.current_action = ActionType.REDUCE
            next_state.stack_depth = self.stack_depth - 1
            left = self.prev_shift.stack_top
            right = self.stack_top
            next_state.stack_top = Node(left, right, next_state, terminal_id=-1)
            next_state.prev_shift = self.prev_shift.prev_shift
            return next_state
        else:
            raise Exception('Unsupported action type!')
        
    def __repr__(self):
        if self.invalid_state:
            return 'invalid state'
        if self.prev_state is not None:
            act_name = f'R' if self.current_action == ActionType.REDUCE else f'S'
            return f'{self.prev_state} {act_name}'
        return ''

    def to_ids(self):
        if self.current_action == ActionType.REDUCE or self.prev_shift is None:
            current_expr = ''
            if self.stack_top is not None:
                current_expr = self.stack_top.to_ids()
            if self.prev_shift is not None:
                return f'{self.prev_shift.to_ids()}, {current_expr}'
            else:
                return current_expr
        else:
            assert self.stack_top is not None
            if self.prev_shift is not None:
                return f'{self.prev_shift.to_ids()}, {self.stack_top.to_ids()}'
            else:
                return f'{self.stack_top.to_ids()}'

    def to_tokens(self, vocab):
        if self.current_action == ActionType.REDUCE or self.prev_shift is None:
            current_expr = ''
            if self.stack_top is not None:
                current_expr = self.stack_top.to_tokens(vocab)
            if self.prev_shift is not None:
                return f'{self.prev_shift.to_tokens(vocab)}, {current_expr}'
            else:
                return current_expr
        else:
            assert self.stack_top is not None
            if self.prev_shift is not None:
                return f'{self.prev_shift.to_tokens(vocab)}, {self.stack_top.to_tokens(vocab)}'
            else:
                return f'{self.stack_top.to_tokens(vocab)}'


class BeamContext:
    def __init__(self, model, batch_size, max_beam_size, max_input_len, compose_dim, config, device, 
                 action_kv_history_len=0, token_kv_history_len=0):
        self.device = device
        self.embeddings = model.embeddings
        self.up_scale = model.up_scale
        self.down_scale = model.down_scale
        # self.layer_norm = model.layer_norm
        self.max_beam_size = max_beam_size
        self.compose_dim = compose_dim
        self.compose_cache = torch.full((batch_size, max_beam_size, max_input_len * 2, compose_dim), fill_value=0.0, device=device)
        # assert input_dim % num_head
        self.head_num = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.action_layer_num = config.action_layer_num
        self.token_layer_num = config.n_layer - config.action_layer_num
        self.token_kv_history_len = token_kv_history_len
        self.action_kv_history_len = action_kv_history_len
        self.prefix_token_lens = [0] * batch_size
        self.token_key_values = [(torch.full((batch_size, max_beam_size, self.head_num, 1 + max_input_len + token_kv_history_len, self.head_dim), 
                                              fill_value=INIT_VAL, device=device),
                                  torch.full((batch_size, max_beam_size, self.head_num, 1 + max_input_len + token_kv_history_len, self.head_dim), 
                                              fill_value=INIT_VAL, device=device)) for _ in range(self.token_layer_num)]
        self.action_key_values = [(torch.full((batch_size, max_beam_size, self.head_num, 1 + 2 * max_input_len + action_kv_history_len, self.head_dim), 
                                              fill_value=INIT_VAL, device=device),
                                   torch.full((batch_size, max_beam_size, self.head_num, 1 + 2 * max_input_len + action_kv_history_len, self.head_dim), 
                                              fill_value=INIT_VAL, device=device)) for _ in range(self.action_layer_num)]
        self.token_masks = torch.full((batch_size, max_beam_size, max_input_len + token_kv_history_len + 2), fill_value=False, dtype=torch.bool, device=device)
        self.token_masks[:, :, -1] = True
        self.action_masks = torch.full((batch_size, max_beam_size, 2 * max_input_len + action_kv_history_len + 2), fill_value=False, dtype=torch.bool, device=device)
        self.action_masks[:, :, -1] = True
        self.gpt_input_cache = torch.full((batch_size, max_beam_size, config.n_embd), fill_value=0.0, dtype=torch.float, device=device)

    def init_history_kv(self, action_key_values, token_key_values, seq_lens, group_ids):
        for layer_i, (k, v) in enumerate(self.action_key_values):
            k[:, 0, :, :self.action_kv_history_len, :] = action_key_values[layer_i][0]
            v[:, 0, :, :self.action_kv_history_len, :] = action_key_values[layer_i][1]

        for layer_i, (k, v) in enumerate(self.token_key_values):
            k[:, 0, :, :self.token_kv_history_len, :] = token_key_values[layer_i][0]
            v[:, 0, :, :self.token_kv_history_len, :] = token_key_values[layer_i][1]

        # set masks
        N = max(group_ids) + 1
        action_len_bucket = [1] * N
        token_len_bucket = [1] * N
        for sent_i, group_id in enumerate(group_ids):
            action_len_bucket[group_id] += 2 * seq_lens[sent_i] - 1
            token_len_bucket[group_id] += seq_lens[sent_i]
        self.prefix_token_lens = [l - 1 for l in token_len_bucket]
        for batch_i in range(N):
            self.action_masks[batch_i, 0, :action_len_bucket[batch_i]] = True
            self.token_masks[batch_i, 0, :token_len_bucket[batch_i]] = True
        return action_len_bucket, token_len_bucket
        

    def prepare_compositions(self, states_batch, beam_context):
        composition_indices = []
        for batch_i, states in enumerate(states_batch):
            for state in states:
                if state.current_action == ActionType.REDUCE and not state.is_finished:
                    l_top, r_top, ext_vocab_id = state.prev_state.top_states()

                    assert l_top is not None
                    assert r_top is not None
                    left_idx = l_top.current_step
                    right_idx = r_top.current_step

                    composition_indices.append((batch_i, state.beam_id, left_idx, right_idx))

        if len(composition_indices) > 0:
            composition_indices = torch.tensor(composition_indices, dtype=torch.long, device=self.device)
            left_repr = self.compose_cache[composition_indices[:, 0], composition_indices[:, 1], composition_indices[:, 2], :] # (?, dim)
            right_repr = self.compose_cache[composition_indices[:, 0], composition_indices[:, 1], composition_indices[:, 3], :] # (?, dim)
            return torch.stack([left_repr, right_repr], dim=1)  # (?, 2, dim)
        else:
            return None

    def update(self, states_batch: List[List[State]], sync_steps: List[int], reduce_repr, 
               last_action_kv, last_token_kv):
        beam_size = max(map(len, states_batch))
        org_beam_ids = []
        reduce_states_indices = []
        shift_states_indices = []
        next_token_ids = []
        token_kv_indices = []
        action_kv_indices = []
        token_kv_reorder_indices = []
        action_kv_reorder_indices = []
        for batch_i, states in enumerate(states_batch):
            beam_ids = []
            for new_beam_id, state in enumerate(states):
                beam_ids.append(state.beam_id)
                state.beam_id = new_beam_id
                if state.current_action == ActionType.REDUCE and not state.is_finished:
                    assert state.current_step > 2
                    reduce_states_indices.append((batch_i, new_beam_id, state.current_step))
                if not state.updated:
                    # ignore those already generated next tokens
                    state.updated = True
                    assert state.batch_idx != -1
                    if state.current_action == ActionType.SHIFT:
                        # only update token kv for shift states
                        token_kv_indices.append((batch_i, new_beam_id, 
                            state.token_offset - 1 + self.token_kv_history_len))
                        token_kv_reorder_indices.append(state.batch_idx)
                    action_kv_indices.append((batch_i, new_beam_id, 
                        state.current_step + self.action_kv_history_len))
                    action_kv_reorder_indices.append(state.batch_idx)
                    # assert state.stack_top.token_id != -1
                    if state.token_offset > sync_steps[batch_i]:
                        assert state.current_action == ActionType.SHIFT
                        shift_states_indices.append((batch_i, new_beam_id, state.current_step))
                        assert state.stack_top.token_id != -1
                        next_token_ids.append(state.stack_top.token_id)
            while len(beam_ids) < self.max_beam_size:
                beam_ids.append(self.max_beam_size - 1)

            org_beam_ids.append(beam_ids)

        if reduce_repr is not None:
            assert len(reduce_states_indices) == reduce_repr.shape[0]
        else:
            assert len(reduce_states_indices) == 0

        # reorder all caches
        org_beam_ids = torch.tensor(org_beam_ids, dtype=torch.long, device=self.device)
        action_kv_indices = torch.tensor(action_kv_indices, dtype=torch.long).to(self.device, non_blocking=True)
        assert org_beam_ids.shape[-1] == self.max_beam_size
        L = self.compose_cache.shape[2]
        self.compose_cache = self.compose_cache.gather(dim=1, index=org_beam_ids.unsqueeze(2).unsqueeze(3)\
            .repeat(1, 1, L, self.compose_dim))
        # (batch_size, max_beam_size, num_head, L, head_dim)
        L = self.token_key_values[0][0].shape[3]
        self.token_key_values = [(k.gather(dim=1, index=org_beam_ids.unsqueeze(2).unsqueeze(3).unsqueeze(4)\
                                    .repeat(1, 1, self.head_num, L, self.head_dim)),
                                  v.gather(dim=1, index=org_beam_ids.unsqueeze(2).unsqueeze(3).unsqueeze(4)\
                                    .repeat(1, 1, self.head_num, L, self.head_dim))) for k,v in self.token_key_values]
        L = self.action_key_values[0][0].shape[3]
        self.action_key_values = [(k.gather(dim=1, index=org_beam_ids.unsqueeze(2).unsqueeze(3).unsqueeze(4)\
                                    .repeat(1, 1, self.head_num, L, self.head_dim)),
                                  v.gather(dim=1, index=org_beam_ids.unsqueeze(2).unsqueeze(3).unsqueeze(4)\
                                    .repeat(1, 1, self.head_num, L, self.head_dim))) for k,v in self.action_key_values]
        L = self.token_masks.shape[-1]
        self.token_masks = self.token_masks.gather(dim=1, index=org_beam_ids.unsqueeze(-1).repeat(1, 1, L))
        L = self.action_masks.shape[-1]
        self.action_masks = self.action_masks.gather(dim=1, index=org_beam_ids.unsqueeze(-1).repeat(1, 1, L))
        D = self.gpt_input_cache.shape[-1]
        self.gpt_input_cache = self.gpt_input_cache.gather(dim=1, index=org_beam_ids.unsqueeze(-1).repeat(1, 1, D))

        # fillin compositional representations
        if len(reduce_states_indices) > 0:
            reduce_states_indices = torch.tensor(reduce_states_indices, device=self.device, dtype=torch.long)
            assert not torch.all(self.compose_cache[reduce_states_indices[:, 0], reduce_states_indices[:,1], reduce_states_indices[:,2] - 1, :] == 0)
            assert torch.all(self.compose_cache[reduce_states_indices[:, 0], reduce_states_indices[:,1], reduce_states_indices[:,2], :] == 0)
            self.compose_cache[reduce_states_indices[:, 0], reduce_states_indices[:,1], reduce_states_indices[:,2], :] = reduce_repr
            self.gpt_input_cache[reduce_states_indices[:, 0], reduce_states_indices[:, 1], :] = self.up_scale(reduce_repr)
        # fillin action_key_values

        assert action_kv_indices.shape[0] > 0
        if last_action_kv is not None:
            action_kv_reorder_indices = torch.tensor(action_kv_reorder_indices, dtype=torch.long, device=self.device)
            for layer_i, kv in enumerate(self.action_key_values):
                # if torch.all(action_kv_indices[:, 2] - 2 - self.action_kv_history_len >= 0):
                #     print(action_kv_indices[:, 2] - 2 - self.action_kv_history_len)
                #     assert torch.any(kv[0][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 2, :] != INIT_VAL)
                assert torch.all(kv[0][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 1, :] == INIT_VAL)
                # print(f'tgt shape {kv[0][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 1, :].shape}')
                # print(f'src shape {last_action_kv[layer_i][0].shape}')
                kv[0][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 1, :] = \
                    last_action_kv[layer_i][0].index_select(dim=0, index=action_kv_reorder_indices).squeeze(-2)
                kv[1][action_kv_indices[:, 0], action_kv_indices[:,1], :, action_kv_indices[:,2] - 1, :] = \
                    last_action_kv[layer_i][1].index_select(dim=0, index=action_kv_reorder_indices).squeeze(-2)
            # NOTE: check if reduce_state_indices[:,2] or reduce_state_indices[:,2]-1
            if torch.all(action_kv_indices[:, 2] - 1 >= 0):
                # assert torch.all(self.action_masks[action_kv_indices[:, 0], action_kv_indices[:,1], action_kv_indices[:,2] - 1])
                assert torch.all(~self.action_masks[action_kv_indices[:, 0], action_kv_indices[:,1], action_kv_indices[:,2] - 1])
        
            self.action_masks[action_kv_indices[:, 0], action_kv_indices[:,1], action_kv_indices[:,2] - 1] = True
        # print(self.action_masks[0, :5, :5])
        # print(self.action_key_values[0][0][0, :5, 0, :5, :5])

        # fillin token kv into shift ones
        
        next_token_ids = torch.tensor(next_token_ids, dtype=torch.long, device=self.device)
        token_kv_indices = torch.tensor(token_kv_indices, dtype=torch.long, device=self.device)
        token_repr = self.embeddings(next_token_ids)
        if len(shift_states_indices) > 0:
            shift_states_indices = torch.tensor(shift_states_indices, dtype=torch.long, device=self.device)

            self.compose_cache[shift_states_indices[:, 0], shift_states_indices[:, 1], shift_states_indices[:, 2], :] = self.down_scale(token_repr)
            self.gpt_input_cache[shift_states_indices[:, 0], shift_states_indices[:, 1], :] = token_repr
        if token_kv_indices.shape[0] > 0 and last_token_kv is not None:
            # print(f'token kv indices: {token_kv_indices}, {self.token_key_values[0][0][0, :, 0, :5, :5]}')
            token_kv_reorder_indices = torch.tensor(token_kv_reorder_indices, dtype=torch.long, device=self.device)
            for layer_i, kv in enumerate(self.token_key_values):
                assert torch.all(kv[0][token_kv_indices[:, 0], token_kv_indices[:,1], :, token_kv_indices[:,2], :] == INIT_VAL)
                kv[0][token_kv_indices[:, 0], token_kv_indices[:,1], :, token_kv_indices[:,2], :] = \
                    last_token_kv[layer_i][0].index_select(dim=0, index=token_kv_reorder_indices).squeeze(-2)
                kv[1][token_kv_indices[:, 0], token_kv_indices[:,1], :, token_kv_indices[:,2], :] = \
                    last_token_kv[layer_i][1].index_select(dim=0, index=token_kv_reorder_indices).squeeze(-2)
        
            self.token_masks[token_kv_indices[:, 0], token_kv_indices[:,1], token_kv_indices[:,2]] = True


    def prepare_gpt_input(self, states_batch: List[List[State]], sync_steps: List[int]):
        gpt_input_indices = []
        position_ids = []
        kv_indices = []
        for batch_i, states in enumerate(states_batch):
            for state_i, state in enumerate(states):
                # assert state.beam_id == state_i
                if state.token_offset == sync_steps[batch_i] and not state.is_finished:
                    gpt_input_indices.append((batch_i, state.beam_id))
                    position_ids.append(state.token_offset + self.prefix_token_lens[batch_i])
                    kv_indices.append((batch_i, state.beam_id))

        if len(gpt_input_indices) > 0:
            gpt_input_indices = torch.tensor(gpt_input_indices, dtype=torch.long, device=self.device)
            kv_indices = torch.tensor(kv_indices, dtype=torch.long, device=self.device)
            # gpt_inputs = self.composition_cache[gpt_input_indices[:, 0], gpt_input_indices[:, 1], gpt_input_indices[:, 2], :]
            # gpt_inputs = self.layer_norm(self.gpt_input_cache[gpt_input_indices[:,0], gpt_input_indices[:, 1]])
            gpt_inputs = self.gpt_input_cache[gpt_input_indices[:,0], gpt_input_indices[:, 1]]
            position_ids = torch.tensor(position_ids, dtype=torch.long, device=self.device)
            token_kv = [(k[kv_indices[:, 0], kv_indices[:, 1], :, :, :], v[kv_indices[:, 0], kv_indices[:, 1], :, :, :]) 
                        for k,v in self.token_key_values]  # (?, num_head, L, head_dim)
            action_kv = [(k[kv_indices[:, 0], kv_indices[:, 1], :, :, :], v[kv_indices[:, 0], kv_indices[:, 1], :, :, :])
                         for k,v in self.action_key_values]  # (?, num_head, L, head_dim)
            token_masks = self.token_masks[kv_indices[:, 0], kv_indices[:, 1], :]  #(?, L)
            action_masks = self.action_masks[kv_indices[:, 0], kv_indices[:, 1], :]  #(?, L)
            return gpt_inputs, position_ids, action_kv, token_kv, action_masks, token_masks
        else:
            return None, None, None, None, None, None