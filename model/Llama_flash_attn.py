# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple, Type
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_seq_len: int = 1024
    apply_norm: bool = True


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
#     t = torch.arange(end, device=freqs.device)  # type: ignore
#     freqs = torch.outer(t, freqs)  # type: ignore
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'{freqs_cis.shape},{(x.shape[1], x.shape[-1])}'
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    # xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    # freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    # xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # return xq_out.type_as(xq), xk_out.type_as(xk)

    cos, sin = freqs_cis
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        with torch.backends.cuda.sdp_kernel():
            if attn_mask is not None:
                if len(attn_mask.shape) == 2:
                    output = F.scaled_dot_product_attention(xq, xk, values, attn_mask=attn_mask.unsqueeze(1).unsqueeze(1))
                elif len(attn_mask.shape) == 3:
                    output = F.scaled_dot_product_attention(xq, xk, values, attn_mask=attn_mask.unsqueeze(1).to(xq.dtype))
                else:
                    raise Exception("Unsupported mask format")
            else:
                output = F.scaled_dot_product_attention(xq, xk, values, is_causal=True)

        # L, S = xq.size(-2), keys.size(-2)
        # scale_factor = 1 / math.sqrt(xq.size(-1))
        # attn_weight = xq @ keys.transpose(-2, -1) * scale_factor
        # attn_weight += attn_mask.unsqueeze(1).unsqueeze(1)
        # attn_weight = torch.softmax(attn_weight, dim=-1)
        # output = attn_weight @ values

        output = output.transpose(1, 2).reshape(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


def convert_linear_to_bnb(float_linear):
    new_layer = InferenceQuantizedLinear(
        float_linear.in_features,
        float_linear.out_features,
        bias=float_linear.bias is not None,
    )
    new_layer._parameters["weight"] = bnb.nn.Int8Params(
        float_linear.weight.data.cpu(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    if float_linear.bias is not None:
        new_layer._parameters["bias"] = float_linear.bias
    return new_layer


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        if params.apply_norm:
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        else:
            self.norm = lambda x: x

        # self._freqs_cis = None
        self._init_sin_cos()
    
    def _init_sin_cos(self, base=10000.0):
        device=next(self.parameters()).device
        dim = self.params.dim // self.params.n_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.params.max_seq_len, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype=torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        
    def freqs_cis(self, position_ids, h):
        # if self._freqs_cis is None:
            # self._freqs_cis = precompute_freqs_cis(
            #     self.params.dim // self.params.n_heads, self.params.max_seq_len
            # )
            # self._freqs_cis = self._freqs_cis.to(next(self.parameters()).device)
           
        # return self._freqs_cis

        return (
            self.cos_cached[position_ids].to(dtype=h.dtype),
            self.sin_cached[position_ids].to(dtype=h.dtype),
        )

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor, position_ids: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        if len(position_ids.shape) == 1:
            position_ids = position_ids.unsqueeze(0).repeat(_bsz, 1)
        freqs_cis = self.freqs_cis(position_ids, h)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask=attn_mask)

        h = self.norm(h)
        # output = self.output(h)
        return h