"""Causal GPT-style temporal encoder for Run100 full-sequence modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def _build_causal_mask(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.split(x.size(-1) // 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to q and k.
    q, k: [B, L, H, Dh]; positions: [L]
    """
    dim = q.size(-1)
    device = q.device
    dtype = q.dtype
    freqs = torch.arange(0, dim, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (freqs / dim))
    # angles: [L, Dh/2]
    theta = torch.einsum("l,d->ld", positions.to(dtype), inv_freq)
    sin, cos = theta.sin(), theta.cos()
    # reshape for broadcast: [1, L, 1, Dh/2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    def _rope(x):
        x1, x2 = x.split(dim // 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return _rope(q), _rope(k)


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: float = 8.0 / 3.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(hidden_mult * d_model)
        self.w_u = nn.Linear(d_model, hidden)
        self.w_v = nn.Linear(d_model, hidden)
        self.out = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.silu(self.w_u(x))
        v = self.w_v(x)
        return self.dropout(self.out(u * v))


class RoPEAttention(nn.Module):
    """Multi-head attention with optional RoPE on Q/K."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rope: bool = True, rope_base: float = 10000.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = max(1, int(n_heads))
        assert d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // self.n_heads
        self.rope = rope
        self.rope_base = rope_base
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)
        if self.rope and positions is not None:
            q, k = apply_rope(q, k, positions, base=self.rope_base)
        # attention
        q = q.transpose(1, 2)  # [B, H, L, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, L]
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_out = torch.matmul(attn_probs, v)  # [B, H, L, Dh]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(attn_out)


class PreRMSNormTransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer block with residual scaling and SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: float = 8.0 / 3.0,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
        rope: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attn = RoPEAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, rope=rope, rope_base=rope_base)
        self.ff = SwiGLU(d_model=d_model, hidden_mult=ff_mult, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.res_scale = residual_scale

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, d]
        h = self.norm1(x)
        attn_out = self.attn(h, attn_mask=attn_mask)
        x = x + self.res_scale * attn_out
        h = self.norm2(x)
        x = x + self.res_scale * self.ff(h)
        return x


@dataclass
class TimeLLMConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    ff_mult: float = 8.0 / 3.0  # ~ (2/3) * 4d for SwiGLU
    dropout: float = 0.1
    rope: bool = False  # RoPE disabled by default (use absolute content-only)
    rope_base: float = 10000.0
    residual_scale: Optional[float] = None  # if None, defaults to 1/sqrt(n_layers)


class TimeLLM(nn.Module):
    """Causal GPT-style encoder shared across assets and factor bank."""

    def __init__(self, config: TimeLLMConfig) -> None:
        super().__init__()
        self.config = config
        res_scale = config.residual_scale
        if res_scale is None:
            res_scale = 1.0 / math.sqrt(max(1, config.n_layers))
        self.layers = nn.ModuleList(
            [
                PreRMSNormTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ff_mult=config.ff_mult,
                    dropout=config.dropout,
                    residual_scale=res_scale,
                    rope=config.rope,
                    rope_base=config.rope_base,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B, L, d]
        L = x.size(1)
        if attn_mask is None:
            attn_mask = _build_causal_mask(L, x.device, x.dtype)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 1:
                key_padding_mask = key_padding_mask.unsqueeze(0)
            pad_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            attn_mask = attn_mask + pad_mask.to(dtype=attn_mask.dtype) * torch.finfo(x.dtype).min
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.final_norm(x)
