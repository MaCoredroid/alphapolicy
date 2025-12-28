"""Factor cross-attention block for Run100."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    """Single cross-attention + FFN block."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=max(1, int(n_heads)),
            dropout=dropout,
            batch_first=True,
            bias=True,
            kdim=d_model,
            vdim=d_model,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(queries)
        attn_out = self.attn(
            h,
            kv,
            kv,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = queries + attn_out
        x = x + self.ff(self.norm2(x))
        return x


@dataclass
class FactorCrossAttentionConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.1


class FactorCrossAttention(nn.Module):
    """Stacked cross-attention from target sequence to factor bank sequences."""

    def __init__(self, config: FactorCrossAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ff_mult=config.ff_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

    def forward(
        self,
        target_seq: torch.Tensor,
        factor_seq: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        factor_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # target_seq: [B, L_q, d], factor_seq: [B, L_kv, d]
        x = target_seq
        for layer in self.layers:
            x = layer(
                x,
                factor_seq,
                attn_mask=attn_mask,
                key_padding_mask=factor_padding_mask,
            )
        return x
