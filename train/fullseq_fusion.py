"""Fusion utilities for Run100 full-sequence model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DailyNewsPooler(nn.Module):
    """Price-conditioned pooling of per-day news embeddings."""

    def __init__(
        self,
        doc_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        intensity_dim: int = 4,
    ) -> None:
        super().__init__()
        self.doc_proj = nn.Linear(doc_dim, hidden_dim)
        self.intensity_proj = nn.Linear(intensity_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=max(1, int(num_heads)),
            dropout=0.0,
            batch_first=True,
            bias=True,
            kdim=hidden_dim,
            vdim=hidden_dim,
        )
        self.no_news = nn.Parameter(torch.zeros(hidden_dim))
        gate_hid = max(4, hidden_dim // 4)
        self.missing_gate = nn.Sequential(
            nn.Linear(intensity_dim, gate_hid),
            nn.GELU(),
            nn.Linear(gate_hid, 1),
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        news_docs: torch.Tensor,
        news_mask: torch.Tensor,
        intensity: torch.Tensor,
    ) -> torch.Tensor:
        # price_seq: [B, L, d], news_docs: [B, L, K, doc_dim], news_mask: [B, L, K] bool, intensity: [B, L, I]
        B, L, K, _ = news_docs.shape
        device = price_seq.device
        if K == 0:
            base = self.no_news.view(1, 1, -1).to(device).expand(B, L, -1)
            intensity_vec = self.intensity_proj(intensity)
            gate = torch.sigmoid(self.missing_gate(intensity))  # [B, L, 1]
            pooled = gate * base + (1 - gate) * base
            return pooled + intensity_vec
        proj_docs = self.doc_proj(news_docs).reshape(B * L, K, -1)
        mask_flat = news_mask.reshape(B * L, K)
        queries = price_seq.reshape(B * L, 1, -1)
        attn_out = torch.zeros(B * L, 1, price_seq.size(-1), device=device)
        valid = mask_flat.any(dim=1)
        if valid.any():
            attn = self.attn(
                queries[valid],
                proj_docs[valid],
                proj_docs[valid],
                key_padding_mask=~mask_flat[valid],
                need_weights=False,
            )[0]
            if attn_out.dtype != attn.dtype:
                attn_out = attn_out.to(attn.dtype)
            attn_out[valid] = attn
        attn_out = attn_out.view(B, L, -1)
        has_news = news_mask.any(dim=-1, keepdim=True)
        attn_out = torch.where(
            has_news,
            attn_out,
            self.no_news.view(1, 1, -1).to(device),
        )
        intensity_vec = self.intensity_proj(intensity)
        gate = torch.sigmoid(self.missing_gate(intensity))  # [B, L, 1]
        fallback = self.no_news.view(1, 1, -1).to(device)
        pooled = gate * attn_out + (1 - gate) * fallback
        return pooled + intensity_vec


class GatedPriceNewsFusion(nn.Module):
    """Per-day token constructor: fuse price + news + optional aux."""

    def __init__(self, hidden_dim: int, aux_dim: int = 0) -> None:
        super().__init__()
        in_dim = hidden_dim * 2 + aux_dim
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        price_seq: torch.Tensor,
        news_seq: torch.Tensor,
        aux_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # All inputs: [B, L, dim]
        parts = [price_seq, news_seq]
        if aux_seq is not None:
            parts.append(aux_seq)
        fused_in = torch.cat(parts, dim=-1)
        gate = self.gate(fused_in)
        mix = self.mix(fused_in)
        out = price_seq + gate * mix
        return self.norm(out)
