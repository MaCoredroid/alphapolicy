"""Composite price encoder: frozen Kronos-mini + trainable numeric MLP branch."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.kronos_mini_fullseq import KronosMiniPriceEncoderFullSeq


def _rolling_sum(x: torch.Tensor, window: int) -> torch.Tensor:
    """Left-padded rolling sum over time dimension."""
    if window <= 1:
        return x
    x = x.unsqueeze(1)  # [B, 1, L]
    weight = x.new_ones(1, 1, window)
    x_pad = F.pad(x, (window - 1, 0))
    y = F.conv1d(x_pad, weight)
    return y.squeeze(1)


def _rolling_std(x: torch.Tensor, window: int, eps: float = 1e-6) -> torch.Tensor:
    sum_x = _rolling_sum(x, window)
    sum_x2 = _rolling_sum(x * x, window)
    mean = sum_x / window
    mean2 = sum_x2 / window
    var = (mean2 - mean * mean).clamp_min(eps)
    return torch.sqrt(var)


def build_price_features(price_seq: torch.Tensor, seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute backward-looking numeric features from normalized OHLCV+return."""
    lr = price_seq[..., -1]
    open_p = price_seq[..., 0]
    high_p = price_seq[..., 1]
    low_p = price_seq[..., 2]
    close_p = price_seq[..., 3]
    volume_z = price_seq[..., 4]

    r1 = lr
    r5 = _rolling_sum(lr, 5)
    r21 = _rolling_sum(lr, 21)
    vol21 = _rolling_std(lr, 21)
    hl = (high_p - low_p) / (close_p.abs() + 1e-3)

    feats = torch.stack([r1, r5, r21, vol21, hl, volume_z], dim=-1)
    if seq_mask is not None:
        mask = seq_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mask = mask.to(price_seq.device)
        feats = feats * mask.unsqueeze(-1).to(feats.dtype)
    return feats


class PriceMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim * 2),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KronosMLPFusion(nn.Module):
    """Token-wise gate between Kronos and numeric MLP encodings."""

    def __init__(self, hidden_dim: int, gate_bias: float = -2.0) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.gate[-1].bias, gate_bias)
        nn.init.zeros_(self.gate[-1].weight)

    def forward(
        self,
        h_basic: torch.Tensor,
        h_kronos: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cat = torch.cat([h_basic, h_kronos], dim=-1)
        g = torch.sigmoid(self.gate(h_cat))
        mask = None
        if seq_mask is not None:
            mask = seq_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            mask = mask.to(g.device)
            g = g * mask.unsqueeze(-1).to(g.dtype)
        price_hidden = g * h_basic + (1.0 - g) * h_kronos
        if mask is not None:
            price_hidden = price_hidden * mask.unsqueeze(-1).to(price_hidden.dtype)
        return price_hidden, g


class PriceEncoderWithMLP(nn.Module):
    """Parallel trainable numeric encoder fused with frozen Kronos-mini tokens."""

    def __init__(
        self,
        kronos_encoder: KronosMiniPriceEncoderFullSeq,
        feat_dim: int,
        hidden_dim: int,
        freeze_kronos: bool = True,
        mlp_dropout: float = 0.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.kronos = kronos_encoder
        self.hidden_dim = hidden_dim
        if freeze_kronos:
            for param in self.kronos.parameters():
                param.requires_grad_(False)
        self.price_mlp = PriceMLP(in_dim=feat_dim, hidden_dim=hidden_dim, dropout=mlp_dropout)
        self.fusion = KronosMLPFusion(hidden_dim=hidden_dim, gate_bias=gate_bias)

    def forward(
        self,
        price_seq: torch.Tensor,
        *,
        asset: str,
        time_features: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if price_seq is None or price_seq.numel() == 0:
            device = next(self.parameters(), torch.zeros(1)).device
            batch = price_seq.size(0) if price_seq is not None else 1
            hidden = torch.zeros(batch, 0, self.hidden_dim, device=device)
            gate = torch.zeros(batch, 0, 1, device=device)
            return hidden, gate
        with torch.no_grad():
            kronos_out = self.kronos(price_seq, asset=asset, time_features=time_features)
            h_kronos = kronos_out.detach()
        feats = build_price_features(price_seq, seq_mask=seq_mask)
        feats = feats.to(self.price_mlp.net[0].weight.device)
        h_basic = self.price_mlp(feats)
        if h_basic.device != h_kronos.device:
            h_kronos = h_kronos.to(h_basic.device)
        price_hidden, gate = self.fusion(h_basic, h_kronos, seq_mask=seq_mask)
        return price_hidden, gate
