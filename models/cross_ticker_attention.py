"""Cross-ticker attention block for Run101 (attend across tickers at each time step)."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class CrossTickerConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.1


class CrossTickerAttention(nn.Module):
    """Transformer encoder applied across tickers per time step.

    Input shape: [B, T, N, d]
    Output shape: same
    """

    def __init__(self, cfg: CrossTickerConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

    def forward(self, x: torch.Tensor, ticker_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, N, d], ticker_mask: [N, T] or [B, T, N] with True indicating a valid ticker/time step.
        B, T, N, d = x.shape
        # Flatten batch+time so encoder attends across tickers only; keep batch_first.
        x_enc = x.reshape(B * T, N, d)  # [B*T, N, d]
        pad_mask = None
        if ticker_mask is not None:
            tm = ticker_mask.to(dtype=torch.bool)
            # Normalize mask shape to [B, T, N]; incoming data uses [N, T].
            if tm.dim() == 2:
                if tm.shape == (N, T):
                    tm = tm.transpose(0, 1)  # [T, N]
                elif tm.shape != (T, N):
                    raise ValueError(f"ticker_mask dim=2 but shape {tm.shape} does not match (T,N) or (N,T)")
                tm = tm.unsqueeze(0)  # [1, T, N]
            elif tm.dim() == 3:
                if tm.shape == (B, N, T):
                    tm = tm.permute(0, 2, 1)  # [B, T, N]
                elif tm.shape != (B, T, N):
                    raise ValueError(f"ticker_mask dim=3 but shape {tm.shape} does not match (B,T,N) or (B,N,T)")
            else:
                raise ValueError(f"ticker_mask must have 2 or 3 dims, got {tm.dim()}")
            if tm.shape[0] != B:
                if tm.shape[0] == 1:
                    tm = tm.expand(B, -1, -1)
                else:
                    raise ValueError(f"ticker_mask batch dim {tm.shape[0]} does not match input batch {B}")
            # Transformer expects True where positions are padding/ignored; ticker_mask is True for valid tokens.
            pad_mask = (~tm).reshape(B * T, N)
        out = self.encoder(x_enc, src_key_padding_mask=pad_mask)
        return out.view(B, T, N, d)
