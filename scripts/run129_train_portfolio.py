#!/usr/bin/env python3
"""
Non-RL differentiable portfolio trainer (run129 design).

Uses a cross-sectional transformer policy to map cached alpha signals to weights.
Objective: maximize Sharpe (mean / std of daily PnL) with turnover cost; all
operations are differentiable end-to-end (no REINFORCE).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")


def load_signals(npz_path: Path) -> Dict:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "mu": data["mu"],  # [N, L]
        "target": data["target"],
        "ticker_mask": data["ticker_mask"],
        "split_mask": data["split_mask"],
        "dates": data["dates"].tolist(),
        "ticker_order": data["ticker_order"].tolist(),
    }


def precompute_static_features(
    mu: np.ndarray,
    ret: np.ndarray,
    live: np.ndarray,
    lookbacks: Tuple[int, ...] = (1, 5, 20),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-ticker static features (except held_prev) and global stats."""
    T, N = mu.shape
    rank = np.argsort(np.argsort(-mu, axis=1), axis=1).astype(np.float32)
    if N > 1:
        rank /= float(N - 1)
    else:
        rank[:] = 0.0

    roll_feats = []
    for lb in lookbacks:
        rf = np.zeros_like(ret, dtype=np.float32)
        for t in range(T):
            if t - lb + 1 >= 0:
                rf[t] = ret[t - lb + 1 : t + 1].sum(axis=0)
            else:
                rf[t] = 0.0
        roll_feats.append(rf.astype(np.float32))
    ret_1d, ret_5d, ret_20d = roll_feats

    X_static = np.stack(
        [
            mu.astype(np.float32),
            rank,
            ret_1d,
            ret_5d,
            ret_20d,
            live.astype(np.float32),
        ],
        axis=-1,
    )  # (T, N, 6)

    g = np.zeros((T, 5), dtype=np.float32)
    for t in range(T):
        live_t = live[t]
        mu_live = mu[t][live_t]
        ret_live = ret_1d[t][live_t]
        if mu_live.size > 0:
            mu_mean = mu_live.mean()
            mu_std = mu_live.std()
            mu_max = mu_live.max()
            mu_min = mu_live.min()
            ret_mean = ret_live.mean()
        else:
            mu_mean = mu_std = mu_max = mu_min = ret_mean = 0.0
        g[t] = np.array([mu_mean, mu_std, mu_max, mu_min, ret_mean], dtype=np.float32)
    return X_static, g


def to_device(X: np.ndarray, g: np.ndarray, ret: np.ndarray, live: np.ndarray, device: torch.device):
    X_t = torch.from_numpy(X).to(device)  # (T, N, 6)
    g_t = torch.from_numpy(g).to(device)  # (T, 5)
    ret_t = torch.from_numpy(ret).to(device)  # (T, N)
    live_t = torch.from_numpy(live.astype(bool)).to(device)  # (T, N)
    return X_t, g_t, ret_t, live_t


class CrossSectionPolicy(nn.Module):
    def __init__(self, feat_dim: int, global_dim: int, hidden: int = 64, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Linear(feat_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden * 2, 1)  # per-ticker score

    def forward(self, x: torch.Tensor, g: torch.Tensor, live_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], g: [B, G], live_mask: [B, N]
        # If no live names in a batch element, return large negative scores to avoid transformer nesting issues.
        if (live_mask.sum(dim=1) == 0).any():
            return torch.full(live_mask.shape, -1e9, device=x.device)

        h = self.embed(x)
        key_padding_mask = ~live_mask  # True for padding
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        g_emb = self.global_proj(g)  # [B, H]
        g_rep = g_emb.unsqueeze(1).expand(-1, h.size(1), -1)
        scores = self.head(torch.cat([h, g_rep], dim=-1)).squeeze(-1)  # [B, N]
        scores = scores.masked_fill(~live_mask, -1e9)
        return scores


def scores_to_weights(
    scores: torch.Tensor,
    live_mask: torch.Tensor,
    temperature: float,
    w_max: float | None,
    gross: float,
    k_cap: int | None,
) -> torch.Tensor:
    """Convert scores to weights; optionally cap active names via top-k."""
    logits = scores / max(temperature, 1e-6)
    logits = logits.masked_fill(~live_mask, -1e9)

    if k_cap is not None and k_cap > 0:
        # Keep only top-k scores, zero elsewhere after softmax.
        topk_vals, topk_idx = torch.topk(logits, k=min(k_cap, logits.size(-1)), dim=-1)
        mask_top = torch.zeros_like(logits, dtype=torch.bool)
        mask_top.scatter_(dim=-1, index=topk_idx, value=True)
        logits = logits.masked_fill(~mask_top, -1e9)
    else:
        mask_top = live_mask

    w = torch.softmax(logits, dim=-1)  # [B, N], sums to 1 over live names
    # Zero out anything outside the chosen top-k and renormalize.
    w = w * mask_top.float()
    total = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    w = w / total * gross
    w = w * gross
    if w_max is not None:
        w = torch.minimum(w, torch.full_like(w, w_max))
        total = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        w = w / total * gross
    return w


def episode_objective(
    policy: nn.Module,
    X_static: torch.Tensor,
    g: torch.Tensor,
    live: torch.Tensor,
    ret: torch.Tensor,
    cost: float,
    temperature: float,
    w_max: float | None,
    gross: float,
    lambda_turnover: float,
    lambda_effk: float,
    k_cap: int | None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Unroll a full episode deterministically and return loss + metrics."""
    T, N, _ = X_static.shape
    w_prev = torch.zeros(N, device=X_static.device)
    pnls = []
    turnovers = []
    eff_ks = []

    for t in range(T - 1):
        base_t = X_static[t]  # (N, 6)
        live_t = live[t]  # (N,)
        held_prev = (w_prev > 0).float().unsqueeze(1)
        x_t = torch.empty(N, 7, device=X_static.device)
        x_t[:, 0:5] = base_t[:, 0:5]
        x_t[:, 5] = held_prev.squeeze(1)
        x_t[:, 6] = base_t[:, 5]

        x = x_t.unsqueeze(0)
        g_t = g[t].unsqueeze(0)
        live_mask = live_t.unsqueeze(0)

        scores = policy(x, g_t, live_mask)  # [1, N]
        w = scores_to_weights(scores, live_mask, temperature, w_max, gross, k_cap)[0]  # (N,)

        turnover = 0.5 * (w - w_prev).abs().sum()
        pnl = (w_prev * ret[t + 1]).sum() - cost * turnover

        pnls.append(pnl)
        turnovers.append(turnover)
        eff_ks.append((w > 1e-4).sum())

        w_prev = w

    pnls_t = torch.stack(pnls)
    mean = pnls_t.mean()
    vol = pnls_t.std(unbiased=False) + 1e-8
    sharpe = mean / vol * math.sqrt(252.0)
    turnover_mean = torch.stack(turnovers).mean()
    eff_k_mean = torch.stack(eff_ks).float().mean()

    penalty_effk = lambda_effk * F.relu(eff_k_mean - (k_cap if k_cap is not None else eff_k_mean))
    loss = -sharpe + lambda_turnover * turnover_mean + penalty_effk
    metrics = {
        "mean": mean,
        "vol": vol,
        "sharpe": sharpe,
        "turnover": turnover_mean,
        "eff_k": eff_k_mean,
    }
    return loss, metrics


def metrics_to_float(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in metrics.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--val_npz", required=True)
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", default="logs/run129_portfolio")
    ap.add_argument("--ckpt_dir", default="outputs/run129_portfolio")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.3, help="Softmax temperature; lower -> spikier weights")
    ap.add_argument("--w_max", type=float, default=0.05, help="Max single-name weight; set None for unconstrained")
    ap.add_argument("--gross", type=float, default=1.0, help="Gross exposure (long-only)")
    ap.add_argument("--lambda_turnover", type=float, default=0.0, help="Penalty on average turnover (per day)")
    ap.add_argument("--k_cap", type=int, default=10, help="Hard cap on active names via top-k (per day)")
    ap.add_argument("--lambda_effk", type=float, default=0.05, help="Penalty on effK above k_cap")
    args = ap.parse_args()

    device = torch.device(args.device)
    cost = args.cost_bps * 1e-4
    log_dir = Path(args.save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    train = load_signals(Path(args.train_npz))
    val = load_signals(Path(args.val_npz))
    test = load_signals(Path(args.test_npz))

    def prep(split):
        mu = np.nan_to_num(split["mu"], nan=0.0).T.astype(np.float32)
        ret = np.nan_to_num(split["target"], nan=0.0).T.astype(np.float32)
        live = split["ticker_mask"].T.astype(bool) & split["split_mask"].T.astype(bool)
        return mu, ret, live

    mu_tr, ret_tr, mask_tr = prep(train)
    mu_va, ret_va, mask_va = prep(val)
    mu_te, ret_te, mask_te = prep(test)

    X_tr, g_tr = precompute_static_features(mu_tr, ret_tr, mask_tr)
    X_va, g_va = precompute_static_features(mu_va, ret_va, mask_va)
    X_te, g_te = precompute_static_features(mu_te, ret_te, mask_te)

    X_tr_t, g_tr_t, ret_tr_t, live_tr_t = to_device(X_tr, g_tr, ret_tr, mask_tr, device)
    X_va_t, g_va_t, ret_va_t, live_va_t = to_device(X_va, g_va, ret_va, mask_va, device)
    X_te_t, g_te_t, ret_te_t, live_te_t = to_device(X_te, g_te, ret_te, mask_te, device)

    feat_dim = 7
    global_dim = 5
    policy = CrossSectionPolicy(feat_dim, global_dim, hidden=args.hidden, layers=args.layers, heads=args.heads).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for ep in range(args.epochs):
        policy.train()
        loss, train_metrics = episode_objective(
            policy,
            X_tr_t,
            g_tr_t,
            live_tr_t,
            ret_tr_t,
            cost,
            args.temperature,
            args.w_max,
            args.gross,
            args.lambda_turnover,
            args.lambda_effk,
            args.k_cap,
        )
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()

        policy.eval()
        with torch.no_grad():
            _, val_metrics = episode_objective(
                policy,
                X_va_t,
                g_va_t,
                live_va_t,
                ret_va_t,
                cost,
                args.temperature,
                args.w_max,
                args.gross,
                args.lambda_turnover,
                args.lambda_effk,
                args.k_cap,
            )
            _, test_metrics = episode_objective(
                policy,
                X_te_t,
                g_te_t,
                live_te_t,
                ret_te_t,
                cost,
                args.temperature,
                args.w_max,
                args.gross,
                args.lambda_turnover,
                args.lambda_effk,
                args.k_cap,
            )

        tm = metrics_to_float(train_metrics)
        vm = metrics_to_float(val_metrics)
        tem = metrics_to_float(test_metrics)
        log_line = (
            f"[epoch {ep+1}] "
            f"train Sharpe={tm['sharpe']:.3f} mean={tm['mean']:.6f} vol={tm['vol']:.6f} "
            f"turnover={tm['turnover']:.4f} effK={tm['eff_k']:.2f} | "
            f"val Sharpe={vm['sharpe']:.3f} mean={vm['mean']:.6f} vol={vm['vol']:.6f} "
            f"turnover={vm['turnover']:.4f} effK={vm['eff_k']:.2f} | "
            f"test Sharpe={tem['sharpe']:.3f} mean={tem['mean']:.6f} vol={tem['vol']:.6f} "
            f"turnover={tem['turnover']:.4f} effK={tem['eff_k']:.2f}"
        )
        print(log_line, flush=True)
        log_path.open("a").write(log_line + "\n")

        torch.save(
            {"policy": policy.state_dict(), "epoch": ep + 1, "args": vars(args)},
            ckpt_dir / f"epoch_{ep+1}.pt",
        )


if __name__ == "__main__":
    main()
