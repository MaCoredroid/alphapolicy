"""Alpha policy/backtest on top of a frozen Run100 model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import csv
import sys

from train.run101_multi_dataset import Run101MultiTickerDataset
from train.run100_model import Run100FullSeqModel


def _bin_centers(edges: torch.Tensor) -> torch.Tensor:
    return 0.5 * (edges[:-1] + edges[1:])


def _to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    return obj


def policy_linear_neutral(alpha: np.ndarray, gross: float, z_cap: float, w_max: float | None) -> np.ndarray:
    if alpha.size == 0:
        return np.zeros_like(alpha)
    z = (alpha - alpha.mean()) / (alpha.std() + 1e-8)
    if z_cap > 0:
        z = np.clip(z, -z_cap, z_cap)
    # dollar-neutral: de-mean and scale to gross target
    z = z - z.mean()
    gross_raw = np.sum(np.abs(z)) + 1e-8
    w = gross * z / gross_raw
    if w_max is not None and w_max > 0:
        w = np.clip(w, -w_max, w_max)
        gross_raw = np.sum(np.abs(w)) + 1e-8
        w = gross * w / gross_raw
    return w


def policy_topk(alpha: np.ndarray, k: int, long_short: bool = False) -> np.ndarray:
    n = alpha.size
    if n == 0 or k <= 0:
        return np.zeros_like(alpha)
    k = min(k, n)
    idx = np.argsort(alpha)
    w = np.zeros_like(alpha)
    if long_short:
        bottom = idx[:k]
        top = idx[-k:]
        w[top] = 0.5 / k
        w[bottom] = -0.5 / k
    else:
        top = idx[-k:]
        w[top] = 1.0 / k
    return w


def compute_mu(
    model: Run100FullSeqModel,
    batch: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[torch.Tensor]]:
    """Return mu [N,L], target returns [N,L], ticker order, bin edges."""
    batch = _to_device(batch, device)
    with torch.no_grad():
        out = model(batch)
        probs = out["probs"]  # [N,L,bins]
    ticker_order = batch.get("ticker_order", [])
    bin_edges_list = batch.get("bin_edges_stack", [])
    target_idx = batch.get("target_indices")
    if target_idx is not None and len(ticker_order) == len(bin_edges_list):
        # align to target tickers only (model output is already target-only)
        ticker_order = [ticker_order[i] for i in target_idx.tolist()]
        bin_edges_list = [bin_edges_list[i] for i in target_idx.tolist()]
    centers = []
    for be in bin_edges_list:
        tensor = be if torch.is_tensor(be) else torch.tensor(be, device=probs.device)
        centers.append(_bin_centers(tensor.to(probs.device)))
    mu_list = []
    for i, c in enumerate(centers):
        p = probs[i]  # [L,bins]
        mu_list.append((p * c.unsqueeze(0)).sum(dim=-1))
    mu = torch.stack(mu_list, dim=0)  # [N,L]
    target = batch.get("target_log_return_stack")
    if target is None:
        raise ValueError("target_log_return_stack missing from batch.")
    return mu, target, ticker_order, bin_edges_list


def backtest(
    mu: torch.Tensor,
    target: torch.Tensor,
    ticker_mask: torch.Tensor,
    split_mask: torch.Tensor,
    policy: str,
    gross: float,
    z_cap: float,
    w_max: float | None,
    top_k: int,
    top_pct: float,
    cost_bps: float,
    leverage: float,
    margin_rate: float,
    dates: List[str] | None = None,
    ticker_order: List[str] | None = None,
    log_trades_csv: Path | None = None,
    log_daily_csv: Path | None = None,
) -> Dict[str, float]:
    device = mu.device
    N, L = mu.shape
    mu_np = mu.cpu().numpy()
    target_np = np.nan_to_num(target.cpu().numpy(), nan=0.0)
    ticker_mask_np = ticker_mask.cpu().numpy().astype(bool)
    split_mask_np = split_mask.cpu().numpy().astype(bool)
    weights_prev = np.zeros(N, dtype=np.float32)
    daily_pnl: List[float] = []
    daily_turnover: List[float] = []
    daily_log = []
    writer = None
    csv_file = None
    if log_trades_csv is not None:
        log_trades_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = log_trades_csv.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "day_idx",
                "date",
                "ticker",
                "action",
                "w_prev",
                "w_new",
                "delta_w",
                "return",
                "pnl_gross",
                "cost_alloc",
                "pnl_net",
            ]
        )
    for t in range(L):
        live = ticker_mask_np[:, t] & split_mask_np[:, t]
        if not live.any():
            continue
        alpha_t = mu_np[live, t]
        if policy == "linear":
            w_live = policy_linear_neutral(alpha_t, gross=gross, z_cap=z_cap, w_max=w_max)
        elif policy == "topk":
            k = top_k if top_k > 0 else max(1, int(top_pct * live.sum()))
            w_live = policy_topk(alpha_t, k=k, long_short=False)
        elif policy == "longshort":
            k = top_k if top_k > 0 else max(1, int(top_pct * live.sum()))
            w_live = policy_topk(alpha_t, k=k, long_short=True)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        w_full = np.zeros(N, dtype=np.float32)
        w_full[live] = w_live
        if leverage != 1.0:
            w_full = w_full * leverage
        # turnover
        turnover = 0.5 * np.sum(np.abs(w_full - weights_prev))
        # pnl from t to t+1 using target[t]
        borrow = max(0.0, float(np.sum(np.abs(weights_prev)) - 1.0))
        borrow_cost = borrow * (margin_rate / 252.0)
        pnl = float(np.sum(weights_prev * target_np[:, t]) - (cost_bps * 1e-4) * turnover - borrow_cost)
        daily_pnl.append(pnl)
        daily_turnover.append(turnover)
        if log_daily_csv is not None:
            date_val = dates[t] if dates and t < len(dates) else ""
            daily_log.append(
                {
                    "day_idx": t,
                    "date": date_val,
                    "pnl": pnl,
                    "turnover": turnover,
                }
            )
        if writer is not None and turnover > 0:
            date_val = dates[t] if dates and t < len(dates) else ""
            delta = w_full - weights_prev
            cost_total = (cost_bps * 1e-4) * turnover
            abs_delta = np.abs(delta)
            alloc = abs_delta / (abs_delta.sum() + 1e-12)
            for idx, dw in enumerate(delta):
                if dw == 0 and weights_prev[idx] == 0 and w_full[idx] == 0:
                    continue
                cost_alloc = cost_total * alloc[idx]
                pnl_gross = weights_prev[idx] * target_np[idx, t]
                pnl_net = pnl_gross - cost_alloc
                action = "buy" if dw > 0 else "sell" if dw < 0 else "hold"
                name = ticker_order[idx] if ticker_order and idx < len(ticker_order) else str(idx)
                writer.writerow(
                    [
                        t,
                        date_val,
                        name,
                        action,
                        round(float(weights_prev[idx]), 6),
                        round(float(w_full[idx]), 6),
                        round(float(dw), 6),
                        round(float(target_np[idx, t]), 6),
                        round(float(pnl_gross), 8),
                        round(float(cost_alloc), 8),
                        round(float(pnl_net), 8),
                    ]
                )
        weights_prev = w_full
    if csv_file is not None:
        csv_file.close()
    if log_daily_csv is not None and daily_log:
        log_daily_csv.parent.mkdir(parents=True, exist_ok=True)
        with log_daily_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["day_idx", "date", "pnl", "turnover"])
            for row in daily_log:
                writer.writerow([row["day_idx"], row["date"], row["pnl"], row["turnover"]])
    if not daily_pnl:
        return {"mean": float("nan"), "vol": float("nan"), "sharpe": float("nan"), "turnover": float("nan")}
    pnl_arr = np.array(daily_pnl)
    mean = pnl_arr.mean()
    vol = pnl_arr.std()
    sharpe = mean / (vol + 1e-12) * math.sqrt(252.0)
    return {
        "mean": mean,
        "vol": vol,
        "sharpe": sharpe,
        "turnover": float(np.mean(daily_turnover)),
    }


def trading_preview(
    mu: torch.Tensor,
    ticker_mask: torch.Tensor,
    ticker_order: List[str],
    policy: str,
    gross: float,
    z_cap: float,
    w_max: float | None,
    top_k: int,
    top_pct: float,
) -> List[Tuple[str, float, float]]:
    mu_np = mu.cpu().numpy()
    mask_np = ticker_mask.cpu().numpy().astype(bool)
    # last day with any live tickers
    valid_days = np.where(mask_np.any(axis=0))[0]
    if len(valid_days) == 0:
        return []
    t = int(valid_days[-1])
    live = mask_np[:, t]
    alpha_t = mu_np[live, t]
    names = np.array(ticker_order)[live]
    if policy == "linear":
        w_live = policy_linear_neutral(alpha_t, gross=gross, z_cap=z_cap, w_max=w_max)
    elif policy == "topk":
        k = top_k if top_k > 0 else max(1, int(top_pct * live.sum()))
        w_live = policy_topk(alpha_t, k=k, long_short=False)
    else:
        k = top_k if top_k > 0 else max(1, int(top_pct * live.sum()))
        w_live = policy_topk(alpha_t, k=k, long_short=True)
    return sorted(zip(names, alpha_t.tolist(), w_live.tolist()), key=lambda x: x[2], reverse=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha policy on frozen model.")
    p.add_argument("--manifest", required=True, help="Path to run100 manifest (fullseq)")
    p.add_argument("--ckpt", required=True, help="Frozen model checkpoint")
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Data split to use")
    p.add_argument("--policy", default="linear", choices=["linear", "topk", "longshort"], help="Policy type")
    p.add_argument("--gross", type=float, default=1.0, help="Gross exposure target for linear policy")
    p.add_argument("--z_cap", type=float, default=3.0, help="Z-score cap for linear policy")
    p.add_argument("--w_max", type=float, default=0.05, help="Per-name cap for linear policy (<=0 to disable)")
    p.add_argument("--top_k", type=int, default=50, help="Top-K for topk/longshort policies (ignored if 0)")
    p.add_argument("--top_pct", type=float, default=0.1, help="Top percentile for topk/longshort if top_k=0")
    p.add_argument("--cost_bps", type=float, default=5.0, help="Round-trip cost per unit turnover (bps)")
    p.add_argument("--leverage", type=float, default=1.0, help="Portfolio leverage multiplier (gross scaling)")
    p.add_argument("--margin_rate", type=float, default=0.0, help="Annualized margin rate for borrowed notional (e.g., 0.1 = 10%)")
    p.add_argument("--mode", default="backtest", choices=["backtest", "trade"], help="Backtest vs. trade preview")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_trades_csv", default="", help="Optional path to write per-day trade log CSV")
    p.add_argument("--log_daily_csv", default="", help="Optional path to write per-day aggregate PnL/turnover CSV")
    p.add_argument("--signals_npz", default="", help="Optional precomputed signals file (.npz) to skip model forward")
    p.add_argument(
        "--exclude_tickers",
        default="VIX",
        help="Comma-separated tickers to exclude from trading/universe (case-insensitive).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    w_max = None if args.w_max <= 0 else args.w_max
    log_trades_csv = Path(args.log_trades_csv) if args.log_trades_csv else None
    log_daily_csv = Path(args.log_daily_csv) if args.log_daily_csv else None
    leverage = float(args.leverage)
    margin_rate = float(args.margin_rate)
    use_cached = bool(args.signals_npz)
    exclude = {t.strip().upper() for t in args.exclude_tickers.split(",") if t.strip()}
    if use_cached:
        data = np.load(args.signals_npz, allow_pickle=True)
        mu = torch.from_numpy(data["mu"])
        target = torch.from_numpy(data["target"])
        ticker_mask = torch.from_numpy(data["ticker_mask"])
        split_mask = torch.from_numpy(data["split_mask"])
        ticker_order = data["ticker_order"].tolist()
        dates = data["dates"].tolist()
    else:
        dataset = Run101MultiTickerDataset(
            Path(args.manifest),
            target_assets=None,
            context_assets=["VIX"],
            factor_assets=[f.strip().upper() for f in "SPY,BTC,VIX,QQQ".split(",")],
            split=args.split,
        )
        dates: List[str] | None = None
        if dataset.ticker_order:
            first = dataset.ticker_order[0]
            rec = dataset.records.get(first)
            if rec and "dates" in rec:
                dates = rec["dates"]
        model = Run100FullSeqModel(
            price_stats=dataset.price_stats,
            news_dim=dataset.news_emb_dim,
            bin_count=dataset.bin_count,
            cross_ticker=True,
            hidden_dim=320,
            time_layers=12,
            time_heads=4,
        ).to(device)
        state = torch.load(Path(args.ckpt), map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        batch = dataset[0]
        ticker_mask = batch["ticker_mask_stack"]
        target_idx = batch.get("target_indices")
        if target_idx is not None:
            ticker_mask = ticker_mask[target_idx]
        split_mask = batch["split_mask_stack"].get(args.split, torch.ones_like(ticker_mask, dtype=torch.bool))
        if target_idx is not None:
            split_mask = split_mask[target_idx]
        mu, target, ticker_order, _ = compute_mu(model, batch, device)
        if target_idx is not None:
            target = target[target_idx]
    # Apply ticker exclusions if requested
    if exclude:
        keep_indices = [i for i, name in enumerate(ticker_order) if name not in exclude]
        if keep_indices:
            idx = torch.tensor(keep_indices, device=mu.device, dtype=torch.long)
            mu = mu[idx]
            target = target[idx]
            ticker_mask = ticker_mask[idx.to(ticker_mask.device)]
            split_mask = split_mask[idx.to(split_mask.device)]
            ticker_order = [ticker_order[i] for i in keep_indices]
    if args.mode == "backtest":
        metrics = backtest(
            mu=mu,
            target=target,
            ticker_mask=ticker_mask,
            split_mask=split_mask,
            policy=args.policy,
            gross=args.gross,
            z_cap=args.z_cap,
            w_max=w_max,
            top_k=args.top_k,
            top_pct=args.top_pct,
            cost_bps=args.cost_bps,
            leverage=leverage,
            margin_rate=margin_rate,
            dates=dates,
            ticker_order=ticker_order,
            log_trades_csv=log_trades_csv,
            log_daily_csv=log_daily_csv,
        )
        print(
            f"[backtest {args.split}] Sharpe={metrics['sharpe']:.4f} "
            f"mean={metrics['mean']:.6f} vol={metrics['vol']:.6f} "
            f"turnover={metrics['turnover']:.4f} cost_bps={args.cost_bps}"
        )
    else:
        trades = trading_preview(
            mu=mu,
            ticker_mask=ticker_mask,
            ticker_order=ticker_order,
            policy=args.policy,
            gross=args.gross,
            z_cap=args.z_cap,
            w_max=w_max,
            top_k=args.top_k,
            top_pct=args.top_pct,
        )
        print(f"[trade preview {args.split}] last-day weights (top 20 by weight):")
        for name, alpha, weight in trades[:20]:
            print(f"  {name:6s} alpha={alpha:+.5f} w={weight:+.5f}")


if __name__ == "__main__":
    main()
