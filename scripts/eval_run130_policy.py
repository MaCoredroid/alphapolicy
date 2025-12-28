#!/usr/bin/env python3
"""
Evaluate a run130 portfolio checkpoint on cached signals and write trades/daily logs plus a SPY comparison plot.

Usage example:
  python3 scripts/eval_run130_policy.py \
      --signals_npz logs/run125/run123_pre2019_signals_excl.npz \
      --ckpt outputs/run130_portfolio/epoch_10.pt \
      --spy data/run102_golden_pre2019/spy_run102_golden.pt \
      --out_dir logs/run130_portfolio
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import torch

from run129_train_portfolio import (
    CrossSectionPolicy,
    load_signals,
    precompute_static_features,
    scores_to_weights,
)


def evaluate(
    signals_path: Path,
    ckpt_path: Path,
    spy_path: Path,
    out_dir: Path,
    tag: str,
    device: str = "cpu",
    start_equity: float = 1_000_000.0,
    emit_last_day: bool = False,
    start_date: str | None = None,
) -> Tuple[Path, Path, Path]:
    torch_device = torch.device(device)
    data = load_signals(signals_path)
    dates = data["dates"]
    L = len(dates)
    mu_raw = np.nan_to_num(data["mu"][:, :L], nan=0.0).astype(np.float32)  # (N, L)
    ret_raw = np.nan_to_num(data["target"][:, :L], nan=0.0).astype(np.float32)
    live_raw = (data["ticker_mask"][:, :L] & data["split_mask"][:, :L]).astype(bool)

    start_idx = 0
    if start_date:
        for i, d in enumerate(dates):
            if d >= start_date:
                start_idx = i
                break
        else:
            raise ValueError(f"start_date {start_date} not found in dates")
        dates = dates[start_idx:]
        mu_raw = mu_raw[:, start_idx:]
        ret_raw = ret_raw[:, start_idx:]
        live_raw = live_raw[:, start_idx:]

    mu = mu_raw.T  # (T, N)
    ret = ret_raw.T
    live = live_raw.T

    X, g = precompute_static_features(mu, ret, live)

    ckpt = torch.load(ckpt_path, map_location=torch_device)
    args = ckpt["args"]
    policy = CrossSectionPolicy(
        feat_dim=7, global_dim=5, hidden=args["hidden"], layers=args["layers"], heads=args["heads"]
    ).to(torch_device)
    # Allow extra keys (e.g., risk heads) in newer checkpoints.
    policy.load_state_dict(ckpt["policy"], strict=False)
    policy.eval()

    X_t = torch.from_numpy(X).to(torch_device)
    g_t = torch.from_numpy(g).to(torch_device)
    ret_t = torch.from_numpy(ret).to(torch_device)
    live_t = torch.from_numpy(live).to(torch_device)

    cost = args["cost_bps"] * 1e-4
    temperature = args["temperature"]
    w_max = args["w_max"]
    gross = args["gross"]
    k_cap = args["k_cap"]

    w_prev = torch.zeros(mu.shape[1], device=torch_device)
    daily_rows = []
    trade_rows = []
    pnls = []

    for t in range(mu.shape[0] - 1):
        base_t = X_t[t]  # (N, 6)
        live_mask = live_t[t]
        if not bool(live_mask.any()):
            daily_rows.append((t + 1, dates[min(t + 1, L - 1)], 0.0, 0.0))
            continue

        held_prev = (w_prev > 0).float().unsqueeze(1)
        x_t = torch.empty(mu.shape[1], 7, device=torch_device)
        x_t[:, 0:5] = base_t[:, 0:5]
        x_t[:, 5] = held_prev.squeeze(1)
        x_t[:, 6] = base_t[:, 5]

        x = x_t.unsqueeze(0)
        g_curr = g_t[t].unsqueeze(0)
        live_curr = live_mask.unsqueeze(0)

        with torch.no_grad():
            scores = policy(x, g_curr, live_curr)
            w = scores_to_weights(scores, live_curr, temperature, w_max, gross, k_cap)[0]

        turnover = 0.5 * (w - w_prev).abs().sum().item()
        pnl = (w_prev * ret_t[t + 1]).sum().item() - cost * turnover
        pnls.append(pnl)
        daily_rows.append((t + 1, dates[min(t + 1, L - 1)], pnl, turnover))

        delta = w - w_prev
        abs_delta = delta.abs()
        cost_total = cost * turnover
        alloc = abs_delta / (abs_delta.sum() + 1e-12)
        if turnover > 0:
            for idx in range(mu.shape[1]):
                dw = delta[idx].item()
                if dw == 0 and w_prev[idx] == 0 and w[idx] == 0:
                    continue
                cost_alloc = cost_total * alloc[idx].item()
                pnl_gross = (w_prev[idx] * ret_t[t + 1][idx]).item()
                pnl_net = pnl_gross - cost_alloc
                ticker = data["ticker_order"][idx] if idx < len(data["ticker_order"]) else str(idx)
                trade_rows.append(
                    (
                        t + 1,
                        dates[min(t + 1, L - 1)],
                        ticker,
                        "buy" if dw > 0 else "sell" if dw < 0 else "hold",
                        float(w_prev[idx]),
                        float(w[idx]),
                        float(dw),
                        float(ret_t[t + 1][idx]),
                        float(pnl_gross),
                        float(cost_alloc),
                        float(pnl_net),
                    )
                )

        w_prev = w

    # Optionally emit last-day weights without PnL (no t+1 returns available).
    if emit_last_day and mu.shape[0] > 0:
        t = mu.shape[0] - 1
        base_t = X_t[t]
        live_mask = live_t[t]
        if bool(live_mask.any()):
            held_prev = (w_prev > 0).float().unsqueeze(1)
            x_t = torch.empty(mu.shape[1], 7, device=torch_device)
            x_t[:, 0:5] = base_t[:, 0:5]
            x_t[:, 5] = held_prev.squeeze(1)
            x_t[:, 6] = base_t[:, 5]

            x = x_t.unsqueeze(0)
            g_curr = g_t[t].unsqueeze(0)
            live_curr = live_mask.unsqueeze(0)

            with torch.no_grad():
                scores = policy(x, g_curr, live_curr)
                w = scores_to_weights(scores, live_curr, temperature, w_max, gross, k_cap)[0]

            delta = w - w_prev
            abs_delta = delta.abs()
            cost_total = 0.0
            alloc = abs_delta / (abs_delta.sum() + 1e-12) if abs_delta.sum() > 0 else abs_delta
            if t + 1 < len(dates):
                date_label = dates[t + 1]
            else:
                # Roll forward to the next business day for the preview label.
                date_label = pd.date_range(pd.Timestamp(dates[-1]), periods=2, freq="B")[1].date().isoformat()
            for idx in range(mu.shape[1]):
                dw = delta[idx].item()
                if dw == 0 and w_prev[idx] == 0 and w[idx] == 0:
                    continue
                ticker = data["ticker_order"][idx] if idx < len(data["ticker_order"]) else str(idx)
                trade_rows.append(
                    (
                        t + 1,
                        date_label,
                        ticker,
                        "buy" if dw > 0 else "sell" if dw < 0 else "hold",
                        float(w_prev[idx]),
                        float(w[idx]),
                        float(dw),
                        0.0,
                        0.0,
                        float(cost_total * alloc[idx].item()) if abs_delta.sum() > 0 else 0.0,
                        0.0,
                    )
                )
            daily_rows.append((t + 1, date_label, 0.0, 0.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / f"{tag}_trades.csv"
    daily_path = out_dir / f"{tag}_daily.csv"
    plot_path = out_dir / f"{tag}_vs_spy.png"

    with trades_path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(
            ["day_idx", "date", "ticker", "action", "w_prev", "w_new", "delta_w", "return", "pnl_gross", "cost_alloc", "pnl_net"]
        )
        wtr.writerows(trade_rows)

    with daily_path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["day_idx", "date", "pnl", "turnover"])
        wtr.writerows(daily_rows)

    # Equity curves
    spy_obj = torch.load(spy_path, map_location=torch_device)
    spy_ret = np.nan_to_num(spy_obj["target_log_return"][: ret.shape[0]].detach().cpu().numpy(), nan=0.0)

    strategy_equity = [start_equity]
    spy_equity = [start_equity]
    for i, pnl in enumerate(pnls):
        strategy_equity.append(strategy_equity[-1] * (1.0 + pnl))
        if i + 1 < len(spy_ret):
            spy_equity.append(spy_equity[-1] * (1.0 + spy_ret[i + 1]))

    plot_dates = dates[1 : len(strategy_equity)]
    truncate = min(len(plot_dates), len(spy_equity) - 1)
    plot_dates = plot_dates[:truncate]
    strategy_series = strategy_equity[1 : truncate + 1]
    spy_series = spy_equity[1 : truncate + 1]

    # Use monthly ticks for readability.
    plot_dt = mdates.datestr2num(plot_dates)
    plt.figure(figsize=(12, 6))
    policy_label = f"Policy ({tag})"
    plt.plot(plot_dt, strategy_series, label=policy_label)
    plt.plot(plot_dt, spy_series, label="SPY buy & hold")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    title = f"Equity curve: {tag} vs SPY"
    if plot_dates:
        title += f" ({plot_dates[0]} -> {plot_dates[-1]})"
    plt.title(title)
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)

    return trades_path, daily_path, plot_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals_npz", required=True, help="Cached signals .npz (mu/target/masks).")
    ap.add_argument("--ckpt", required=True, help="run130 portfolio checkpoint to evaluate.")
    ap.add_argument("--spy", required=True, help="Path to SPY cached tensor (run102_golden style).")
    ap.add_argument("--out_dir", required=True, help="Output directory for logs/plots.")
    ap.add_argument("--tag", default="run130_pre2019_epoch10", help="Prefix for output filenames.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run eval on.")
    ap.add_argument("--emit_last_day", action="store_true", help="Emit last-day trades/weights with zero PnL (no t+1 return).")
    ap.add_argument("--start_date", default=None, help="Optional start date (YYYY-MM-DD) to begin equity/trades.")
    args = ap.parse_args()

    trades_path, daily_path, plot_path = evaluate(
        Path(args.signals_npz),
        Path(args.ckpt),
        Path(args.spy),
        Path(args.out_dir),
        tag=args.tag,
        device=args.device,
        emit_last_day=bool(args.emit_last_day),
        start_date=args.start_date,
    )
    print(f"Wrote trades: {trades_path}")
    print(f"Wrote daily:  {daily_path}")
    print(f"Wrote plot:   {plot_path}")


if __name__ == "__main__":
    main()
