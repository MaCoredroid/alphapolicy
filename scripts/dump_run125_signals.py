#!/usr/bin/env python3
"""Dump per-ticker daily mu/targets/masks for a frozen Run100 model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from train.run101_multi_dataset import Run101MultiTickerDataset
from train.run100_model import Run100FullSeqModel
from scripts.run125_alpha_policy import compute_mu


def dump_signals(manifest: Path, ckpt: Path, split: str, device: torch.device, out_path: Path, exclude: set[str]) -> None:
    # Include all assets as targets; exclude only those explicitly requested.
    dataset = Run101MultiTickerDataset(
        manifest,
        target_assets=None,
        context_assets=[],
        factor_assets=[],
        split=split,
    )
    model = Run100FullSeqModel(
        price_stats=dataset.price_stats,
        news_dim=dataset.news_emb_dim,
        bin_count=dataset.bin_count,
        hidden_dim=320,
        time_layers=12,
        time_heads=4,
        factor_cross_layers=2,
        factor_cross_heads=4,
        cross_ticker=True,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    batch = dataset[0]
    mask = batch["ticker_mask_stack"]
    split_mask = batch["split_mask_stack"][split]
    mu, target, ticker_order, bin_edges = compute_mu(model, batch, device)
    if exclude:
        keep = [i for i, name in enumerate(ticker_order) if name.upper() not in exclude]
        if keep:
            idx = torch.tensor(keep, device=mu.device, dtype=torch.long)
            mu = mu[idx]
            target = target[idx]
            mask = mask[idx.to(mask.device)]
            split_mask = split_mask[idx.to(split_mask.device)]
            ticker_order = [ticker_order[i] for i in keep]
            bin_edges = [bin_edges[i] for i in keep]

    # Grab dates from the first remaining target ticker to stay on the equity trading calendar.
    dates: List[str] = []
    if ticker_order:
        rec = dataset.records.get(ticker_order[0])
        if rec and "dates" in rec:
            dates = rec["dates"]

    # Keep only dates where at least one target ticker is live.
    live_mask = (mask & split_mask)
    keep = live_mask.any(dim=0)
    if keep.sum().item() > 0 and keep.sum().item() < mask.size(1):
        keep_np = keep.cpu().numpy()
        mu = mu[:, keep]
        target = target[:, keep]
        mask = mask[:, keep.to(mask.device)]
        split_mask = split_mask[:, keep.to(split_mask.device)]
        dates = list(np.array(dates)[keep_np[: len(dates)]])  # align to trading calendar length

    out = {
        "mu": mu.cpu().numpy(),  # [N,L]
        "target": target.cpu().numpy(),  # [N,L]
        "ticker_mask": mask.cpu().numpy(),  # [N,L]
        "split_mask": split_mask.cpu().numpy(),  # [N,L]
        "ticker_order": np.array(ticker_order),
        "dates": np.array(dates),
        "bin_edges": np.array([np.array(torch.tensor(be).cpu().numpy()) for be in bin_edges], dtype=object),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"Saved signals to {out_path} ({out['mu'].shape[0]} tickers, {out['mu'].shape[1]} days)")


def main() -> None:
    p = argparse.ArgumentParser(description="Dump per-ticker mu/target/masks for a frozen Run100 model.")
    p.add_argument("--manifest", required=True, help="Path to run100 manifest")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--exclude_tickers",
        default="",
        help="Comma-separated tickers to exclude (case-insensitive).",
    )
    args = p.parse_args()
    exclude = {t.strip().upper() for t in args.exclude_tickers.split(",") if t.strip()}
    dump_signals(Path(args.manifest), Path(args.ckpt), args.split, torch.device(args.device), Path(args.out), exclude)


if __name__ == "__main__":
    main()
