#!/usr/bin/env python3
"""Auto-build run102 full-sequence PT files from run101 price-only sequences.

Collects all *_sequences_run101.jsonl files (or a provided subset), optionally
adds factor sequences, and emits run100_fullseq PTs with precomputed Kronos-mini
price_hidden embeddings and next-day targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from scripts.build_run100_fullseq import _compute_price_stats, _load_env_dotenv, build_fullseq_for_asset
from train.multichannel_dataset import load_sequence_file
from models.kronos_mini_fullseq import KronosMiniPriceEncoderFullSeq


DEFAULT_FACTORS = {
    "SPY": Path("data/processed/spy_sequences_run48.jsonl"),
    "BTC": Path("data/processed/btc_sequences_run48.jsonl"),
    "VIX": Path("data/processed/vix_sequences_run48.jsonl"),
    "QQQ": Path("data/processed/qqq_sequences_run48.jsonl"),
}


def _collect_run101_sequences(run101_dir: Path, symbols: List[str] | None) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(run101_dir.glob("*_sequences_run101.jsonl")):
        asset = path.name.split("_sequences_run101.jsonl")[0].upper()
        if symbols and asset not in symbols:
            continue
        mapping[asset] = path
    return mapping


def _parse_factor_overrides(raw: str | None) -> Dict[str, Path]:
    if not raw:
        return {}
    mapping: Dict[str, Path] = {}
    for entry in raw.split(","):
        if not entry.strip():
            continue
        if "=" not in entry:
            raise ValueError("Factor overrides must be NAME=path")
        name, path = entry.split("=", 1)
        mapping[name.strip().upper()] = Path(path.strip())
    return mapping


def _price_stats_for_assets(targets: Dict[str, Path], factors: Dict[str, Path]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for asset, path in {**targets, **factors}.items():
        samples = load_sequence_file(path)
        stats[asset.upper()] = _compute_price_stats(samples, asset, train_only=True)
    return stats


def build_targets(
    targets: Dict[str, Path],
    factors: Dict[str, Path],
    output_dir: Path,
    kronos_encoder: KronosMiniPriceEncoderFullSeq | None,
    device: torch.device,
    skip_existing: bool,
) -> List[Dict]:
    manifest: List[Dict] = []
    for asset, path in {**targets, **factors}.items():
        out_path = output_dir / f"{asset.lower()}_run100_fullseq.pt"
        if skip_existing and out_path.exists():
            print(f"[skip] {asset}: {out_path} already exists")
            manifest.append({"asset": asset, "path": str(out_path), "len": len(torch.load(out_path, map_location="cpu")["dates"])})
            continue
        info = build_fullseq_for_asset(
            asset,
            path,
            output_dir,
            kronos_encoder=kronos_encoder,
            device=device,
        )
        manifest.append(info)
        print(f"[build] {info['asset']} -> {info['path']} ({info['len']} days)")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-build run102 full-sequence dataset from run101 files.")
    parser.add_argument("--run101-dir", type=Path, default=Path("data/processed"), help="Directory with *_sequences_run101.jsonl files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/run102_fullseq"), help="Output directory for run100_fullseq PTs.")
    parser.add_argument("--symbols", default="", help="Optional comma-separated subset of symbols to build (default: all run101).")
    parser.add_argument("--factor-overrides", default="", help="Comma list NAME=path to override default factor sources.")
    parser.add_argument("--no-factors", action="store_true", help="Disable factor inclusion.")
    parser.add_argument("--device", default="cpu", help="Device for Kronos precompute.")
    parser.add_argument("--env-path", default=".env", help="Path to .env with HF token (for Kronos).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip PTs already present in output_dir.")
    parser.add_argument("--no-kronos", action="store_true", help="Do not precompute Kronos price_hidden (keeps None).")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of tickers to build (0 = all).")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run101_dir = args.run101_dir
    targets = _collect_run101_sequences(run101_dir, symbols if symbols else None)
    if args.limit and args.limit > 0:
        targets = dict(list(targets.items())[: args.limit])
    if not targets:
        raise ValueError("No run101 sequences found to build.")

    factor_map = {} if args.no_factors else dict(DEFAULT_FACTORS)
    factor_map.update(_parse_factor_overrides(args.factor_overrides))
    factor_map = {k: v for k, v in factor_map.items() if v.exists()}

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    kronos_encoder = None
    if not args.no_kronos:
        _load_env_dotenv(Path(args.env_path))
        stats = _price_stats_for_assets(targets, factor_map)
        kronos_encoder = KronosMiniPriceEncoderFullSeq(
            price_stats=stats,
            hidden_dim=256,
            freeze_kronos=True,
            stopgrad=True,
        ).to(device)

    manifest = build_targets(targets, factor_map, output_dir, kronos_encoder, device, args.skip_existing)
    manifest_path = output_dir / "run100_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
