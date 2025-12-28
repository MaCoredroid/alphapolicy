"""Build full-sequence Run100 datasets from existing run48/run49 JSONL files (no new fetches)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from train.multichannel_dataset import (
    ASSET_KEYS,
    INTENSITY_FIELDS,
    _build_time_feature_tensor,
    compute_stats,
    extract_price_sequence,
    load_news_embeddings,
    load_sequence_file,
)
from models.kronos_mini_fullseq import KronosMiniPriceEncoderFullSeq


# Run51 splits
TRAIN_END = dt.date(2024, 6, 30)
VAL_START = dt.date(2024, 7, 1)
VAL_END = dt.date(2024, 9, 30)
TEST_START = dt.date(2024, 10, 1)
TEST_END = dt.date(2024, 12, 31)


def _parse_asset_paths(spec: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for entry in spec.split(","):
        if not entry.strip():
            continue
        if "=" not in entry:
            raise ValueError(f"Asset spec must be NAME=path, got '{entry}'")
        name, path = entry.split("=", 1)
        mapping[name.upper()] = Path(path.strip())
    return mapping


def _collect_dates(samples: List[Dict]) -> List[str]:
    date_map: Dict[str, Dict] = {}
    for sample in samples:
        for day in sample["context_window"]:
            date = day.get("date")
            if date and date not in date_map:
                date_map[date] = day
    return sorted(date_map)


def _build_price_tensor(context: List[Dict], asset: str, asset_stats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    key = ASSET_KEYS.get(asset.upper(), "equity")
    seq = extract_price_sequence(context, key)
    tensor = torch.tensor(seq, dtype=torch.float32)
    mean, std = asset_stats
    return (tensor - mean) / std


def _is_train_date(date_str: str) -> bool:
    try:
        d = dt.date.fromisoformat(date_str)
    except Exception:
        return False
    return d <= TRAIN_END


def _compute_price_stats(samples: List[Dict], asset: str, train_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    key = ASSET_KEYS.get(asset.upper(), "equity")
    rows: List[List[float]] = []
    for sample in samples:
        for day in sample["context_window"]:
            date_val = day.get("date")
            if train_only and (date_val is None or not _is_train_date(str(date_val))):
                continue
            rows.append(extract_price_sequence([day], key)[0])
    matrix = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(1, 6)
    return compute_stats(matrix)


def _compute_intensity_stats(samples: List[Dict], asset: str, train_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    rows: List[List[float]] = []
    for sample in samples:
        for day in sample["context_window"]:
            date_val = day.get("date")
            if train_only and (date_val is None or not _is_train_date(str(date_val))):
                continue
            stats = day.get("news_assets", {}).get(asset, {}).get("stats", {})
            rows.append([float(stats.get(name, 0.0)) for name in INTENSITY_FIELDS])
    matrix = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros(1, len(INTENSITY_FIELDS))
    return compute_stats(matrix)


def _build_news_tensors(
    context: List[Dict],
    asset: str,
    embeddings: Dict[str, torch.Tensor],
    zero_vec: torch.Tensor,
    max_news: int,
    intensity_stats: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    emb_dim = zero_vec.size(0)
    emb_tensor = torch.zeros(len(context), max_news, emb_dim, dtype=torch.float32)
    mask_tensor = torch.zeros(len(context), max_news, dtype=torch.bool)
    intensity_tensor = torch.zeros(len(context), len(INTENSITY_FIELDS), dtype=torch.float32)
    mean_int, std_int = intensity_stats

    for idx, day in enumerate(context):
        info = day.get("news_assets", {}).get(asset, {})
        ids = info.get("ids", [])[:max_news]
        for j, article_id in enumerate(ids):
            emb_tensor[idx, j] = embeddings.get(str(article_id), zero_vec)
            mask_tensor[idx, j] = True
        stats_dict = info.get("stats", {})
        vals = torch.tensor([float(stats_dict.get(name, 0.0)) for name in INTENSITY_FIELDS], dtype=torch.float32)
        intensity_tensor[idx] = (vals - mean_int) / std_int
    return emb_tensor, mask_tensor, intensity_tensor


def _bucketize(value: float, edges: List[float]) -> int:
    tensor_val = torch.tensor(value, dtype=torch.float32)
    edge_tensor = torch.tensor(edges, dtype=torch.float32)
    idx = torch.bucketize(tensor_val, edge_tensor) - 1
    return int(idx.clamp(0, len(edges) - 2))


def _target_probs(idx: int, bin_count: int, main: float = 0.98, neighbor: float = 0.01) -> torch.Tensor:
    probs = torch.zeros(bin_count, dtype=torch.float32)
    idx = max(0, min(idx, bin_count - 1))
    probs[idx] = main
    if idx - 1 >= 0:
        probs[idx - 1] = neighbor
    if idx + 1 < bin_count:
        probs[idx + 1] = neighbor
    probs /= probs.sum()
    return probs


def _split_mask(date_strs: List[str]) -> Dict[str, torch.Tensor]:
    if os.environ.get("RUN100_ALL_SPLITS_TEST", "").lower() in {"1", "true", "yes"}:
        ones = torch.ones(len(date_strs), dtype=torch.bool)
        return {"train": ones.clone(), "val": ones.clone(), "test": ones.clone()}
    train = torch.zeros(len(date_strs), dtype=torch.bool)
    val = torch.zeros(len(date_strs), dtype=torch.bool)
    test = torch.zeros(len(date_strs), dtype=torch.bool)
    for i, ds in enumerate(date_strs):
        d = dt.date.fromisoformat(ds)
        if d <= TRAIN_END:
            train[i] = True
        elif VAL_START <= d <= VAL_END:
            val[i] = True
        elif TEST_START <= d <= TEST_END:
            test[i] = True
    return {"train": train, "val": val, "test": test}


def _load_env_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key and val and key not in os.environ:
            os.environ[key] = val


def build_fullseq_for_asset(
    asset: str,
    path: Path,
    output_dir: Path,
    *,
    kronos_encoder: KronosMiniPriceEncoderFullSeq | None,
    device: torch.device,
) -> Dict:
    samples = load_sequence_file(path)
    if not samples:
        raise ValueError(f"No samples in {path}")
    meta_path = path.with_suffix(".meta.json")
    with meta_path.open("r", encoding="utf-8") as fp:
        meta = json.load(fp)
    bin_edges = meta.get("bin_edges") or meta.get("z_bin_edges")
    if not bin_edges:
        raise ValueError(f"Missing bin edges in {meta_path}")
    bin_edges = [float(x) for x in bin_edges]
    bin_count = len(bin_edges) - 1
    max_news = int(meta.get("max_news_per_day", 0))
    news_path = Path(meta["news_embedding_path"])
    if not news_path.exists() and not news_path.is_absolute():
        candidate = meta_path.parent / news_path
        if candidate.exists():
            news_path = candidate
    news_embeddings, emb_dim = load_news_embeddings(news_path)
    zero_news = torch.zeros(emb_dim, dtype=torch.float32)

    dates = _collect_dates(samples)
    context_map = {d: None for d in dates}
    for sample in samples:
        for day in sample["context_window"]:
            date = day.get("date")
            if date in context_map and context_map[date] is None:
                context_map[date] = day
    context = [context_map[d] for d in dates]

    # Compute normalization stats on train dates only to avoid leaking val/test.
    price_stats = _compute_price_stats(samples, asset, train_only=True)
    intensity_stats = _compute_intensity_stats(samples, asset, train_only=True)
    price = _build_price_tensor(context, asset, price_stats)
    news_emb, news_mask, news_intensity = _build_news_tensors(
        context,
        asset,
        news_embeddings,
        zero_news,
        max_news,
        intensity_stats,
    )
    time_features = _build_time_feature_tensor(dates)

    price_hidden = None
    if kronos_encoder is not None:
        kronos_encoder.eval()
        with torch.no_grad():
            price_hidden = (
                kronos_encoder(
                    price.unsqueeze(0).to(device),
                    asset=asset,
                    time_features=time_features.unsqueeze(0).to(device),
                )
                .squeeze(0)
                .cpu()
            )

    # Targets: next-day log return (t -> t+1) derived from the denormalized price log-return feature.
    log_return_raw = price[:, 5] * price_stats[1][5] + price_stats[0][5]
    target_lr = torch.full((len(dates),), float("nan"), dtype=torch.float32)
    target_bin = torch.full((len(dates),), -1, dtype=torch.long)
    target_probs = torch.zeros(len(dates), bin_count, dtype=torch.float32)
    for i in range(len(dates) - 1):
        lr = float(log_return_raw[i + 1].item())
        if not math.isfinite(lr):
            continue
        target_lr[i] = lr
        idx = _bucketize(lr, bin_edges)
        target_bin[i] = idx
        target_probs[i] = _target_probs(idx, bin_count)

    split_mask = _split_mask(dates)

    record = {
        "asset": asset.upper(),
        "dates": dates,
        "bin_edges": bin_edges,
        "price": price,
        "news_emb": news_emb,
        "news_mask": news_mask,
        "news_intensity": news_intensity,
        "target_log_return": target_lr,
        "target_bin": target_bin,
        "target_probs": target_probs,
        "price_mean": price_stats[0],
        "price_std": price_stats[1],
        "intensity_mean": intensity_stats[0],
        "intensity_std": intensity_stats[1],
        "split_mask": split_mask,
        "max_news": max_news,
        "news_embedding_dim": emb_dim,
        "price_hidden": price_hidden if price_hidden is not None else None,
        "time_features": time_features,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{asset.lower()}_run100_fullseq.pt"
    torch.save(record, out_path)
    return {"asset": asset, "path": str(out_path), "len": len(dates)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Run100 full-sequence datasets from existing JSONL files.")
    parser.add_argument("--targets", required=True, help="Comma list NAME=path for target tickers")
    parser.add_argument("--factors", default="", help="Comma list NAME=path for factor tickers")
    parser.add_argument("--output_dir", default="data/run100_fullseq", help="Output directory for .pt files")
    parser.add_argument("--precompute_kronos", action="store_true", help="Precompute Kronos-mini hidden states into outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--env_path", default=".env", help="Path to .env containing HF token (optional)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    target_paths = _parse_asset_paths(args.targets)
    factor_paths = _parse_asset_paths(args.factors)

    kronos_encoder = None
    device = torch.device(args.device)
    if args.precompute_kronos:
        _load_env_dotenv(Path(args.env_path))
        all_stats = {}
        for asset, path in {**target_paths, **factor_paths}.items():
            samples = load_sequence_file(path)
            all_stats[asset.upper()] = _compute_price_stats(samples, asset, train_only=True)
        kronos_encoder = KronosMiniPriceEncoderFullSeq(
            price_stats=all_stats,
            hidden_dim=256,
            freeze_kronos=True,
            stopgrad=True,
        ).to(device)

    summary = []
    for asset, path in {**target_paths, **factor_paths}.items():
        info = build_fullseq_for_asset(
            asset,
            path,
            output_dir,
            kronos_encoder=kronos_encoder,
            device=device,
        )
        summary.append(info)
        print(f"Built {info['asset']} -> {info['path']} ({info['len']} days)")

    summary_path = output_dir / "run100_manifest.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Saved manifest to {summary_path}")


if __name__ == "__main__":
    main()
