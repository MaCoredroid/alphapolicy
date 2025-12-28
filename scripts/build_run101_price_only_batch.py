#!/usr/bin/env python3
"""Batch-build price-only Run101 sequences for a list of tickers.

Uses zero news (max_news_per_day=0) and the ZeroEmbedder to match the existing
run101 price-only setup. Intended to fill S&P 500 gaps after fetching raw CSVs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable, List

import torch

from scripts.build_sequence_dataset import ZeroEmbedder, build_sequences


def write_sequence_files(symbol: str, sequences: List[dict], bin_edges, embedder, output_dir: Path) -> None:
    symbol_lower = symbol.lower()
    out_jsonl = output_dir / f"{symbol_lower}_sequences_run101.jsonl"
    out_meta = output_dir / f"{symbol_lower}_sequences_run101.meta.json"
    out_news = output_dir / f"{symbol_lower}_sequences_run101.news_embeddings.pt"

    # bin_edges may come from numpy (full build) or JSON (incremental); normalize to a plain list.
    bin_edges_list = bin_edges.tolist() if hasattr(bin_edges, "tolist") else list(bin_edges)

    output_dir.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fp:
        for item in sequences:
            fp.write(json.dumps(item) + "\n")
    torch.save({"embeddings": {}, "dim": embedder.dim}, out_news)

    meta = {
        "symbol": symbol.upper(),
        "bin_count": len(bin_edges_list) - 1,
        "bin_edges": bin_edges_list,
        "context_length": sequences[0]["context_length"] if sequences else 0,
        "start": sequences[0]["start"] if sequences else "",
        "end": sequences[-1]["target_date"] if sequences else "",
        "news_assets": [],
        "max_news_per_day": 0,
        "news_embedding_dim": embedder.dim,
        "news_embedding_model": "text-embedding-3-large",
        "news_embedding_backend": "zero",
        "news_embedding_path": str(out_news),
    }
    with out_meta.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(f"[write] {symbol.upper()} -> {out_jsonl}")


def _load_bin_edges(symbol: str, bin_edges_dir: Path | None) -> List[float] | None:
    if bin_edges_dir is None:
        return None
    symbol_lower = symbol.lower()
    meta_path = bin_edges_dir / f"{symbol_lower}_sequences_run101.meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    edges = meta.get("bin_edges") or meta.get("z_bin_edges")
    if not edges:
        return None
    return [float(x) for x in edges]


def build_one(
    symbol: str,
    start: dt.date,
    end: dt.date,
    context: int,
    output_dir: Path,
    skip_existing: bool,
    *,
    bin_edges_dir: Path | None = None,
    bin_edges_override: List[float] | None = None,
) -> None:
    symbol_upper = symbol.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / f"{symbol_upper.lower()}_sequences_run101.jsonl"
    if skip_existing and out_jsonl.exists():
        print(f"[skip] {symbol_upper}: already exists")
        return

    if bin_edges_override is None:
        bin_edges_override = _load_bin_edges(symbol_upper, bin_edges_dir)
    embedder = ZeroEmbedder(embedding_dim=1)
    sequences, bin_edges, _ = build_sequences(
        symbol=symbol_upper,
        peers=[],
        context_len=context,
        start=start,
        end=end,
        news_assets=[],
        max_news_per_day=0,
        embedder=embedder,
        asset_type="auto",
        bin_edges_override=bin_edges_override,
    )
    if not sequences:
        print(f"[skip] {symbol_upper}: no sequences built (possibly missing price data)")
        return

    # annotate context length and start for metadata
    for item in sequences:
        item["context_length"] = context
        item["start"] = start.isoformat()
    write_sequence_files(symbol_upper, sequences, bin_edges, embedder, output_dir)


def build_one_incremental(
    symbol: str,
    start: dt.date,
    end: dt.date,
    context: int,
    output_dir: Path,
    skip_existing: bool,
    *,
    bin_edges_dir: Path | None = None,
    bin_edges_override: List[float] | None = None,
) -> None:
    """Append only new days by reusing existing sequences/bin_edges and regenerating the tail."""
    symbol_upper = symbol.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / f"{symbol_upper.lower()}_sequences_run101.jsonl"
    out_meta = output_dir / f"{symbol_upper.lower()}_sequences_run101.meta.json"

    # If no existing data, fall back to full build.
    if not out_jsonl.exists() or not out_meta.exists():
        build_one(
            symbol,
            start,
            end,
            context,
            output_dir,
            skip_existing=False,
            bin_edges_dir=bin_edges_dir,
            bin_edges_override=bin_edges_override,
        )
        return

    if skip_existing:
        # Quick skip if the last target_date already covers end.
        try:
            last_line = None
            with out_jsonl.open("r", encoding="utf-8") as fp:
                for last_line in fp:
                    pass
            if last_line:
                last_rec = json.loads(last_line)
                last_target = dt.date.fromisoformat(last_rec["target_date"])
                if last_target >= end:
                    print(f"[skip] {symbol_upper}: up to date (last target {last_target})")
                    return
        except Exception:
            pass

    # Load existing sequences and bin_edges
    sequences: List[dict] = []
    with out_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            sequences.append(json.loads(line))
    meta = json.loads(out_meta.read_text())
    bin_edges = meta.get("bin_edges") or meta.get("z_bin_edges")
    if bin_edges_override is None:
        bin_edges_override = bin_edges

    last_target_date = dt.date.fromisoformat(sequences[-1]["target_date"]) if sequences else start
    # Back up enough calendar days to guarantee a full context window even across weekends/holidays.
    incremental_start = max(start, last_target_date - dt.timedelta(days=context * 2))

    embedder = ZeroEmbedder(embedding_dim=1)
    new_sequences, _, _ = build_sequences(
        symbol=symbol_upper,
        peers=[],
        context_len=context,
        start=incremental_start,
        end=end,
        news_assets=[],
        max_news_per_day=0,
        embedder=embedder,
        asset_type="auto",
        bin_edges_override=bin_edges_override,
    )
    # Keep only strictly newer target dates
    existing_dates = {rec["target_date"] for rec in sequences}
    new_filtered = [rec for rec in new_sequences if rec["target_date"] not in existing_dates and dt.date.fromisoformat(rec["target_date"]) > last_target_date]
    if not new_filtered:
        print(f"[skip] {symbol_upper}: no new target dates to append")
        return
    sequences.extend(new_filtered)
    # Reuse original bin_edges for consistency
    write_sequence_files(symbol_upper, sequences, bin_edges, embedder, output_dir)


def parse_symbols(values: Iterable[str]) -> List[str]:
    symbols: List[str] = []
    for v in values:
        cleaned = v.strip()
        if cleaned:
            symbols.append(cleaned.upper())
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-build price-only Run101 sequences (zero news).")
    parser.add_argument("--symbols", nargs="*", required=True, help="Symbols to process (comma-separated or space-separated).")
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--context", type=int, default=60, help="Context window length.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory for run101 sequences.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip symbols that already have run101 files.")
    parser.add_argument(
        "--bin-edges-dir",
        type=Path,
        default=None,
        help="Optional directory with existing run101 meta files to reuse bin_edges (keeps mu stable).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Append new days only (reuses existing sequences/bin_edges; regenerates last 60d window).",
    )
    args = parser.parse_args()

    symbols = []
    for sym in args.symbols:
        symbols.extend(parse_symbols(sym.split(",") if "," in sym else [sym]))
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    for sym in symbols:
        try:
            if args.incremental:
                build_one_incremental(
                    sym,
                    start,
                    end,
                    args.context,
                    args.output_dir,
                    args.skip_existing,
                    bin_edges_dir=args.bin_edges_dir,
                )
            else:
                build_one(
                    sym,
                    start,
                    end,
                    args.context,
                    args.output_dir,
                    args.skip_existing,
                    bin_edges_dir=args.bin_edges_dir,
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[error] {sym}: {exc}")


if __name__ == "__main__":
    main()
