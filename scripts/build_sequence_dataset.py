"""Construct enriched JSONL sequence dataset with aligned price + news data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from ingest.local_clients import LocalCryptoClient, LocalEquityClient, LocalNewsClient

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_CONTEXT = 60
BIN_COUNT = 11
DEFAULT_NEWS_ASSETS = ["MARA", "SPY", "BTC"]
OPENAI_MODEL_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}
DEFAULT_OPENAI_TIMEOUT = 30.0
DEFAULT_OPENAI_MAX_RETRIES = 5


def compute_bin_edges(series: pd.Series, bin_count: int) -> np.ndarray:
    series = series.dropna()
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.quantile(series.values, quantiles)
    if len(np.unique(edges)) < len(edges):
        min_val = float(series.min())
        max_val = float(series.max())
        padding = 0.01 * (max_val - min_val if max_val > min_val else 1.0)
        edges = np.linspace(min_val - padding, max_val + padding, bin_count + 1)
    return edges


def load_equity(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / "equities" / f"{symbol.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity data for {symbol}: {path}")
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]  # keep latest if multiple rows per day
    return df


def load_crypto(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / "crypto" / f"{symbol.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing crypto data for {symbol}: {path}")
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]  # keep latest if multiple rows per day
    return df


def load_news(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    client = LocalNewsClient([symbol])
    df = client.fetch([symbol], start, end)
    if df.empty:
        return df
    if "publish_time" not in df.columns:
        raise ValueError("publish_time missing in news data")
    df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    return df.sort_values("publish_time")


class EmbeddingBackend(Protocol):
    dim: Optional[int]

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        ...


class SentenceTransformerEmbedder:
    """Wrapper around SentenceTransformer with optional batching."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        inputs = [(text.strip() or " ") for text in texts]
        embeddings = self.model.encode(
            inputs,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return [np.asarray(vec, dtype=np.float32) for vec in embeddings]


class OpenAIEmbedder:
    """Calls the OpenAI embeddings endpoint with basic retry logic."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout: float = DEFAULT_OPENAI_TIMEOUT,
        max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
        embedding_dim: Optional[int] = None,
        max_batch_size: int = 16,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                "The openai package is required for --news-embedding-backend=openai"
            ) from exc

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        self.dim: Optional[int] = embedding_dim or OPENAI_MODEL_DIMS.get(model_name)
        self.max_batch_size = max_batch_size

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), self.max_batch_size):
            chunk = texts[start : start + self.max_batch_size]
            payload = [(text.strip() or " ") for text in chunk]
            last_error: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=payload,
                    )
                    chunk_vecs = [
                        np.asarray(item.embedding, dtype=np.float32)
                        for item in response.data
                    ]
                    if self.dim is None and chunk_vecs:
                        self.dim = chunk_vecs[0].shape[0]
                    vectors.extend(chunk_vecs)
                    break
                except Exception as exc:  # pragma: no cover - external dependency
                    last_error = exc
                    sleep_seconds = min(2 ** attempt, 30)
                    time.sleep(sleep_seconds)
            else:
                raise RuntimeError(
                    f"OpenAI embedding failed after {self.max_retries} retries"
                ) from last_error
        return vectors


class ZeroEmbedder:
    """Deterministic embedder that returns zero vectors with a fixed dimension."""

    def __init__(self, embedding_dim: Optional[int] = None) -> None:
        if not embedding_dim or embedding_dim <= 0:
            raise ValueError("ZeroEmbedder requires --openai-embedding-dim to be set (>0).")
        self.dim = int(embedding_dim)
        self.skip_encode = True  # allows callers to bypass expensive loops

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        zeros = np.zeros(self.dim, dtype=np.float32)
        return [zeros for _ in texts]


def parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            cleaned = value.strip().strip('"').strip("'")
            values[key.strip()] = cleaned
    return values


def resolve_openai_api_key(explicit_key: Optional[str], env_file: Optional[Path]) -> Optional[str]:
    if explicit_key:
        return explicit_key
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    if env_file:
        if not env_file.exists():
            raise FileNotFoundError(f"OpenAI env file not found: {env_file}")
        values = parse_env_file(env_file)
        return values.get("OPENAI_API_KEY")
    return None


def create_embedder(args: argparse.Namespace) -> EmbeddingBackend:
    backend = getattr(args, "news_embedding_backend", "openai")
    if backend == "zero":
        return ZeroEmbedder(args.openai_embedding_dim)
    if backend == "openai":
        api_key = resolve_openai_api_key(args.openai_api_key, args.openai_env_file)
        if not api_key:
            raise ValueError(
                "Missing OPENAI_API_KEY. Provide --openai-api-key, set the env var, or pass --openai-env-file."
            )
        return OpenAIEmbedder(
            model_name=args.news_embedding_model,
            api_key=api_key,
            timeout=args.openai_timeout,
            max_retries=args.openai_max_retries,
            embedding_dim=args.openai_embedding_dim,
            max_batch_size=args.embedding_batch_size,
        )
    return SentenceTransformerEmbedder(
        args.news_embedding_model,
        batch_size=args.embedding_batch_size,
    )


def compute_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def assign_bins(values: Iterable[float], edges: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges) - 1
    bins = np.clip(bins, 0, len(edges) - 2)
    return bins


def compute_metrics(
    equity_close: pd.Series,
    log_returns: pd.Series,
    btc_close: pd.Series,
    btc_log_returns: pd.Series,
) -> pd.DataFrame:
    df = pd.DataFrame(index=log_returns.index)
    df["ret_20d"] = equity_close.pct_change(20)
    df["ret_60d"] = equity_close.pct_change(60)
    df["vol_20d"] = log_returns.rolling(20).std()
    rolling_mean = equity_close.rolling(60).mean()
    rolling_std = equity_close.rolling(60).std()
    df["level_z"] = (equity_close - rolling_mean) / rolling_std
    cov = log_returns.rolling(60).cov(btc_log_returns)
    var = btc_log_returns.rolling(60).var()
    df["beta_60d"] = cov / var
    return df


def compute_btc_metrics(btc_log_returns: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=btc_log_returns.index)
    df["btc_vol_20d"] = btc_log_returns.rolling(20).std()
    df["btc_ret_20d"] = np.exp(btc_log_returns.rolling(20).sum()) - 1
    return df


def load_news_assets(
    symbols: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, Dict[pd.Timestamp, List[Dict]]]:
    def _safe_text(row, key: str) -> str:
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        return str(val)

    news_by_symbol: Dict[str, Dict[pd.Timestamp, List[Dict]]] = {}
    for sym in symbols:
        df = load_news(sym, start, end)
        if df.empty:
            continue
        df["day"] = df["publish_time"].dt.floor("D")
        daily = {}
        for day, group in df.groupby("day"):
            entries = []
            for _, row in group.iterrows():
                article_id = str(row.get("id") or row.get("uuid") or f"{sym}_{row.name}")
                title = _safe_text(row, "title") or _safe_text(row, "headline")
                summary = _safe_text(row, "body") or _safe_text(row, "summary")
                text = (title + " " + summary).strip()
                entries.append(
                    {
                        "id": article_id,
                        "publish_time": row["publish_time"],
                        "title": title,
                        "summary": summary,
                        "text": text,
                        "length": len(text),
                    }
                )
            daily[day] = entries
        news_by_symbol[sym.upper()] = daily
    return news_by_symbol


def prepare_news_day(
    asset: str,
    day: pd.Timestamp,
    news_store: Dict[str, Dict[pd.Timestamp, List[Dict]]],
    article_texts: Dict[str, str],
    max_news: int,
    last_news_time: Dict[str, pd.Timestamp | None],
) -> Dict:
    entries = news_store.get(asset, {}).get(day, [])
    has_news = 1.0 if entries else 0.0
    stats = {
        "count": len(entries),
        "mean_len": float(np.mean([e["length"] for e in entries])) if entries else 0.0,
        "max_len": max([e["length"] for e in entries], default=0),
        "has_news": has_news,
    }
    hours_since_last = 999.0
    if entries:
        latest_time = max(e["publish_time"] for e in entries)
        prev_time = last_news_time.get(asset)
        if prev_time is not None:
            hours_since_last = (latest_time - prev_time).total_seconds() / 3600.0
        last_news_time[asset] = latest_time
    else:
        prev_time = last_news_time.get(asset)
        if prev_time is not None:
            end_of_day = pd.Timestamp(day) + pd.Timedelta(hours=16)
            hours_since_last = (end_of_day - prev_time).total_seconds() / 3600.0
    stats["hours_since_last"] = float(hours_since_last)

    selected = sorted(entries, key=lambda e: e["publish_time"], reverse=True)[:max_news]
    ids: List[str] = []
    for item in selected:
        article_id = item["id"]
        text = item.get("text") or f"{item.get('title', '')} {item.get('summary', '')}"
        ids.append(article_id)
        if article_id not in article_texts:
            article_texts[article_id] = (text or "").strip() or " "

    return {
        "ids": ids,
        "stats": stats,
    }


def build_embedding_cache(article_texts: Dict[str, str], embedder: EmbeddingBackend) -> Dict[str, np.ndarray]:
    if getattr(embedder, "skip_encode", False):
        return {}
    if not article_texts:
        return {}
    ids = list(article_texts.keys())
    cache: Dict[str, np.ndarray] = {}
    batch_size = getattr(embedder, "batch_size", getattr(embedder, "max_batch_size", 16))
    total = len(ids)
    for start in range(0, total, batch_size):
        chunk_ids = ids[start : start + batch_size]
        chunk_texts = [article_texts[cid] for cid in chunk_ids]
        vectors = embedder.encode_batch(chunk_texts)
        if len(vectors) != len(chunk_ids):
            raise RuntimeError("Embedding backend returned mismatched batch length")
        for cid, vec in zip(chunk_ids, vectors):
            cache[cid] = np.asarray(vec, dtype=np.float32)
            if embedder.dim is None:
                embedder.dim = cache[cid].shape[0]
        if total >= 200 and ((start // batch_size + 1) % 20 == 0):
            processed = min(len(cache), total)
            print(f"[embeddings] processed {processed}/{total} articles")
    return cache


def gather_context(
    idx: int,
    dates: List[pd.Timestamp],
    context_len: int,
    equity: pd.DataFrame,
    peers: Dict[str, pd.DataFrame],
    spy: pd.DataFrame,
    btc: pd.DataFrame,
    eth: pd.DataFrame | None,
    vix: pd.DataFrame | None,
    metrics: pd.DataFrame,
    btc_metrics: pd.DataFrame,
    news_assets: List[str],
    news_store: Dict[str, Dict[pd.Timestamp, List[Dict]]],
    article_texts: Dict[str, str],
    max_news_per_day: int,
    last_news_time: Dict[str, pd.Timestamp | None],
) -> List[Dict]:
    context = []
    for d in dates[idx - context_len : idx]:
        entry = {
            "date": d.strftime("%Y-%m-%d"),
            "equity": equity.loc[d].to_dict(),
            "spy": spy.loc[d].to_dict() if d in spy.index else {},
            "btc": btc.loc[d].to_dict() if d in btc.index else {},
        }
        if eth is not None:
            entry["eth"] = eth.loc[d].to_dict() if d in eth.index else {}
        if vix is not None:
            entry["vix"] = vix.loc[d].to_dict() if d in vix.index else {}
        if peers:
            entry["peers"] = {sym: df.loc[d].to_dict() for sym, df in peers.items() if d in df.index}
        entry["metrics"] = {
            k: float(v)
            for k, v in (metrics.loc[d] if d in metrics.index else pd.Series(dtype=float)).dropna().items()
        }
        entry["btc_metrics"] = {
            k: float(v)
            for k, v in (btc_metrics.loc[d] if d in btc_metrics.index else pd.Series(dtype=float)).dropna().items()
        }
        news_payload = {}
        for asset in news_assets:
            news_payload[asset] = prepare_news_day(
                asset=asset,
                day=d,
                news_store=news_store,
                article_texts=article_texts,
                max_news=max_news_per_day,
                last_news_time=last_news_time,
            )
        entry["news_assets"] = news_payload
        context.append(entry)
    return context


def build_sequences(
    symbol: str,
    peers: List[str],
    context_len: int,
    start: dt.date,
    end: dt.date,
    news_assets: List[str],
    max_news_per_day: int,
    embedder: EmbeddingBackend,
    asset_type: str = "auto",
    bin_edges_override: Optional[List[float]] = None,
) -> Tuple[List[Dict], np.ndarray, Dict[str, np.ndarray]]:
    symbol_upper = symbol.upper()
    eq_path = RAW_DIR / "equities" / f"{symbol_upper}.csv"
    crypto_path = RAW_DIR / "crypto" / f"{symbol_upper}.csv"
    mode = asset_type.lower()
    if mode not in {"auto", "equity", "crypto"}:
        raise ValueError("asset_type must be one of {'auto','equity','crypto'}.")
    if mode == "auto":
        if eq_path.exists():
            mode = "equity"
        elif crypto_path.exists():
            mode = "crypto"
        else:
            raise FileNotFoundError(f"No equity or crypto data found for {symbol_upper}.")
    if mode == "equity":
        equity = load_equity(symbol_upper)
    else:
        equity = load_crypto(symbol_upper)
    spy = load_equity("SPY")
    btc = load_crypto("BTC")
    eth = load_crypto("ETH") if (RAW_DIR / "crypto" / "ETH.csv").exists() else None
    vix = load_equity("VIX") if (RAW_DIR / "equities" / "VIX.csv").exists() else None
    peer_dfs = {
        peer: load_equity(peer)
        for peer in peers
        if (RAW_DIR / "equities" / f"{peer.upper()}.csv").exists()
    }

    # align indices and compute log returns
    close = equity["Close"] if "Close" in equity.columns else equity["close"]
    log_returns = compute_log_returns(close).dropna()
    equity = equity.loc[log_returns.index]
    spy = spy.reindex(log_returns.index).ffill()
    btc = btc.reindex(log_returns.index).ffill()
    if eth is not None:
        eth = eth.reindex(log_returns.index).ffill()
    if vix is not None:
        vix = vix.reindex(log_returns.index).ffill()
    peer_dfs = {sym: df.reindex(log_returns.index).ffill() for sym, df in peer_dfs.items()}

    dates = [d for d in log_returns.index if start <= d.date() <= end]

    news_store = load_news_assets(news_assets, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
    article_texts: Dict[str, str] = {}
    last_news_time = {asset: None for asset in news_assets}

    btc_close_series = btc["Close"] if "Close" in btc.columns else btc["close"]
    btc_log_returns = compute_log_returns(btc_close_series).dropna()
    btc_metrics = compute_btc_metrics(btc_log_returns)
    metrics = compute_metrics(close, log_returns, btc_close_series, btc_log_returns)

    sequences: List[Dict] = []
    if bin_edges_override is not None:
        bin_edges = np.array([float(x) for x in bin_edges_override], dtype=np.float64)
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise ValueError("bin_edges_override must have at least 2 values")
    else:
        bin_edges = compute_bin_edges(log_returns, BIN_COUNT)
    bin_series = pd.Series(assign_bins(log_returns.values, bin_edges), index=log_returns.index)

    for idx in range(context_len, len(dates)):
        target_day = dates[idx]
        target_return = float(log_returns.loc[target_day])
        target_bin = int(bin_series.loc[target_day])
        context = gather_context(
            idx=idx,
            dates=dates,
            context_len=context_len,
            equity=equity,
            peers=peer_dfs,
            spy=spy,
            btc=btc,
            eth=eth,
            vix=vix,
            metrics=metrics,
            btc_metrics=btc_metrics,
            news_assets=news_assets,
            news_store=news_store,
            article_texts=article_texts,
            max_news_per_day=max_news_per_day,
            last_news_time=last_news_time,
        )
        sequences.append(
            {
                "symbol": symbol,
                "target_date": target_day.strftime("%Y-%m-%d"),
                "context_window": context,
                "target_log_return": target_return,
                "target_bin": target_bin,
            }
        )
    # Append a preview sample for the next business day so we can emit tomorrow's trade
    # without needing the t+1 return yet. Target return/bin are left invalid.
    if dates:
        preview_day = pd.bdate_range(dates[-1], periods=2, freq="B")[1]
        ctx_idx = len(dates)
        context = gather_context(
            idx=ctx_idx,
            dates=dates,
            context_len=context_len,
            equity=equity,
            peers=peer_dfs,
            spy=spy,
            btc=btc,
            eth=eth,
            vix=vix,
            metrics=metrics,
            btc_metrics=btc_metrics,
            news_assets=news_assets,
            news_store=news_store,
            article_texts=article_texts,
            max_news_per_day=max_news_per_day,
            last_news_time=last_news_time,
        )
        sequences.append(
            {
                "symbol": symbol,
                "target_date": preview_day.date().isoformat(),
                "context_window": context,
                "target_log_return": float("nan"),
                "target_bin": -1,
            }
        )
    embedding_cache = build_embedding_cache(article_texts, embedder)
    return sequences, bin_edges, embedding_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enriched sequence dataset")
    parser.add_argument("--symbol", default="MARA")
    parser.add_argument(
        "--asset-type",
        choices=["auto", "equity", "crypto"],
        default="auto",
        help="Force the primary asset loader type (auto detects from available files).",
    )
    parser.add_argument("--peers", nargs="*", default=["RIOT", "HUT"])
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "mara_sequences.jsonl")
    parser.add_argument("--news-assets", nargs="*", default=DEFAULT_NEWS_ASSETS)
    parser.add_argument("--max-news-per-day", type=int, default=4)
    parser.add_argument(
        "--news-embedding-backend",
        choices=["sentence-transformers", "openai", "zero"],
        default="openai",
    )
    parser.add_argument(
        "--news-embedding-model",
        default="text-embedding-3-large",
        help="SentenceTransformer or OpenAI embedding model name.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size per embedding request.",
    )
    parser.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY.")
    parser.add_argument(
        "--openai-env-file",
        type=Path,
        default=None,
        help="Optional .env-style file to read OPENAI_API_KEY from.",
    )
    parser.add_argument(
        "--openai-timeout",
        type=float,
        default=DEFAULT_OPENAI_TIMEOUT,
        help="Per-request timeout for OpenAI embedding calls.",
    )
    parser.add_argument(
        "--openai-max-retries",
        type=int,
        default=DEFAULT_OPENAI_MAX_RETRIES,
        help="Retries for OpenAI embedding calls.",
    )
    parser.add_argument(
        "--openai-embedding-dim",
        type=int,
        default=None,
        help="Force the embedding dimension when using the OpenAI backend.",
    )
    parser.add_argument(
        "--target-embedding-dim",
        type=int,
        default=None,
        help="Optional dimension to pad/truncate embeddings to after encoding.",
    )
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    embedder = create_embedder(args)
    news_assets = [asset.upper() for asset in args.news_assets]
    sequences, bin_edges, embedding_cache = build_sequences(
        symbol=args.symbol,
        peers=args.peers,
        context_len=args.context,
        start=start,
        end=end,
        news_assets=news_assets,
        max_news_per_day=args.max_news_per_day,
        embedder=embedder,
        asset_type=args.asset_type,
    )

    target_dim = args.target_embedding_dim
    if target_dim:
        def _resize(vec: np.ndarray) -> np.ndarray:
            if vec.shape[0] == target_dim:
                return vec
            resized = np.zeros(target_dim, dtype=np.float32)
            length = min(target_dim, vec.shape[0])
            resized[:length] = vec[:length]
            return resized

        embedding_cache = {key: _resize(vec) for key, vec in embedding_cache.items()}
        embedder.dim = target_dim

    if embedder.dim is None:
        raise ValueError(
            "Embedding dimension is undefined. Provide --openai-embedding-dim or ensure at least one article produced embeddings."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        for item in sequences:
            fp.write(json.dumps(item) + "\n")
    print(f"Wrote {len(sequences)} sequences to {args.output}")

    embeddings_path = args.output.with_suffix(".news_embeddings.pt")
    torch.save({"embeddings": embedding_cache, "dim": embedder.dim}, embeddings_path)
    print(f"Saved {len(embedding_cache)} news embeddings to {embeddings_path}")

    meta = {
        "symbol": args.symbol,
        "bin_count": BIN_COUNT,
        "bin_edges": bin_edges.tolist(),
        "context_length": args.context,
        "start": args.start,
        "end": args.end,
        "news_assets": news_assets,
        "max_news_per_day": args.max_news_per_day,
        "news_embedding_dim": embedder.dim,
        "news_embedding_model": args.news_embedding_model,
        "news_embedding_backend": args.news_embedding_backend,
        "news_embedding_path": str(embeddings_path),
    }
    meta_path = args.output.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
