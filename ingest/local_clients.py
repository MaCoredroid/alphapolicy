"""
Local data loaders reading from data/raw directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

RAW_DIR = Path("data/raw")
EQUITY_DIR = RAW_DIR / "equities"
CRYPTO_DIR = RAW_DIR / "crypto"
NEWS_DIR = RAW_DIR / "news"


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass
class LocalEquityClient:
    symbols: Iterable[str]

    def fetch_hourly(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        path = EQUITY_DIR / f"{symbol.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, parse_dates=["ts"])
        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        df = df[(df["ts"] >= start_utc) & (df["ts"] <= end_utc)]
        df = df.rename(columns={col: col.lower() for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns})
        df = df.set_index("ts").sort_index()
        return df


@dataclass
class LocalCryptoClient:
    symbols: Iterable[str]

    def fetch_reference_rates(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        path = CRYPTO_DIR / f"{symbol.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, parse_dates=["ts"])
        start_utc = _to_utc(start)
        end_utc = _to_utc(end)
        df = df[(df["ts"] >= start_utc) & (df["ts"] <= end_utc)]
        df = df.rename(columns={"Close": "close"})
        df = df.set_index("ts").sort_index()
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0.0
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_funding_rates(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame(columns=["ts", "funding_rate", "interval_hours"]).set_index("ts")


@dataclass
class LocalNewsClient:
    symbols: Iterable[str]

    def fetch(self, tickers: Iterable[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        records: List[pd.DataFrame] = []
        for symbol in tickers:
            path = NEWS_DIR / f"{symbol.upper()}.jsonl"
            if not path.exists():
                continue
            df = pd.read_json(path, lines=True)
            if "publish_time" in df.columns:
                df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True)
            elif "published_at" in df.columns:
                df["publish_time"] = pd.to_datetime(df["published_at"], utc=True)
            elif "datetime" in df.columns:
                date_series = df["datetime"]
                if pd.api.types.is_numeric_dtype(date_series):
                    df["publish_time"] = pd.to_datetime(date_series, unit="s", utc=True)
                else:
                    df["publish_time"] = pd.to_datetime(date_series, utc=True)
            elif "time" in df.columns:
                df["publish_time"] = pd.to_datetime(df["time"], utc=True)
            else:
                continue
            start_utc = _to_utc(start)
            end_utc = _to_utc(end)
            df = df[(df["publish_time"] >= start_utc) & (df["publish_time"] <= end_utc)]
            if "tickers" in df.columns:
                df["tickers"] = df["tickers"].apply(
                    lambda vals: [str(v).upper() for v in (vals or [])] or [symbol.upper()]
                )
            elif "related" in df.columns:
                df["tickers"] = df["related"].apply(
                    lambda vals: [str(v).upper() for v in (vals or [])] or [symbol.upper()]
                )
            else:
                df["tickers"] = [[symbol.upper()]] * len(df)
            df["source"] = df.get("source", "finnhub")
            df["symbol"] = symbol.upper()
            records.append(df)
        if not records:
            return pd.DataFrame(columns=["id", "publish_time", "source", "title", "body", "tickers"])
        combined = pd.concat(records, ignore_index=True)
        required_cols = ["id", "publish_time", "source", "headline", "summary", "tickers"]
        for col in required_cols:
            if col not in combined.columns:
                combined[col] = None
        combined = combined.rename(columns={"headline": "title", "summary": "body"})
        return combined[["id", "publish_time", "source", "title", "body", "tickers"]]
