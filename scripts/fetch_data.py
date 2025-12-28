"""Download daily equity, crypto, and news data for MARA research pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

DATA_DIR = Path("data/raw")
EQUITY_DIR = DATA_DIR / "equities"
CRYPTO_DIR = DATA_DIR / "crypto"
NEWS_DIR = DATA_DIR / "news"

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
MARKETAUX_NEWS_URL = "https://api.marketaux.com/v1/news/all"

CRYPTO_YAHOO_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}
YAHOO_SYMBOL_OVERRIDES = {
    "VIX": "^VIX",
}


def _mk_dirs() -> None:
    EQUITY_DIR.mkdir(parents=True, exist_ok=True)
    CRYPTO_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_DIR.mkdir(parents=True, exist_ok=True)


def _date_range_to_unix(start: dt.date, end: dt.date) -> Tuple[int, int]:
    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.max)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def _normalize_yahoo_symbol(symbol: str) -> str:
    """Map local symbols to Yahoo-friendly tickers (handle dots/overrides)."""
    upper = symbol.upper()
    if upper in YAHOO_SYMBOL_OVERRIDES:
        return YAHOO_SYMBOL_OVERRIDES[upper]
    if "." in symbol:
        return symbol.replace(".", "-")
    return symbol


def fetch_yahoo_equity(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    request_symbol = _normalize_yahoo_symbol(symbol)
    period1, period2 = _date_range_to_unix(start, end)
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "includePrePost": "false",
        "events": "history",
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LumoStockGPT/0.1)"}
    url = YAHOO_CHART_URL.format(symbol=request_symbol)
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    try:
        return parse_yahoo_chart(payload)
    except ValueError:
        # Fallback for newly listed tickers: fetch max range then trim.
        fallback_params = {
            "range": "max",
            "interval": "1d",
            "includePrePost": "false",
            "events": "history",
        }
        resp = requests.get(url, params=fallback_params, headers=headers, timeout=30)
        resp.raise_for_status()
        alt_payload = resp.json()
        df = parse_yahoo_chart(alt_payload)
        if df.empty:
            return df
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        return df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)].reset_index(drop=True)


def parse_yahoo_chart(payload: Dict) -> pd.DataFrame:
    if "chart" not in payload or payload["chart"].get("error"):
        raise ValueError("Yahoo Finance chart payload error")
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    if not timestamps:
        return pd.DataFrame()
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(quote)
    df["ts"] = pd.to_datetime(timestamps, unit="s", utc=True)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df[["ts", "Open", "High", "Low", "Close", "Volume"]]


def save_equity(symbol: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    out = EQUITY_DIR / f"{symbol.upper()}.csv"
    frame.to_csv(out, index=False)


def fetch_crypto_via_yahoo(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    yahoo_symbol = CRYPTO_YAHOO_SYMBOLS[symbol.upper()]
    df = fetch_yahoo_equity(yahoo_symbol, start, end)
    return df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})


def save_crypto(symbol: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    out = CRYPTO_DIR / f"{symbol.upper()}.csv"
    frame.to_csv(out, index=False)


def _chunked_ranges(start: dt.date, end: dt.date, chunk_days: int = 90) -> List[Tuple[dt.date, dt.date]]:
    ranges = []
    cur = start
    delta = dt.timedelta(days=chunk_days)
    while cur <= end:
        nxt = min(end, cur + delta)
        ranges.append((cur, nxt))
        cur = nxt + dt.timedelta(days=1)
    return ranges


def fetch_finnhub_news(symbol: str, start: dt.date, end: dt.date, api_key: str) -> List[Dict]:
    records: List[Dict] = []
    for chunk_start, chunk_end in _chunked_ranges(start, end, chunk_days=90):
        params = {
            "symbol": symbol,
            "from": chunk_start.isoformat(),
            "to": chunk_end.isoformat(),
            "token": api_key,
        }
        for attempt in range(6):
            resp = requests.get(FINNHUB_COMPANY_NEWS_URL, params=params, timeout=30)
            if resp.status_code == 429:
                sleep_s = 10 * (attempt + 1)
                print(f"Finnhub 429 for {symbol} {chunk_start}->{chunk_end}; sleeping {sleep_s}s")
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list):
                records.extend(payload)
            break
        time.sleep(0.5)
    records.sort(key=lambda x: x.get("datetime", 0))
    return records


def fetch_marketaux_news(symbol: str, start: dt.date, end: dt.date, api_token: str, limit: int = 50) -> List[Dict]:
    """Fetch news via Marketaux with pagination and chunked date windows."""
    records: List[Dict] = []
    for chunk_start, chunk_end in _chunked_ranges(start, end, chunk_days=90):
        page = 1
        while True:
            params = {
                "api_token": api_token,
                "symbols": symbol,
                "language": "en",
                "published_after": chunk_start.isoformat(),
                "published_before": chunk_end.isoformat(),
                "limit": limit,
                "page": page,
                "group_similar": "true",
                "filter_entities": "true",
            }
            for attempt in range(6):
                resp = requests.get(MARKETAUX_NEWS_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    sleep_s = 10 * (attempt + 1)
                    print(f"Marketaux 429 for {symbol} {chunk_start}->{chunk_end} page {page}; sleeping {sleep_s}s")
                    time.sleep(sleep_s)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                break
            else:
                break
            data = payload.get("data") or []
            records.extend(data)
            returned = payload.get("meta", {}).get("returned", len(data))
            if returned < limit or not data:
                break
            page += 1
            time.sleep(0.5)
        time.sleep(0.5)
    records.sort(key=lambda x: x.get("published_at") or x.get("published_at"))
    return records


def save_news(symbol: str, items: Iterable[Dict]) -> None:
    out = NEWS_DIR / f"{symbol.upper()}.jsonl"
    with out.open("w", encoding="utf-8") as fp:
        for item in items:
            fp.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fetch equity, crypto, and news data")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=dt.date.today().isoformat(), help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--equities",
        nargs="*",
        default=["MARA", "LULU", "SPY", "RIOT", "HUT"],
        help="Equity tickers to download",
    )
    parser.add_argument("--crypto", nargs="*", default=["BTC", "ETH"], help="Crypto tickers")
    parser.add_argument("--news", nargs="*", default=["MARA"], help="Symbols for news pull")
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    _mk_dirs()

    for symbol in args.equities:
        frame = fetch_yahoo_equity(symbol, start, end)
        save_equity(symbol, frame)

    for symbol in args.crypto:
        frame = fetch_crypto_via_yahoo(symbol, start, end)
        save_crypto(symbol, frame)

    marketaux_token = os.getenv("MARKETAUX_API_TOKEN")
    if not marketaux_token:
        raise SystemExit("MARKETAUX_API_TOKEN is required for news; Finnhub fallback is disabled.")
    for raw_symbol in args.news:
        symbol = raw_symbol.upper()
        items = fetch_marketaux_news(symbol, start, end, marketaux_token)
        save_news(symbol, items)
        print(f"[news] {symbol}: downloaded {len(items)} articles via Marketaux (only backend enabled)")
if __name__ == "__main__":
    main()
