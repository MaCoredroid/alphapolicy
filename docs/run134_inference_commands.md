# Run134 inference commands (2020 to today)

End-to-end commands to:
1) Pull the latest prices from Yahoo.
2) Incrementally build price-only run101 sequences (2020 to today, 60d context; tail-only rebuild).
3) Build a fullseq manifest with all dates marked live.
4) Dump signals with `outputs/run123/epoch_1.pt` (exclude SPY/QQQ/BTC/VIX) for trading.
5) Evaluate run133 portfolio checkpoints (1-50) and pick the best epoch.

Adjust `TODAY` and paths if you need a different cut; these examples use CPU for dumping/eval.

Assumptions:
- You run commands from the repo root: `cd /path/to/lumoStockGPT`.
- Required artifacts exist: `data/run102_fullseq/run100_manifest.json`,
  `outputs/run123/epoch_1.pt`, and `outputs/run133_portfolio/epoch_*.pt`.
- If you use different run IDs, update the `outputs/run123` and
  `outputs/run133_portfolio` paths accordingly.
- The ticker universe stays pinned to the training manifest.
  Adding new tickers (e.g., ETH) without retraining changes cross-ticker inputs
  and shifts early-day trades even with the same checkpoint.

## 1) Refresh raw prices (equities + factors + BTC/ETH)
```bash
cd /path/to/lumoStockGPT
TODAY="${TODAY:-$(date +%F)}"  # e.g., 2025-12-15
python3 - <<PY
import datetime as dt, json
import pandas as pd
from pathlib import Path
from scripts.fetch_data import _mk_dirs, fetch_yahoo_equity, fetch_crypto_via_yahoo, EQUITY_DIR, CRYPTO_DIR

START = dt.date(2020, 1, 2)  # ~5y back for this deployment
END = dt.date.fromisoformat("$TODAY")

manifest = json.loads(Path("data/run102_fullseq/run100_manifest.json").read_text())
assets = {str(e["asset"]).upper() for e in manifest} | {"SPY", "QQQ", "VIX", "BTC"}
cryptos = {"BTC"}
equities = sorted([a for a in assets if a not in cryptos])

_mk_dirs()

def append_and_dedup(path: Path, df_new: pd.DataFrame):
    if df_new is None or df_new.empty:
        return
    df_new = df_new.copy()
    df_new["ts"] = pd.to_datetime(df_new["ts"], utc=True)
    frames = [df_new]
    if path.exists():
        old = pd.read_csv(path)
        if not old.empty:
            old["ts"] = pd.to_datetime(old["ts"], utc=True)
            frames.append(old)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ts"]).sort_values("ts")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)

for sym in equities:
    append_and_dedup(EQUITY_DIR / f"{sym}.csv", fetch_yahoo_equity(sym, START, END))

for sym in cryptos:
    append_and_dedup(CRYPTO_DIR / f"{sym}.csv", fetch_crypto_via_yahoo(sym, START, END))
PY
```

## 2) Incremental run101 sequences (append last ~60d)
```bash
cd /path/to/lumoStockGPT
TODAY="${TODAY:-$(date +%F)}"
python3 - <<PY
import json, datetime as dt
from pathlib import Path
from scripts.build_run101_price_only_batch import build_one_incremental

manifest = json.loads(Path("data/run102_fullseq/run100_manifest.json").read_text())
symbols = {entry["asset"].upper() for entry in manifest} | {"SPY", "QQQ", "VIX", "BTC"}

start = dt.date(2020, 1, 1)
end = dt.date.fromisoformat("$TODAY")
out_dir = Path("data/processed_latest")
# Set this to the baseline run101 dir when you want historical trades to match.
# Example: use data/processed_2025 to match the run133 comparison baseline.
bin_edges_dir = Path("data/processed_2025")

for sym in sorted(symbols):
    build_one_incremental(
        sym,
        start,
        end,
        context=60,
        output_dir=out_dir,
        skip_existing=True,
        bin_edges_dir=bin_edges_dir,
    )
PY
```
Tip: keep `data/processed_latest` as the rolling dir so `bin_edges` stay fixed. Rebuilding into a fresh dir will recompute bins and shift `mu` for every date.

## 3) Build full-seq manifest for inference (all dates live)
```bash
cd /path/to/lumoStockGPT
SYMBOLS=$(python3 - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path("data/run102_fullseq/run100_manifest.json").read_text())
print(",".join(sorted({str(e["asset"]).upper() for e in manifest})))
PY
)
RUN100_ALL_SPLITS_TEST=1 PYTHONPATH=. python3 scripts/build_run102_fullseq_auto.py \
  --run101-dir data/processed_latest \
  --output-dir data/run102_fullseq_latest \
  --symbols "$SYMBOLS" \
  --factor-overrides "SPY=data/processed_latest/spy_sequences_run101.jsonl,BTC=data/processed_latest/btc_sequences_run101.jsonl,VIX=data/processed_latest/vix_sequences_run101.jsonl,QQQ=data/processed_latest/qqq_sequences_run101.jsonl" \
  --no-kronos
```

## 4) Dump signals with run123/epoch_1.pt (excl. SPY/QQQ/BTC/VIX)
```bash
cd /path/to/lumoStockGPT
OMP_NUM_THREADS=8 RUN101_OVERRIDE_NEWS_DIM=3072 PYTHONPATH=. \
python3 scripts/dump_run125_signals.py \
  --manifest data/run102_fullseq_latest/run100_manifest.json \
  --ckpt outputs/run123/epoch_1.pt \
  --split test \
  --out logs/run125/run123_signals_latest_excl.npz \
  --exclude_tickers SPY,QQQ,BTC,VIX,ETH \
  --device cpu
```
- Output: ticker count should match the baseline universe minus excluded names; date count depends on the latest raw data cut.
- Trade-date semantics: `mu[t]` uses features on `dates[t]` and is applied to the next trading day (`dates[t+1]`). The last real trade label is the final date in the NPZ (here 2025-12-15). `--emit_last_day` appends a zero-PnL preview on the next business day (2025-12-16 with current data).
- If ETH is not part of your universe, remove it from `--exclude_tickers`.
- If you do not need a news-dim override, drop `RUN101_OVERRIDE_NEWS_DIM=3072`.

## 5) Portfolio eval (run133 epochs 1-50 -> trades/daily/plots)
```bash
cd /path/to/lumoStockGPT
OUT_DIR=logs/run133_portfolio_latest
mkdir -p "$OUT_DIR"
for epoch in $(seq 1 50); do
  TAG="run133_epoch${epoch}"
  PYTHONPATH=. python3 scripts/eval_run130_policy.py \
    --signals_npz logs/run125/run123_signals_latest_excl.npz \
    --ckpt outputs/run133_portfolio/epoch_${epoch}.pt \
    --spy data/run102_fullseq_latest/spy_run100_fullseq.pt \
    --out_dir "$OUT_DIR" \
    --tag "$TAG" \
    --device cpu \
    --emit_last_day \
    --start_date 2025-01-01
done
```
- Outputs per epoch: `{out_dir}/{tag}_trades.csv`, `{tag}_daily.csv`, `{tag}_vs_spy.png` (monthly x-axis ticks).
- Recommended for live: **epoch 7** (Sharpe ~1.50, equity ~1.36x); trades cover 2025-01-03 to 2025-12-15 with a 2025-12-16 preview row from `--emit_last_day`. To run just epoch 7, call `eval_run130_policy.py` with `--ckpt outputs/run133_portfolio/epoch_7.pt --tag run133_latest_epoch7`.
