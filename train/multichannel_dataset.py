"""Dataset producing price & news tensors for multi-asset fusion."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_CONTEXT = 60
RECENT_CONTEXT = 60
PRICE_ASSETS = [
    "MARA",
    "SPY",
    "BTC",
    "LULU",
    "CRWV",
    "NBIS",
    "VIX",
    "HIMS",
    "QQQ",
    "CRM",
    "AMD",
    "AAPL",
    "AMZN",
    "ARKK",
    "AVGO",
    "COIN",
    "GOOG",
    "MAGS",
]
ASSET_KEYS = {
    "MARA": "equity",
    "SPY": "spy",
    "BTC": "btc",
    "LULU": "equity",
    "BMNR": "equity",
    "CRWV": "equity",
    "NBIS": "equity",
    "VIX": "vix",
    "HIMS": "equity",
    "QQQ": "equity",
    "CRM": "equity",
    "AMD": "equity",
    "AAPL": "equity",
    "AMZN": "equity",
    "ARKK": "equity",
    "AVGO": "equity",
    "COIN": "equity",
    "GOOG": "equity",
    "MAGS": "equity",
}
PRICE_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
INTENSITY_FIELDS = ["count", "mean_len", "max_len", "hours_since_last", "has_news"]
DEFAULT_Z_BIN_EDGES = [float(x) for x in torch.linspace(-4.5, 4.5, 16)]  # 15 bins


def _build_time_feature_tensor(dates: List[Optional[str]]) -> torch.Tensor:
    """Convert date strings into minute/hour/weekday/day/month features."""

    features: List[List[float]] = []
    for value in dates:
        if not value:
            features.append([0.0, 0.0, 0.0, 1.0, 1.0])
            continue
        cleaned = value.replace("Z", "")
        try:
            dt_value = dt.datetime.fromisoformat(cleaned)
        except ValueError:
            try:
                # Handle timestamps with explicit timezone offsets by splitting.
                dt_value = dt.datetime.fromisoformat(cleaned.split("+", 1)[0])
            except ValueError:
                dt_value = dt.datetime(2020, 1, 1)
        features.append(
            [
                float(dt_value.minute),
                float(dt_value.hour),
                float(dt_value.weekday()),
                float(dt_value.day),
                float(dt_value.month),
            ]
        )
    if not features:
        features = [[0.0, 0.0, 0.0, 1.0, 1.0]]
    return torch.tensor(features, dtype=torch.float32)


def _is_equity_trading_day(date_value: dt.date) -> bool:
    return date_value.weekday() < 5


def _build_equity_calendar(dates: List[str]) -> Tuple[Dict[str, int], Dict[str, bool], dt.date]:
    index_map: Dict[str, int] = {}
    open_flags: Dict[str, bool] = {}
    last_index = -1
    min_date = None
    for date_str in dates:
        if date_str is None:
            continue
        date_obj = dt.date.fromisoformat(date_str)
        if min_date is None or date_obj < min_date:
            min_date = date_obj
        is_open = _is_equity_trading_day(date_obj)
        open_flags[date_str] = is_open
        if is_open:
            last_index += 1
        index_map[date_str] = max(last_index, 0)
    if not index_map:
        raise ValueError("Equity calendar cannot be empty")
    if last_index < 0:
        for idx, date_str in enumerate(dates):
            if date_str is None:
                continue
            index_map[date_str] = idx
        min_date = dt.date.fromisoformat(dates[0])
    if min_date is None:
        min_date = dt.date.fromisoformat(dates[0])
    return index_map, open_flags, min_date


def _map_to_equity_index(
    date_str: str,
    calendar_index: Dict[str, int],
    min_calendar_date: dt.date,
) -> Tuple[int, bool]:
    if date_str in calendar_index:
        return calendar_index[date_str], True
    date_obj = dt.date.fromisoformat(date_str)
    while date_obj > min_calendar_date:
        date_obj = date_obj - dt.timedelta(days=1)
        alt_key = date_obj.isoformat()
        if alt_key in calendar_index:
            return calendar_index[alt_key], False
    first_key = next(iter(calendar_index))
    return calendar_index[first_key], False


def _build_asset_trading_index(
    context_window: List[Dict],
    asset: str,
    calendar_index: Dict[str, int],
    min_calendar_date: dt.date,
    open_flags: Dict[str, bool],
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices: List[int] = []
    mask: List[bool] = []
    asset_kind = ASSET_KEYS.get(asset.upper(), "equity")
    for day in context_window:
        date_val = day.get("date")
        if date_val is None:
            continue
        idx, is_exact = _map_to_equity_index(date_val, calendar_index, min_calendar_date)
        indices.append(idx)
        if asset_kind in {"equity", "spy"}:
            mask.append(bool(is_exact and open_flags.get(date_val, False)))
        else:
            mask.append(True)
    if not indices:
        indices = list(range(DEFAULT_CONTEXT))
        mask = [True] * len(indices)
    return torch.tensor(indices, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def load_sequence_file(path: Path) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_news_embeddings(path: Path) -> Tuple[Dict[str, torch.Tensor], int]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    emb_dim = data.get("dim")
    embeddings = {
        str(article_id): torch.tensor(vec, dtype=torch.float32)
        for article_id, vec in data.get("embeddings", {}).items()
    }
    return embeddings, int(emb_dim)


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(val):
        return val
    return 0.0


def extract_price_sequence(context_window: List[Dict], asset_key: str) -> List[List[float]]:
    seq: List[List[float]] = []
    prev_close = None
    for day in context_window:
        data = day.get(asset_key, {})
        open_price = _safe_float(data.get("Open") or data.get("open"))
        high_price = _safe_float(data.get("High") or data.get("high"))
        low_price = _safe_float(data.get("Low") or data.get("low"))
        close_price = _safe_float(data.get("Close") or data.get("close"))
        volume = _safe_float(data.get("Volume") or data.get("volume"))
        log_volume = math.log1p(max(volume, 0.0))
        log_return = 0.0
        if prev_close and prev_close > 0 and close_price > 0:
            log_return = math.log(close_price / prev_close)
        prev_close = close_price if close_price > 0 else prev_close
        seq.append(
            [
                open_price,
                high_price,
                low_price,
                close_price,
                log_volume,
                log_return,
            ]
        )
    return seq


def compute_stats(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0)
    std = torch.clamp(std, min=1e-6)
    return mean, std


class AssetNewsDataset(Dataset):
    """Returns tensors for price/news per asset plus smoothed targets."""

    def __init__(
        self,
        samples: List[Dict],
        bin_edges: List[float],
        meta: Dict,
        news_embedding_path: Path,
        smoothing_main: float = 0.98,
        smoothing_neighbor: float = 0.01,
        bin_edges_override: Optional[List[float]] = None,
        cache_preprocessed: bool = False,
    ) -> None:
        self.samples = samples
        if bin_edges_override:
            if len(bin_edges_override) < 2:
                raise ValueError("bin_edges_override must contain at least two values.")
            self.z_bin_edges = [float(x) for x in bin_edges_override]
        else:
            self.z_bin_edges: List[float] = meta.get("z_bin_edges", DEFAULT_Z_BIN_EDGES)
        self.z_bin_edges_tensor = torch.tensor(self.z_bin_edges, dtype=torch.float32)
        self.z_bin_centers = 0.5 * (self.z_bin_edges_tensor[:-1] + self.z_bin_edges_tensor[1:])
        self.bin_count = len(self.z_bin_edges) - 1
        self.seq_len = int(meta.get("context_length", DEFAULT_CONTEXT))
        self.max_news = int(meta.get("max_news_per_day", 4))
        self.news_assets: List[str] = [asset.upper() for asset in meta.get("news_assets", PRICE_ASSETS)]
        self.smoothing_main = smoothing_main
        self.smoothing_neighbor = smoothing_neighbor
        self._cache_enabled = cache_preprocessed
        self._cached_records: Optional[List[Dict[str, torch.Tensor]]] = None
        if self._cache_enabled:
            self._cached_records = [self._build_record(idx) for idx in range(len(self.samples))]

        if not news_embedding_path.exists():
            raise FileNotFoundError(f"News embedding file not found: {news_embedding_path}")

        self.news_embeddings, self.news_emb_dim = load_news_embeddings(news_embedding_path)
        self.zero_news_embedding = torch.zeros(self.news_emb_dim, dtype=torch.float32)

        self.price_stats = self._compute_price_stats()
        self.intensity_stats = self._compute_intensity_stats()
        self.har_alpha = float(meta.get("har_alpha", 1e-6))
        self.sigma_scale = float(meta.get("sigma_scale_default", 0.8))
        self.har_beta_d = float(meta.get("har_beta_d", 0.5))
        self.har_beta_w = float(meta.get("har_beta_w", 0.3))
        self.har_beta_m = float(meta.get("har_beta_m", 0.2))
        self.primary_asset = meta.get("symbol", "MARA").upper()
        self.task_assets = [self.primary_asset]
        self.factor_assets = [asset for asset in PRICE_ASSETS if asset != self.primary_asset]
        self.price_assets = PRICE_ASSETS
        self.z_bin_edges_per_asset = {self.primary_asset: self.z_bin_edges}
        self.z_bin_centers_per_asset = {self.primary_asset: self.z_bin_centers}
        self.bin_count_per_asset = {self.primary_asset: self.bin_count}
        self.sigma_scale_per_asset = {self.primary_asset: self.sigma_scale}

    def _compute_price_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for asset in PRICE_ASSETS:
            asset_key = ASSET_KEYS[asset]
            all_rows = []
            for sample in self.samples:
                seq = extract_price_sequence(sample["context_window"], asset_key)
                all_rows.extend(seq)
            if not all_rows:
                stats[asset] = (
                    torch.zeros(6, dtype=torch.float32),
                    torch.ones(6, dtype=torch.float32),
                )
                continue
            matrix = torch.tensor(all_rows, dtype=torch.float32)
            stats[asset] = compute_stats(matrix)
        return stats

    def _compute_intensity_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for asset in self.news_assets:
            rows = []
            for sample in self.samples:
                for day in sample["context_window"]:
                    info = day.get("news_assets", {}).get(asset, {})
                    stats_dict = info.get("stats", {})
                    rows.append(
                        [
                            float(stats_dict.get(name, 0.0))
                            for name in INTENSITY_FIELDS
                        ]
                    )
            if not rows:
                stats[asset] = (
                    torch.zeros(len(INTENSITY_FIELDS), dtype=torch.float32),
                    torch.ones(len(INTENSITY_FIELDS), dtype=torch.float32),
                )
            else:
                matrix = torch.tensor(rows, dtype=torch.float32)
                stats[asset] = compute_stats(matrix)
        return stats

    def __len__(self) -> int:
        return len(self.samples)

    def _target_probs(self, target_bin: int) -> torch.Tensor:
        probs = torch.zeros(self.bin_count, dtype=torch.float32)
        idx = max(0, min(target_bin, self.bin_count - 1))
        probs[idx] = self.smoothing_main
        if idx - 1 >= 0:
            probs[idx - 1] = self.smoothing_neighbor
        if idx + 1 < self.bin_count:
            probs[idx + 1] = self.smoothing_neighbor
        probs /= probs.sum()
        return probs

    def _target_probs_from_value(self, value: float) -> Tuple[int, torch.Tensor]:
        tensor_val = torch.tensor(value, dtype=torch.float32)
        idx = torch.bucketize(tensor_val, self.z_bin_edges_tensor) - 1
        idx = int(idx.clamp(0, self.z_bin_centers.numel() - 1))
        return idx, self._target_probs(idx)

    def _extract_log_returns(self, context_window: List[Dict]) -> List[float]:
        returns: List[float] = []
        prev_close = None
        for day in context_window:
            eq = day.get("equity", {})
            close = _safe_float(eq.get("Close") or eq.get("close"))
            if prev_close and prev_close > 0 and close > 0:
                returns.append(math.log(close / prev_close))
            prev_close = close if close > 0 else prev_close
        return returns

    def _forecast_sigma(self, context_window: List[Dict]) -> float:
        returns = self._extract_log_returns(context_window)
        if not returns:
            return 1e-3
        rv_d = returns[-1] ** 2
        rv_w = sum(r ** 2 for r in returns[-5:]) / min(5, len(returns))
        rv_m = sum(r ** 2 for r in returns[-22:]) / min(22, len(returns))
        har = self.har_alpha + self.har_beta_d * rv_d + self.har_beta_w * rv_w + self.har_beta_m * rv_m
        return math.sqrt(max(har, 1e-8))

    def _build_price_tensor(self, context_window: List[Dict], asset: str) -> torch.Tensor:
        asset_key = ASSET_KEYS[asset]
        seq = extract_price_sequence(context_window, asset_key)
        tensor = torch.tensor(seq, dtype=torch.float32)
        mean, std = self.price_stats[asset]
        return (tensor - mean) / std

    def _build_news_tensors(
        self,
        context_window: List[Dict],
        asset: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_tensor = torch.zeros(self.seq_len, self.max_news, self.news_emb_dim, dtype=torch.float32)
        mask_tensor = torch.zeros(self.seq_len, self.max_news, dtype=torch.bool)
        intensity_tensor = torch.zeros(self.seq_len, len(INTENSITY_FIELDS), dtype=torch.float32)
        mean_int, std_int = self.intensity_stats[asset]

        for idx, day in enumerate(context_window):
            info = day.get("news_assets", {}).get(asset, {})
            ids = info.get("ids", [])[: self.max_news]
            for j, article_id in enumerate(ids):
                embedding = self.news_embeddings.get(article_id, self.zero_news_embedding)
                emb_tensor[idx, j] = embedding
                mask_tensor[idx, j] = True
            stats_dict = info.get("stats", {})
            intensity_vals = torch.tensor(
                [
                    float(stats_dict.get(name, 0.0))
                    for name in INTENSITY_FIELDS
                ],
                dtype=torch.float32,
            )
            intensity_tensor[idx] = (intensity_vals - mean_int) / std_int

        return emb_tensor, mask_tensor, intensity_tensor

    def _build_record(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        context = sample["context_window"]
        record: Dict[str, torch.Tensor] = {}
        recent_lengths: Dict[str, int] = {}
        for asset in PRICE_ASSETS:
            price_tensor = self._build_price_tensor(context, asset)
            recent_len = min(price_tensor.size(0), RECENT_CONTEXT)
            old_len = price_tensor.size(0) - recent_len
            record[f"price_{asset.lower()}"] = price_tensor
            record[f"price_recent_{asset.lower()}"] = price_tensor[-recent_len:, :]
            record[f"price_old_{asset.lower()}"] = price_tensor[:old_len, :] if old_len > 0 else price_tensor.new_zeros(0, price_tensor.size(-1))
            recent_lengths[asset] = recent_len

        context_dates = [day.get("date") for day in context]
        record["context_timestamps"] = context_dates
        record["context_time_features"] = _build_time_feature_tensor(context_dates)
        for asset in self.news_assets:
            emb, mask, intensity = self._build_news_tensors(context, asset)
            record[f"news_{asset.lower()}_emb"] = emb
            record[f"news_{asset.lower()}_mask"] = mask
            record[f"news_{asset.lower()}_intensity"] = intensity

        sigma_hat = self._forecast_sigma(context)
        z_value = float(sample["target_log_return"]) / max(sigma_hat, 1e-6)
        z_value = float(max(min(z_value, self.z_bin_edges[-1]), self.z_bin_edges[0]))
        target_bin, target_probs = self._target_probs_from_value(z_value)

        record["sigma_hat"] = torch.tensor(sigma_hat * self.sigma_scale, dtype=torch.float32)
        record["target_z"] = torch.tensor(z_value, dtype=torch.float32)
        record["target_bin"] = torch.tensor(target_bin, dtype=torch.long)
        record["target_probs"] = target_probs
        record["target_log_return"] = torch.tensor(float(sample["target_log_return"]), dtype=torch.float32)
        record["target_timestamp"] = sample.get("target_date")
        calendar_index, open_flags, min_equity_date = _build_equity_calendar(context_dates)
        trading_index, trading_mask = _build_asset_trading_index(
            context,
            self.primary_asset,
            calendar_index,
            min_equity_date,
            open_flags,
        )
        record["trading_day_index"] = trading_index
        record[f"trading_day_index_{self.primary_asset.lower()}"] = trading_index
        record[f"trading_day_mask_{self.primary_asset.lower()}"] = trading_mask
        recent_len = recent_lengths.get(self.primary_asset, min(len(context), RECENT_CONTEXT))
        record[f"trading_day_index_recent_{self.primary_asset.lower()}"] = trading_index[-recent_len:]
        record[f"trading_day_mask_recent_{self.primary_asset.lower()}"] = trading_mask[-recent_len:]
        return record
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._cached_records is not None:
            return self._cached_records[index]
        return self._build_record(index)


class MultiAssetCohortDataset(Dataset):
    """Dataset that aligns multiple per-asset sequence files into a single cohort sample."""

    def __init__(
        self,
        task_datasets: Dict[str, Path],
        *,
        factor_assets: Optional[List[str]] = None,
        factor_source_asset: Optional[str] = None,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
        smoothing_main: float = 0.98,
        smoothing_neighbor: float = 0.01,
        bin_edges_override: Optional[Dict[str, List[float]]] = None,
        cache_preprocessed: bool = False,
    ) -> None:
        if not task_datasets:
            raise ValueError("task_datasets must contain at least one asset path")
        self.task_assets = [asset.upper() for asset in task_datasets]
        self.factor_assets = [asset.upper() for asset in (factor_assets or ["SPY", "BTC"])]
        self.factor_source_asset = (factor_source_asset or self.task_assets[0]).upper()
        if self.factor_source_asset not in self.task_assets:
            raise ValueError("factor_source_asset must be one of the task assets")
        self.bin_edges_override = {
            asset.upper(): [float(x) for x in edges]
            for asset, edges in (bin_edges_override or {}).items()
        }

        self.asset_samples: Dict[str, List[Dict]] = {}
        self.asset_sample_maps: Dict[str, Dict[str, Dict]] = {}
        self.asset_meta: Dict[str, Dict] = {}
        self.asset_news_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
        self.asset_zero_news: Dict[str, torch.Tensor] = {}
        self.asset_news_assets: Dict[str, List[str]] = {}
        self.asset_max_news: Dict[str, int] = {}

        seq_lengths: List[int] = []
        news_dims: List[int] = []
        for asset, path in task_datasets.items():
            samples = load_sequence_file(path)
            if not samples:
                raise ValueError(f"Dataset {path} for {asset} has zero samples")
            self.asset_samples[asset.upper()] = samples
            self.asset_sample_maps[asset.upper()] = {sample["target_date"]: sample for sample in samples}
            meta_path = path.with_suffix(".meta.json")
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing meta for {asset}: {meta_path}")
            with meta_path.open("r", encoding="utf-8") as fp:
                meta = json.load(fp)
            self.asset_meta[asset.upper()] = meta
            news_path = Path(meta["news_embedding_path"])
            if not news_path.exists() and not news_path.is_absolute():
                candidate = meta_path.parent / news_path
                if candidate.exists():
                    news_path = candidate
            if not news_path.exists():
                raise FileNotFoundError(f"News embeddings not found for {asset}: {news_path}")
            embeddings, emb_dim = load_news_embeddings(news_path)
            self.asset_news_embeddings[asset.upper()] = embeddings
            self.asset_zero_news[asset.upper()] = torch.zeros(emb_dim, dtype=torch.float32)
            news_dims.append(emb_dim)
            seq_lengths.append(int(meta.get("context_length", DEFAULT_CONTEXT)))
            self.asset_news_assets[asset.upper()] = [a.upper() for a in meta.get("news_assets", [])]
            self.asset_max_news[asset.upper()] = int(meta.get("max_news_per_day", 4))

        if len(set(seq_lengths)) != 1:
            raise ValueError(f"Context lengths differ across datasets: {seq_lengths}")
        if len(set(news_dims)) != 1:
            raise ValueError(f"News embedding dims differ across datasets: {news_dims}")
        self.seq_len = seq_lengths[0]
        self.news_emb_dim = news_dims[0]
        self.max_news = max(self.asset_max_news.values())

        self.price_assets: List[str] = []
        for asset in self.task_assets + self.factor_assets:
            asset = asset.upper()
            if asset in ASSET_KEYS and asset not in self.price_assets:
                self.price_assets.append(asset)
        self.news_assets = sorted({asset for assets in self.asset_news_assets.values() for asset in assets})
        self.smoothing_main = smoothing_main
        self.smoothing_neighbor = smoothing_neighbor

        date_sets = [set(self.asset_sample_maps[asset]) for asset in self.task_assets]
        common_dates = set.intersection(*date_sets)
        if not common_dates:
            raise ValueError("No overlapping target dates across task datasets")
        self.common_dates = sorted(common_dates)
        if start_date or end_date:
            filtered = []
            for date_str in self.common_dates:
                date_val = dt.date.fromisoformat(date_str)
                if start_date and date_val < start_date:
                    continue
                if end_date and date_val > end_date:
                    continue
                filtered.append(date_str)
            if not filtered:
                raise ValueError("Date filter removes all samples.")
            self.common_dates = filtered

        self.z_bin_edges_per_asset: Dict[str, List[float]] = {}
        self.z_bin_edges_tensor: Dict[str, torch.Tensor] = {}
        self.z_bin_centers_per_asset: Dict[str, torch.Tensor] = {}
        self.bin_count_per_asset: Dict[str, int] = {}
        self.har_params_per_asset: Dict[str, Tuple[float, float, float, float]] = {}
        self.sigma_scale_per_asset: Dict[str, float] = {}
        for asset in self.task_assets:
            meta = self.asset_meta[asset]
            override = self.bin_edges_override.get(asset)
            edges = override if override else meta.get("z_bin_edges", DEFAULT_Z_BIN_EDGES)
            self.z_bin_edges_per_asset[asset] = edges
            tensor_edges = torch.tensor(edges, dtype=torch.float32)
            self.z_bin_edges_tensor[asset] = tensor_edges
            self.z_bin_centers_per_asset[asset] = 0.5 * (tensor_edges[:-1] + tensor_edges[1:])
            self.bin_count_per_asset[asset] = len(edges) - 1
            self.har_params_per_asset[asset] = (
                float(meta.get("har_alpha", 1e-6)),
                float(meta.get("har_beta_d", 0.5)),
                float(meta.get("har_beta_w", 0.3)),
                float(meta.get("har_beta_m", 0.2)),
            )
            self.sigma_scale_per_asset[asset] = float(meta.get("sigma_scale_default", 0.8))

        # Legacy attributes for compatibility (default to first task asset)
        primary = self.task_assets[0]
        self.z_bin_edges = self.z_bin_edges_per_asset[primary]
        self.z_bin_centers = self.z_bin_centers_per_asset[primary]
        self.bin_count = self.bin_count_per_asset[primary]
        self.sigma_scale = self.sigma_scale_per_asset[primary]

        self.price_stats = self._compute_price_stats()
        self.intensity_stats = self._compute_intensity_stats()
        self._cache_enabled = cache_preprocessed
        self._cached_records: Optional[List[Dict[str, torch.Tensor]]] = None
        if self._cache_enabled:
            self._cached_records = [self._build_record(idx) for idx in range(len(self.common_dates))]

    def __len__(self) -> int:
        return len(self.common_dates)

    def _context_for(self, asset: str, date: str) -> List[Dict]:
        asset = asset.upper()
        if asset in self.task_assets:
            return self.asset_sample_maps[asset][date]["context_window"]
        return self.asset_sample_maps[self.factor_source_asset][date]["context_window"]

    def _news_store_for(self, asset: str) -> Dict[str, torch.Tensor]:
        if asset in self.asset_news_embeddings:
            return self.asset_news_embeddings[asset]
        return self.asset_news_embeddings[self.factor_source_asset]

    def _zero_news_vec(self, asset: str) -> torch.Tensor:
        if asset in self.asset_zero_news:
            return self.asset_zero_news[asset]
        return self.asset_zero_news[self.factor_source_asset]

    def _compute_price_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for asset in self.price_assets:
            asset_key = ASSET_KEYS.get(asset)
            if asset_key is None:
                continue
            if asset in self.task_assets:
                samples = self.asset_samples[asset]
            else:
                samples = self.asset_samples[self.factor_source_asset]
            rows = []
            for sample in samples:
                seq = extract_price_sequence(sample["context_window"], asset_key)
                rows.extend(seq)
            if not rows:
                stats[asset] = (
                    torch.zeros(6, dtype=torch.float32),
                    torch.ones(6, dtype=torch.float32),
                )
            else:
                matrix = torch.tensor(rows, dtype=torch.float32)
                stats[asset] = compute_stats(matrix)
        return stats

    def _compute_intensity_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for asset in self.news_assets:
            if asset in self.task_assets:
                samples = self.asset_samples[asset]
            else:
                samples = self.asset_samples[self.factor_source_asset]
            rows = []
            for sample in samples:
                for day in sample["context_window"]:
                    info = day.get("news_assets", {}).get(asset, {})
                    stats_dict = info.get("stats", {})
                    rows.append([float(stats_dict.get(name, 0.0)) for name in INTENSITY_FIELDS])
            if not rows:
                stats[asset] = (
                    torch.zeros(len(INTENSITY_FIELDS), dtype=torch.float32),
                    torch.ones(len(INTENSITY_FIELDS), dtype=torch.float32),
                )
            else:
                matrix = torch.tensor(rows, dtype=torch.float32)
                stats[asset] = compute_stats(matrix)
        return stats

    def _target_probs(self, asset: str, target_bin: int) -> torch.Tensor:
        count = self.bin_count_per_asset[asset]
        probs = torch.zeros(count, dtype=torch.float32)
        idx = max(0, min(target_bin, count - 1))
        probs[idx] = self.smoothing_main
        if idx - 1 >= 0:
            probs[idx - 1] = self.smoothing_neighbor
        if idx + 1 < count:
            probs[idx + 1] = self.smoothing_neighbor
        probs /= probs.sum()
        return probs

    def _target_probs_from_value(self, asset: str, value: float) -> Tuple[int, torch.Tensor]:
        tensor_val = torch.tensor(value, dtype=torch.float32)
        idx = torch.bucketize(tensor_val, self.z_bin_edges_tensor[asset]) - 1
        idx = int(idx.clamp(0, self.z_bin_centers_per_asset[asset].numel() - 1))
        return idx, self._target_probs(asset, idx)

    def _forecast_sigma(self, context_window: List[Dict], asset: str) -> float:
        returns: List[float] = []
        prev_close = None
        for day in context_window:
            eq = day.get("equity", {})
            close = _safe_float(eq.get("Close") or eq.get("close"))
            if prev_close and prev_close > 0 and close > 0:
                returns.append(math.log(close / prev_close))
            prev_close = close if close > 0 else prev_close
        if not returns:
            return 1e-3
        alpha, beta_d, beta_w, beta_m = self.har_params_per_asset[asset]
        rv_d = returns[-1] ** 2
        rv_w = sum(r ** 2 for r in returns[-5:]) / min(5, len(returns))
        rv_m = sum(r ** 2 for r in returns[-22:]) / min(22, len(returns))
        har = alpha + beta_d * rv_d + beta_w * rv_w + beta_m * rv_m
        return math.sqrt(max(har, 1e-8))

    def _build_price_tensor(self, context_window: List[Dict], asset: str) -> torch.Tensor:
        asset_key = ASSET_KEYS.get(asset.upper())
        if asset_key is None:
            raise KeyError(f"Unknown asset key for {asset}")
        seq = extract_price_sequence(context_window, asset_key)
        tensor = torch.tensor(seq, dtype=torch.float32)
        mean, std = self.price_stats[asset.upper()]
        return (tensor - mean) / std

    def _build_news_tensors(self, context_window: List[Dict], asset: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_tensor = torch.zeros(self.seq_len, self.max_news, self.news_emb_dim, dtype=torch.float32)
        mask_tensor = torch.zeros(self.seq_len, self.max_news, dtype=torch.bool)
        intensity_tensor = torch.zeros(self.seq_len, len(INTENSITY_FIELDS), dtype=torch.float32)
        mean_int, std_int = self.intensity_stats.get(asset, (
            torch.zeros(len(INTENSITY_FIELDS), dtype=torch.float32),
            torch.ones(len(INTENSITY_FIELDS), dtype=torch.float32),
        ))
        news_store = self._news_store_for(asset)
        zero_vec = self._zero_news_vec(asset)

        for idx, day in enumerate(context_window):
            info = day.get("news_assets", {}).get(asset, {})
            ids = info.get("ids", [])[: self.max_news]
            for j, article_id in enumerate(ids):
                emb_tensor[idx, j] = news_store.get(article_id, zero_vec)
                mask_tensor[idx, j] = True
            stats_dict = info.get("stats", {})
            vals = torch.tensor(
                [float(stats_dict.get(name, 0.0)) for name in INTENSITY_FIELDS],
                dtype=torch.float32,
            )
            intensity_tensor[idx] = (vals - mean_int) / std_int
        return emb_tensor, mask_tensor, intensity_tensor

    def _build_record(self, index: int) -> Dict[str, torch.Tensor]:
        date = self.common_dates[index]
        record: Dict[str, torch.Tensor] = {}
        context_ref = self._context_for(self.factor_source_asset, date)
        context_dates = [day.get("date") for day in context_ref]
        record["context_timestamps"] = context_dates
        record["context_time_features"] = _build_time_feature_tensor(context_dates)
        calendar_index, open_flags, min_equity_date = _build_equity_calendar(context_dates)
        record["trading_day_index"] = torch.tensor(
            [calendar_index.get(date, 0) for date in context_dates],
            dtype=torch.long,
        )

        recent_lengths: Dict[str, int] = {}
        for asset in self.price_assets:
            context = self._context_for(asset, date)
            price_tensor = self._build_price_tensor(context, asset)
            recent_len = min(price_tensor.size(0), RECENT_CONTEXT)
            old_len = price_tensor.size(0) - recent_len
            record[f"price_{asset.lower()}"] = price_tensor
            record[f"price_recent_{asset.lower()}"] = price_tensor[-recent_len:, :]
            record[f"price_old_{asset.lower()}"] = price_tensor[:old_len, :] if old_len > 0 else price_tensor.new_zeros(0, price_tensor.size(-1))
            recent_lengths[asset] = recent_len
            idx_tensor, mask_tensor = _build_asset_trading_index(
                context,
                asset,
                calendar_index,
                min_equity_date,
                open_flags,
            )
            record[f"trading_day_index_{asset.lower()}"] = idx_tensor
            record[f"trading_day_mask_{asset.lower()}"] = mask_tensor
            record[f"trading_day_index_recent_{asset.lower()}"] = idx_tensor[-recent_len:]
            record[f"trading_day_mask_recent_{asset.lower()}"] = mask_tensor[-recent_len:]

        for asset in self.news_assets:
            context = self._context_for(asset, date)
            emb, mask, intensity = self._build_news_tensors(context, asset)
            record[f"news_{asset.lower()}_emb"] = emb
            record[f"news_{asset.lower()}_mask"] = mask
            record[f"news_{asset.lower()}_intensity"] = intensity

        for asset in self.task_assets:
            sample = self.asset_sample_maps[asset][date]
            context = sample["context_window"]
            sigma_hat = self._forecast_sigma(context, asset)
            target_lr = float(sample["target_log_return"])
            z_value = target_lr / max(sigma_hat, 1e-6)
            edges = self.z_bin_edges_per_asset[asset]
            z_value = float(max(min(z_value, edges[-1]), edges[0]))
            target_bin, target_probs = self._target_probs_from_value(asset, z_value)
            record[f"sigma_hat_{asset.lower()}"] = torch.tensor(
                sigma_hat * self.sigma_scale_per_asset[asset],
                dtype=torch.float32,
            )
            record[f"target_z_{asset.lower()}"] = torch.tensor(z_value, dtype=torch.float32)
            record[f"target_bin_{asset.lower()}"] = torch.tensor(target_bin, dtype=torch.long)
            record[f"target_probs_{asset.lower()}"] = target_probs
            record[f"target_log_return_{asset.lower()}"] = torch.tensor(target_lr, dtype=torch.float32)
            record[f"target_timestamp_{asset.lower()}"] = sample.get("target_date")
        return record
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._cached_records is not None:
            return self._cached_records[index]
        return self._build_record(index)
