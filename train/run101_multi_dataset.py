"""Multi-ticker aligned dataset for Run101 cross-ticker attention."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class Run101MultiTickerDataset(Dataset):
    """Emit a single sample containing all tickers aligned over the full sequence.

    Shapes:
      price_seq_stack: [N, L, 6]
      price_hidden_stack: [N, L, H] (if precomputed)
      news_emb_stack: [N, L, K, D]
      news_mask_stack: [N, L, K]
      news_intensity_stack: [N, L, I]
      target_bin_stack: [N, L]
      target_probs_stack: [N, L, bins]
      target_log_return_stack: [N, L]
      split_mask_stack: dict of split -> [N, L]
      target_indices: list of indices for target tickers (context-only tickers excluded from loss/head)
      ticker_roles: list of roles per ticker ("target" or "context")
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        target_assets: Optional[List[str]] = None,
        context_assets: Optional[List[str]] = None,
        factor_assets: Optional[List[str]] = None,
        split: str = "train",
    ) -> None:
        with Path(manifest_path).open("r", encoding="utf-8") as fp:
            manifest = json.load(fp)
        self.records: Dict[str, Dict] = {}
        for entry in manifest:
            asset = str(entry["asset"]).upper()
            rec = torch.load(entry["path"], map_location="cpu", weights_only=False)
            self.records[asset] = rec
        all_assets = list(self.records.keys())
        factors = [a.upper() for a in (factor_assets or []) if a.upper() in self.records]
        ctx = [a.upper() for a in (context_assets or []) if a.upper() in self.records]
        tgt = [a.upper() for a in (target_assets or []) if a.upper() in self.records] if target_assets else [a for a in all_assets if a not in ctx and a not in factors]
        if not tgt:
            raise ValueError("No target tickers provided/found for Run101 multi dataset.")
        self.ticker_order = tgt + ctx  # targets first, then contexts
        self.target_indices = list(range(len(tgt)))
        self.ticker_roles = ["target"] * len(tgt) + ["context"] * len(ctx)
        self.split = split.lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train/val/test")
        dims = []
        for rec in self.records.values():
            dim_val = int(rec.get("news_embedding_dim", rec["news_emb"].size(-1)))
            if dim_val > 0:
                dims.append(dim_val)
        first = self.records[self.ticker_order[0]]
        self.news_emb_dim = max(dims) if dims else int(first["news_emb"].size(-1))
        override_news_dim = os.environ.get("RUN101_OVERRIDE_NEWS_DIM")
        if override_news_dim:
            self.news_emb_dim = int(override_news_dim)
        self.bin_count = first["target_probs"].size(-1)
        self.price_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
            asset: (rec["price_mean"], rec["price_std"]) for asset, rec in self.records.items()
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        del index
        price_list = []
        price_hidden_list = []
        news_emb_list = []
        news_mask_list = []
        news_intensity_list = []
        target_bin_list = []
        target_probs_list = []
        target_lr_list = []
        split_masks = {}
        bin_edges = []
        for asset in self.ticker_order:
            rec = self.records[asset]
            price_list.append(rec["price"])
            # Ignore any stored price_hidden to avoid mixed shapes; always recompute in model.
            price_hidden_list.append(None)
            news_emb_list.append(rec["news_emb"])
            news_mask_list.append(rec["news_mask"])
            news_intensity_list.append(rec["news_intensity"])
            target_bin_list.append(rec["target_bin"])
            target_probs_list.append(rec["target_probs"])
            target_lr_list.append(rec["target_log_return"])
            bin_edges.append(torch.tensor(rec["bin_edges"], dtype=torch.float32))
            sm = rec.get("split_mask", {})
            for k, v in sm.items():
                split_masks.setdefault(k, []).append(v)
        # fill missing split masks with zeros for consistency
        for split in ["train", "val", "test"]:
            if split not in split_masks:
                split_masks[split] = [torch.zeros_like(price_list[0][:,0], dtype=torch.bool) for _ in self.ticker_order]
        # pad to max length across tickers
        L_max = max(p.shape[0] for p in price_list)
        def pad_to(t: torch.Tensor, dim1: int, pad_val: float = 0.0):
            if t.size(0) == dim1:
                return t
            pad_shape = list(t.shape)
            pad_shape[0] = dim1 - t.size(0)
            pad = torch.full(pad_shape, pad_val, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)
        def pad_last_dim(t: torch.Tensor, dim_last: int, pad_val: float = 0.0):
            if t.size(-1) == dim_last:
                return t
            pad_shape = [0, dim_last - t.size(-1)]
            return torch.nn.functional.pad(t, pad_shape, value=pad_val)
        def pad_k(t: torch.Tensor, K: int, pad_val: float = 0.0):
            if K <= 0:
                return t[:, 0:0]  # force zero news slots
            if t.size(1) == K:
                return t
            pad_shape = list(t.shape)
            pad_shape[1] = K - t.size(1)
            pad = torch.full(pad_shape, pad_val, dtype=t.dtype)
            return torch.cat([t, pad], dim=1)
        # To avoid exploding memory with zero-news tickers, disable news padding in cross-ticker path.
        max_k = 0
        D = self.news_emb_dim
        price_seq_stack = torch.stack([pad_to(p, L_max, 0.0) for p in price_list], dim=0)
        price_hidden_stack = torch.stack(
            [pad_to(ph if ph is not None else price_list[i], L_max, 0.0) for i, ph in enumerate(price_hidden_list)],
            dim=0,
        )
        # news_emb_stack with K=0 to avoid huge dense padding
        news_emb_stack = torch.stack([pad_to(pad_k(pad_last_dim(ne, D, 0.0), max_k, 0.0), L_max, 0.0) for ne in news_emb_list], dim=0)
        news_mask_stack = torch.stack([pad_to(pad_k(nm, max_k, False), L_max, False) for nm in news_mask_list], dim=0)
        news_intensity_stack = torch.stack([pad_to(ni, L_max, 0.0) for ni in news_intensity_list], dim=0)
        target_bin_stack = torch.stack([pad_to(tb, L_max, -1) for tb in target_bin_list], dim=0)
        target_probs_stack = torch.stack([pad_to(tp, L_max, 0.0) for tp in target_probs_list], dim=0)
        target_lr_stack = torch.stack([pad_to(tlr, L_max, 0.0) for tlr in target_lr_list], dim=0)
        split_mask_stack = {k: torch.stack([pad_to(v, L_max, False) for v in vals], dim=0) for k, vals in split_masks.items()}
        batch: Dict[str, torch.Tensor] = {
            "price_seq_stack": price_seq_stack,
            "price_hidden_stack": price_hidden_stack,
            "news_emb_stack": news_emb_stack,
            "news_mask_stack": news_mask_stack,
            "news_intensity_stack": news_intensity_stack,
            "target_bin_stack": target_bin_stack,
            "target_probs_stack": target_probs_stack,
            "target_log_return_stack": target_lr_stack,
            "split_mask_stack": split_mask_stack,
            "bin_edges_stack": bin_edges,
            "target_indices": torch.tensor(self.target_indices, dtype=torch.long),
            "ticker_roles": self.ticker_roles,
            "ticker_mask_stack": torch.stack([pad_to(torch.ones_like(price_list[i][:,0], dtype=torch.bool), L_max, False) for i in range(len(price_list))], dim=0),
            "ticker_order": self.ticker_order,
        }
        # Apply split mask to targets so training only sees the requested split
        split_mask = split_mask_stack.get(self.split)
        if split_mask is not None:
            # Mask targets outside the split
            batch["target_bin_stack"] = torch.where(split_mask, batch["target_bin_stack"], torch.full_like(batch["target_bin_stack"], -1))
            batch["target_probs_stack"] = torch.where(split_mask.unsqueeze(-1), batch["target_probs_stack"], torch.zeros_like(batch["target_probs_stack"]))
            batch["target_log_return_stack"] = torch.where(split_mask, batch["target_log_return_stack"], torch.full_like(batch["target_log_return_stack"], 0.0))
            # Also mask feature tokens outside the split to avoid peeking at val/test during train
            feature_mask = split_mask.unsqueeze(-1)  # [N, L, 1]
            batch["price_seq_stack"] = batch["price_seq_stack"] * feature_mask
            batch["price_hidden_stack"] = batch["price_hidden_stack"] * feature_mask
            batch["news_emb_stack"] = batch["news_emb_stack"] * feature_mask.unsqueeze(-2)  # broadcast over K
            batch["news_mask_stack"] = batch["news_mask_stack"] & split_mask.unsqueeze(-1)
            batch["news_intensity_stack"] = batch["news_intensity_stack"] * feature_mask
            batch["ticker_mask_stack"] = batch["ticker_mask_stack"] & split_mask
        return batch
