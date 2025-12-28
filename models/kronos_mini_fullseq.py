"""Frozen Kronos-mini price encoder for full-sequence daily modeling."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class KronosMiniPriceEncoderFullSeq(nn.Module):
    """Encode full daily sequences with Kronos-mini and bridge to model dim."""

    def __init__(
        self,
        *,
        price_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        kronos_model: str = "NeoQuasar/Kronos-mini",
        kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-2k",
        kronos_device: Optional[str] = None,
        max_context: int = 2048,
        hidden_dim: int = 256,
        freeze_kronos: bool = True,
        stopgrad: bool = True,
    ) -> None:
        super().__init__()
        if not price_stats:
            raise ValueError("price_stats are required for Kronos-mini price encoder.")
        self.hidden_dim = int(hidden_dim)
        self.max_context = max(1, int(max_context))
        self.freeze_kronos = bool(freeze_kronos)
        self.stopgrad = bool(stopgrad)
        stats = self._prepare_price_stats(price_stats)
        self.price_stats: Dict[str, Tuple[str, str]] = {}
        for asset, (mean, std) in stats.items():
            safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in asset.lower())
            mean_name = f"_kmini_price_mean_{safe}"
            std_name = f"_kmini_price_std_{safe}"
            self.register_buffer(mean_name, mean)
            self.register_buffer(std_name, std)
            self.price_stats[asset] = (mean_name, std_name)
        self._init_kronos(kronos_model, kronos_tokenizer, kronos_device)
        self.bridge = nn.Linear(self.backbone_dim, self.hidden_dim)
        self._time_cache: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _prepare_price_stats(
        price_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for asset, pair in price_stats.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            mean, std = pair
            stats[asset.upper()] = (
                mean.detach().clone(),
                std.detach().clone(),
            )
        if not stats:
            raise ValueError("price_stats must include mean/std tensors per asset.")
        return stats

    def _init_kronos(
        self,
        model_name: str,
        tokenizer_name: str,
        kronos_device: Optional[str],
    ) -> None:
        try:
            from model import Kronos, KronosTokenizer  # type: ignore
        except ImportError:
            import sys
            from pathlib import Path
            kronos_root = Path(__file__).resolve().parent.parent / "third_party" / "Kronos"
            if kronos_root.exists():
                sys.path.append(str(kronos_root))
            from model import Kronos, KronosTokenizer  # type: ignore
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        self.kronos = Kronos.from_pretrained(model_name)
        self.backbone_dim = getattr(self.kronos, "d_model", None)
        if self.backbone_dim is None:
            self.backbone_dim = getattr(getattr(self.kronos, "config", None), "hidden_size", 832)
        self.kronos_device = self._resolve_device(kronos_device)
        self._update_kronos_device(self.kronos_device)
        if self.freeze_kronos:
            self.kronos.eval()
            for param in self.kronos.parameters():
                param.requires_grad = False
            for param in self.tokenizer.parameters():
                param.requires_grad = False

    def _resolve_device(self, preferred: Optional[str]) -> torch.device:
        if preferred:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _update_kronos_device(self, device: torch.device) -> None:
        self.kronos.to(device)
        self.tokenizer.to(device)
        self.kronos_device = torch.device(device)

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        self._update_kronos_device(self.kronos_device)
        return self

    def _denormalize(self, asset: str, seq: torch.Tensor) -> torch.Tensor:
        asset = asset.upper()
        if asset not in self.price_stats:
            raise KeyError(f"Missing price stats for asset '{asset}'.")
        mean_name, std_name = self.price_stats[asset]
        mean = getattr(self, mean_name).to(seq.device)
        std = getattr(self, std_name).to(seq.device)
        return seq * std + mean

    def _prepare_time_features(
        self,
        time_features: Optional[torch.Tensor],
        batch: int,
        length: int,
    ) -> torch.Tensor:
        if torch.is_tensor(time_features):
            tensor = time_features
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.size(0) not in {1, batch}:
                raise ValueError("time_features batch dimension mismatch.")
            if tensor.size(0) == 1 and batch > 1:
                tensor = tensor.expand(batch, -1, -1)
            if tensor.size(1) < length:
                pad = tensor.new_zeros(tensor.size(0), length - tensor.size(1), tensor.size(2))
                tensor = torch.cat([pad, tensor], dim=1)
            elif tensor.size(1) > length:
                tensor = tensor[:, -length:, :]
            return tensor.to(self.kronos_device)
        cached = self._time_cache.get(length)
        if cached is None:
            base = torch.arange(length, dtype=torch.float32)
            minutes = base % 60
            hours = torch.floor(base / 60) % 24
            weekdays = base % 5
            days = (base % 28) + 1
            months = (torch.floor(base / 28) % 12) + 1
            cached = torch.stack([minutes, hours, weekdays, days, months], dim=-1)
            self._time_cache[length] = cached
        cached = cached.to(self.kronos_device)
        if batch == 1:
            return cached.unsqueeze(0)
        return cached.unsqueeze(0).expand(batch, -1, -1).contiguous()

    def _encode_chunk(self, seq: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        manager = nullcontext() if not self.freeze_kronos else torch.no_grad()
        with manager:
            tokens = self.tokenizer.encode(seq, half=True)
            pre, post = tokens
            _, context = self.kronos.decode_s1(pre, post, stamp=time_features)
        return context

    def _encode_sequence(self, seq: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        if seq.size(1) <= self.max_context:
            return self._encode_chunk(seq, time_features)
        pieces = []
        start = 0
        while start < seq.size(1):
            end = min(seq.size(1), start + self.max_context)
            chunk_seq = seq[:, start:end, :]
            chunk_time = time_features[:, start:end, :]
            pieces.append(self._encode_chunk(chunk_seq, chunk_time))
            start = end
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        price_seq: torch.Tensor,
        *,
        asset: str,
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if price_seq is None or price_seq.numel() == 0:
            device = self.bridge.weight.device
            batch = price_seq.size(0) if price_seq is not None else 1
            return torch.zeros(batch, 0, self.hidden_dim, device=device)
        asset_key = asset.upper()
        denorm = self._denormalize(asset_key, price_seq)
        seq_len = denorm.size(1)
        batch = denorm.size(0)
        kronos_ready = denorm.to(self.kronos_device)
        time_tensor = self._prepare_time_features(time_features, batch, seq_len)
        context = self._encode_sequence(kronos_ready, time_tensor)
        context = context.to(self.bridge.weight.device)
        out = self.bridge(context)
        if self.stopgrad:
            out = out.detach()
        return out
