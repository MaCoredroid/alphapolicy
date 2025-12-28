"""Run100 full-sequence model: Kronos-mini price encoder + news fusion + TimeLLM + factor cross-attn."""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.factor_cross_attention import FactorCrossAttention, FactorCrossAttentionConfig
from models.cross_ticker_attention import CrossTickerAttention, CrossTickerConfig
from models.kronos_mini_fullseq import KronosMiniPriceEncoderFullSeq
from models.price_encoder_with_mlp import PriceEncoderWithMLP
from models.time_llm import TimeLLM, TimeLLMConfig
from train.fullseq_fusion import DailyNewsPooler, GatedPriceNewsFusion
from train.multichannel_dataset import INTENSITY_FIELDS
import torch.utils.checkpoint as checkpoint


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class SharedOrdinalHead(nn.Module):
    """Shared ordinal head with learnable thresholds."""

    def __init__(self, hidden_dim: int, num_bins: int) -> None:
        super().__init__()
        self.adapter = ResidualAdapter(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.threshold_logits = nn.Parameter(torch.zeros(num_bins - 1))
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        # seq: [B, L, H]
        adapted = self.adapter(seq)
        scores = self.score(adapted).squeeze(-1)  # [B, L]
        thresholds = torch.cumsum(F.softplus(self.threshold_logits), dim=0)
        thresholds = thresholds - thresholds.mean()
        temp = torch.exp(self.log_temp).clamp_min(1e-3)
        # broadcast thresholds over time
        cdf = torch.sigmoid((thresholds.view(1, 1, -1) - scores.unsqueeze(-1)) / temp)
        cdf = torch.cat(
            [
                torch.zeros(cdf.size(0), cdf.size(1), 1, device=cdf.device),
                cdf,
                torch.ones(cdf.size(0), cdf.size(1), 1, device=cdf.device),
            ],
            dim=2,
        )
        probs = (cdf[:, :, 1:] - cdf[:, :, :-1]).clamp_min(1e-9)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return {"logits": torch.log(probs), "probs": probs, "raw_score": scores}


class ForwardTimer:
    """Lightweight timer that emits millisecond timings without per-op synchronizations."""

    def __init__(self, use_cuda: bool) -> None:
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.events = []
        self.cpu_times = []

    def start(self, label: str):
        if self.use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            return label, start
        return label, time.perf_counter()

    def stop(self, token) -> None:
        label, start = token
        if self.use_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events.append((label, start, end))
        else:
            self.cpu_times.append((label, (time.perf_counter() - start) * 1000.0))

    def summary_ms(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        # Avoid synchronizing while streams are being captured for CUDA graphs.
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return out
        if self.use_cuda:
            torch.cuda.synchronize()
            for label, start, end in self.events:
                out[label] = float(start.elapsed_time(end))
        else:
            for label, duration in self.cpu_times:
                out[label] = float(duration)
        return out


class Run100FullSeqModel(nn.Module):
    """End-to-end Run100 model."""

    def __init__(
        self,
        *,
        price_stats: Dict[str, tuple[torch.Tensor, torch.Tensor]],
        news_dim: int,
        bin_count: int,
        hidden_dim: int = 256,
        time_layers: int = 8,
        time_heads: int = 4,
        factor_cross_layers: int = 2,
        factor_cross_heads: int = 4,
        cross_ticker: bool = False,
        ticker_roles: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ticker_roles = ticker_roles or {}
        self.cross_ticker_enabled = cross_ticker
        price_dim = next(iter(price_stats.values()))[0].numel()
        kronos_encoder = KronosMiniPriceEncoderFullSeq(
            price_stats=price_stats,
            hidden_dim=hidden_dim,
        )
        self.price_encoder = PriceEncoderWithMLP(
            kronos_encoder=kronos_encoder,
            feat_dim=price_dim,
            hidden_dim=hidden_dim,
            freeze_kronos=True,
            mlp_dropout=0.1,
        )
        self.news_pool = DailyNewsPooler(
            doc_dim=news_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            intensity_dim=len(INTENSITY_FIELDS),
        )
        self.fusion = GatedPriceNewsFusion(hidden_dim=hidden_dim, aux_dim=0)
        self.time_llm = TimeLLM(
            TimeLLMConfig(
                d_model=hidden_dim,
                n_layers=time_layers,
                n_heads=time_heads,
                ff_mult=4,
                dropout=0.1,
            )
        )
        self.cross_attn = FactorCrossAttention(
            FactorCrossAttentionConfig(
                d_model=hidden_dim,
                n_heads=factor_cross_heads,
                n_layers=factor_cross_layers,
                ff_mult=4,
                dropout=0.1,
            )
        )
        self.cross_ticker = (
            CrossTickerAttention(
                CrossTickerConfig(d_model=hidden_dim, n_heads=factor_cross_heads, n_layers=factor_cross_layers)
            )
            if cross_ticker
            else None
        )
        self.head = SharedOrdinalHead(hidden_dim=hidden_dim, num_bins=bin_count)

    @staticmethod
    def _ensure_batch(seq: torch.Tensor) -> torch.Tensor:
        return seq if seq.dim() == 3 else seq.unsqueeze(0)

    @staticmethod
    def _ensure_news_emb_batch(tensor: torch.Tensor) -> torch.Tensor:
        # Expected [B, L, K, D]; if [L, K, D], add batch
        if tensor.dim() == 4:
            return tensor
        if tensor.dim() == 3:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _ensure_news_mask_batch(tensor: torch.Tensor) -> torch.Tensor:
        # Expected [B, L, K]; if [L, K], add batch
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _ensure_news_int_batch(tensor: torch.Tensor) -> torch.Tensor:
        # Expected [B, L, I]; if [L, I], add batch
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        return tensor

    def _build_factor_bank(
        self,
        factor_price: Dict[str, torch.Tensor],
        factor_news: Dict[str, Dict[str, torch.Tensor]],
        timer: Optional[ForwardTimer] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        factor_seqs = []
        factor_masks = []
        device = self.head.threshold_logits.device
        for asset, price_seq in factor_price.items():
            asset_name = str(asset)
            path_token = timer.start(f"price_path_{asset_name}_ms") if timer else None
            price_token = timer.start(f"price_encode_{asset_name}_ms") if timer else None
            price_seq = self._ensure_batch(price_seq)
            f_news = factor_news.get(asset, {})
            mask = f_news.get("trading_day_mask")
            if mask is None:
                mask = torch.ones(price_seq.size(0), price_seq.size(1), dtype=torch.bool, device=price_seq.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            mask = mask.to(price_seq.device)
            news_emb = self._ensure_news_emb_batch(f_news.get("news_emb", price_seq.new_zeros(price_seq.size(1), 1, self.hidden_dim)))
            news_mask = self._ensure_news_mask_batch(f_news.get("news_mask", price_seq.new_zeros(price_seq.size(1), 1, dtype=torch.bool)))
            news_intensity = self._ensure_news_int_batch(
                f_news.get("news_intensity", price_seq.new_zeros(price_seq.size(1), 1))
            )
            precomp = f_news.get("price_hidden")
            use_precomp = False
            if torch.is_tensor(precomp):
                use_precomp = precomp.dim() >= 2 and precomp.size(-1) == self.hidden_dim
            if use_precomp:
                price_hidden = self._ensure_batch(precomp)
            else:
                price_hidden, _ = self.price_encoder(price_seq, asset=asset, time_features=None, seq_mask=mask)
            if timer and price_token:
                timer.stop(price_token)
            news_token = timer.start(f"news_pool_{asset_name}_ms") if timer else None
            news_seq = self.news_pool(price_hidden, news_emb, news_mask, news_intensity)
            if timer and news_token:
                timer.stop(news_token)
            fusion_token = timer.start(f"memory_{asset_name}_ms") if timer else None
            fused = self.fusion(price_hidden, news_seq)
            if timer and fusion_token:
                timer.stop(fusion_token)
            mask = mask.to(device)
            time_token = timer.start(f"time_llm_{asset_name}_ms") if timer else None
            seq = self.time_llm(fused, key_padding_mask=mask)  # [1, L, H]
            if timer and time_token:
                timer.stop(time_token)
            if timer and path_token:
                timer.stop(path_token)
            factor_seqs.append(seq)
            factor_masks.append(mask)
        if not factor_seqs:
            return torch.zeros(1, 0, 0, self.hidden_dim, device=device), torch.zeros(1, 0, 0, dtype=torch.bool, device=device)
        factor_seq = torch.stack(factor_seqs, dim=0).squeeze(1).transpose(0, 1)  # [B, F, L, H]
        factor_mask = torch.stack(factor_masks, dim=0).squeeze(1).transpose(0, 1)  # [B, F, L]
        return factor_seq, factor_mask

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        if "price_seq_stack" in batch:
            return self.forward_multi(batch)
        asset = batch.get("asset")
        price_seq = self._ensure_batch(batch["price_seq"])
        news_emb = self._ensure_news_emb_batch(batch["news_emb"])
        news_mask = self._ensure_news_mask_batch(batch["news_mask"])
        news_intensity = self._ensure_news_int_batch(batch["news_intensity"])
        time_features = batch.get("context_time_features")
        seq_mask = batch.get("seq_mask")
        if torch.is_tensor(seq_mask):
            if seq_mask.dim() == 1:
                seq_mask = seq_mask.unsqueeze(0)
            seq_mask = seq_mask.to(device=price_seq.device, dtype=torch.bool)
        else:
            seq_mask = None

        timer = ForwardTimer(price_seq.is_cuda)
        total_token = timer.start("total_forward_ms")
        asset_name = str(asset[0] if isinstance(asset, (list, tuple)) and len(asset) > 0 else asset)
        if not isinstance(asset_name, str) or not asset_name:
            asset_name = "TARGET"

        path_token = timer.start(f"price_path_{asset_name}_ms")
        price_token = timer.start(f"price_encode_{asset_name}_ms")
        precomp = batch.get("price_hidden")
        price_gate = None
        if torch.is_tensor(precomp) and precomp.size(-1) == self.hidden_dim:
            price_hidden = self._ensure_batch(precomp)
        else:
            price_hidden, price_gate = self.price_encoder(
                price_seq, asset=str(asset_name), time_features=time_features, seq_mask=seq_mask
            )
        timer.stop(price_token)

        news_token = timer.start(f"news_pool_{asset_name}_ms")
        news_seq = self.news_pool(price_hidden, news_emb, news_mask, news_intensity)
        timer.stop(news_token)

        fusion_token = timer.start(f"memory_{asset_name}_ms")
        fused = self.fusion(price_hidden, news_seq)
        timer.stop(fusion_token)

        time_token = timer.start(f"time_llm_{asset_name}_ms")
        if seq_mask is not None:
            fused = fused * seq_mask.unsqueeze(-1)
        target_seq = self.time_llm(fused, key_padding_mask=seq_mask)
        if seq_mask is not None:
            target_seq = target_seq * seq_mask.unsqueeze(-1)
        timer.stop(time_token)
        timer.stop(path_token)

        factor_price = batch.get("factor_price", {})
        factor_news = batch.get("factor_news", {})
        factor_seq, factor_mask = self._build_factor_bank(factor_price, factor_news, timer=timer)

        # Flatten factors day-major: [B, F, L, H] -> [B, L*F, H], mask [B, L*F]
        if factor_seq.numel() > 0:
            if factor_seq.dim() == 3:
                factor_seq = factor_seq.unsqueeze(0)
                factor_mask = factor_mask.unsqueeze(0)
            B, F, L, H = factor_seq.shape
            factor_seq_flat = factor_seq.transpose(1, 2).reshape(B, L * F, H)
            factor_mask_flat = factor_mask.transpose(1, 2).reshape(B, L * F)
        else:
            factor_seq_flat = target_seq.new_zeros(target_seq.size(0), 0, target_seq.size(-1))
            factor_mask_flat = target_seq.new_zeros(target_seq.size(0), 0, dtype=torch.bool)

        L_q = target_seq.size(1)
        L_k = factor_seq_flat.size(1)
        if L_k == 0 or factor_seq_flat.numel() == 0:
            enhanced = target_seq
        else:
            factor_padding = ~factor_mask_flat if factor_mask_flat.numel() > 0 else None
            # Build causal mask so day t cannot see future factor tokens (allow day <= t)
            day_idx = torch.arange(factor_mask.size(2), device=target_seq.device).repeat_interleave(factor_mask.size(1))
            attn_mask = (day_idx.unsqueeze(0) > torch.arange(L_q, device=target_seq.device).unsqueeze(1))
            cross_token = timer.start("cross_asset_ms")
            enhanced = self.cross_attn(
                target_seq,
                factor_seq_flat,
                attn_mask=attn_mask,
                factor_padding_mask=factor_padding,
            )
            timer.stop(cross_token)

        head_token = timer.start("heads_ms")
        head_out = self.head(enhanced)
        timer.stop(head_token)
        timer.stop(total_token)

        timing = timer.summary_ms()
        path_keys = [k for k in timing.keys() if k.startswith("price_path_")]
        if path_keys:
            timing["price_news_mem_ms"] = float(sum(timing[k] for k in path_keys))
        head_out["timing"] = timing
        if price_gate is not None and price_gate.numel() > 0:
            head_out["price_gate_mean"] = price_gate.mean().detach()
        return head_out

    def forward_multi(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        price_hidden_stack = batch.get("price_hidden_stack")
        price_seq_stack = batch.get("price_seq_stack")
        ticker_order = batch.get("ticker_order")
        ticker_mask = batch.get("ticker_mask_stack")
        if not torch.is_tensor(price_hidden_stack) or price_hidden_stack.size(-1) != self.hidden_dim:
            if not (torch.is_tensor(price_seq_stack) and isinstance(ticker_order, (list, tuple))):
                raise ValueError("price_hidden_stack missing and price_seq_stack/ticker_order not provided.")
            # Recompute price_hidden for each ticker using price encoder
            ph_list = []
            for idx, ticker in enumerate(ticker_order):
                seq = price_seq_stack[idx]
                if seq.dim() == 2:
                    seq = seq.unsqueeze(0)
                seq_mask = None
                if torch.is_tensor(ticker_mask):
                    seq_mask = ticker_mask[idx]
                ph, _ = self.price_encoder(seq, asset=str(ticker), time_features=None, seq_mask=seq_mask)
                ph_list.append(ph.squeeze(0))
            price_hidden_stack = torch.stack(ph_list, dim=0)
        news_emb = batch["news_emb_stack"]
        news_mask = batch["news_mask_stack"]
        news_intensity = batch["news_intensity_stack"]
        target_idx = batch.get("target_indices")
        # Process each ticker sequentially to reduce peak memory in TimeLLM
        per_ticker_outputs = []
        for i in range(price_hidden_stack.size(0)):
            ph = price_hidden_stack[i].unsqueeze(0)  # [1, L, H]
            ne = news_emb[i].unsqueeze(0)            # [1, L, K, D]
            nm = news_mask[i].unsqueeze(0)           # [1, L, K]
            ni = news_intensity[i].unsqueeze(0)      # [1, L, I]
            seq_mask = None
            if torch.is_tensor(ticker_mask):
                seq_mask = ticker_mask[i].unsqueeze(0).to(ph.device)
            news_seq = self.news_pool(ph, ne, nm, ni)
            fused = self.fusion(ph, news_seq)
            if seq_mask is not None:
                fused = fused * seq_mask.unsqueeze(-1)
            # Checkpoint TimeLLM to reduce activation memory across many tickers
            if seq_mask is not None:
                out_t = checkpoint.checkpoint(lambda x, m: self.time_llm(x, key_padding_mask=m), fused, seq_mask)
            else:
                out_t = checkpoint.checkpoint(lambda x: self.time_llm(x), fused)
            if seq_mask is not None:
                out_t = out_t * seq_mask.unsqueeze(-1)
            per_ticker_outputs.append(out_t.squeeze(0))  # [L, H]
        if not per_ticker_outputs:
            raise ValueError("No ticker outputs computed in forward_multi.")
        time_out = torch.stack(per_ticker_outputs, dim=0)  # [N, L, H]
        # reshape to [1, L, N, H] for cross-ticker block
        time_out = time_out.unsqueeze(0)  # [1, N, L, H]
        time_out = time_out.transpose(1, 2)  # [1, L, N, H]
        if self.cross_ticker is not None and self.cross_ticker_enabled:
            # Log cross-attn input shape (per time across N tickers)
            print(f"[mem][cross_attn_input] T={time_out.size(1)} N={time_out.size(2)} H={time_out.size(3)}")
            time_out = self.cross_ticker(time_out, ticker_mask=ticker_mask)
        if target_idx is not None and torch.is_tensor(target_idx):
            time_out = time_out[:, :, target_idx, :]
        if time_out.dim() == 4 and time_out.size(0) == 1:
            time_out = time_out.squeeze(0)  # [L, N, H]
        if time_out.dim() == 3:
            time_out = time_out.permute(1, 0, 2)  # [N, L, H]
        head_out = self.head(time_out)
        return head_out
