# LumoStockGPT minimal inference subset

This folder contains only the files needed to run `docs/run134_inference_commands.md` end-to-end.
Data, logs, and model outputs are intentionally excluded.

## File map
- `docs/run134_inference_commands.md`: Runbook for refreshing prices, building sequences, dumping signals, and evaluating run133 portfolio checkpoints.
- `scripts/fetch_data.py`: Yahoo price fetch utilities used in step 1 of the runbook.
- `scripts/build_run101_price_only_batch.py`: Builds price-only run101 sequences; used in step 2.
- `scripts/build_sequence_dataset.py`: Core sequence builder used by the run101 batch script.
- `scripts/build_run100_fullseq.py`: Converts run101 JSONL into full-sequence PT files.
- `scripts/build_run102_fullseq_auto.py`: Orchestrates fullseq build and writes the manifest used in step 3.
- `scripts/dump_run125_signals.py`: Dumps per-ticker mu/targets/masks from run123 (step 4).
- `scripts/run125_alpha_policy.py`: Alpha utilities and `compute_mu` used by `dump_run125_signals.py`.
- `scripts/eval_run130_policy.py`: Evaluates run133 portfolio checkpoints (step 5).
- `scripts/run129_train_portfolio.py`: Cross-sectional policy code used by the run133 evaluator.
- `ingest/__init__.py`: Package marker for local ingest helpers.
- `ingest/local_clients.py`: Local CSV/JSONL loaders used by the sequence builder.
- `models/__init__.py`: Package marker for model components.
- `models/factor_cross_attention.py`: Factor cross-attention block used by the Run100 model.
- `models/cross_ticker_attention.py`: Cross-ticker attention block used by the Run100 model.
- `models/kronos_mini_fullseq.py`: Kronos-mini price encoder wrapper used by the Run100 model.
- `models/price_encoder_with_mlp.py`: Numeric MLP + Kronos fusion encoder used by the Run100 model.
- `models/time_llm.py`: Causal temporal encoder used by the Run100 model.
- `train/__init__.py`: Package marker for training utilities.
- `train/multichannel_dataset.py`: Shared helpers and loaders used during fullseq build.
- `train/run101_multi_dataset.py`: Aligned multi-ticker dataset used to dump signals.
- `train/run100_model.py`: Run100 model definition used for inference.
- `train/fullseq_fusion.py`: News pooling and price-news fusion layers used by the Run100 model.
- `requirements_llm.txt`: Python dependency list (adjust to your environment).
- `third_party/Kronos/`: Trimmed Kronos model code needed for inference; see `third_party/Kronos/README.md` and `third_party/Kronos/LICENSE`.
# alphapolicy
