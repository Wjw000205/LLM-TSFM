# LLM-Guided Dataset-Aware Time-Series Forecasting

This project implements a PyTorch forecasting scaffold for:

`LLM-Guided Dataset-Aware Loss and Feature Mining for Long-Tail Time Series Forecasting`

The LLM is not a forecasting model. It is assumed to run before training and produce dataset-level JSON rules. Training and inference only use normal neural forecasting models plus deterministic masks/features derived from those rules.

## What Is Implemented

- CSV time-series dataset with chronological train/val/test splits.
- ETT fixed split and 7:1:2 split for other datasets.
- Train-only z-score scaling with inverse transform support.
- RevIN layer.
- Runnable DLinear backbone with trend/seasonal decomposition.
- Placeholders for PatchTST, iTransformer, and TimesNet.
- Offline LLM rule parser, event-mask generator, and auxiliary feature generator.
- Dataset-aware loss with event-weighted MSE, zero consistency, peak shape, difference, and frequency terms.
- Training, validation, early stopping, checkpointing, testing, and result saving.

## Data Format

CSV files should use:

```text
date, feature_1, feature_2, ..., target
```

The first column is parsed as time. The `--target` column is used for univariate or MS forecasting.

## Run ETTm1

Place `ETTm1.csv` at `./dataset/ETT-small/ETTm1.csv`, then run:

```bash
bash scripts/run_ettm1.sh
```

Equivalent config usage:

```bash
python main.py --config configs/ettm1.yaml
```

## Run Traffic

Place `traffic.csv` at `./dataset/traffic/traffic.csv`, then run:

```bash
bash scripts/run_traffic.sh
```

## Useful Ablations

```bash
bash scripts/run_ablation.sh
```

You can also toggle:

```text
--use_zscore
--use_revin
--use_llm_features
--use_dataset_aware_loss
--use_event_weighted_loss
--use_zero_consistency_loss
--use_peak_shape_loss
--use_diff_loss
--use_freq_loss
```

## Outputs

Test results are saved under `results/<setting>/`:

- `metrics.npy`
- `pred.npy`
- `true.npy`
- `event_metrics.json`
- `setting.txt`

## Notes

- No LLM API is called in training, forward passes, or inference.
- Rule JSON files in `llm_rules/example_rules/` are deterministic inputs.
- DLinear is the supported runnable backbone; the other model files are extension points.
