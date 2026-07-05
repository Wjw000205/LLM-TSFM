# LLM-Guided Dataset-Aware Time-Series Forecasting

This repository implements a PyTorch framework for:

`LLM-Guided Dataset-Aware Loss and Feature Mining for Long-Tail Time Series Forecasting`

This is not an LLM-as-forecaster project. The LLM does not make point-wise predictions, does not run inside model `forward`, does not run for every batch, and is not called at test-time inference. The LLM is used once before training to produce a dataset-level rule JSON from train-split evidence.

## Core Contribution

- Dataset-level characteristic mining.
- Long-tail event rule extraction.
- Rule-based feature injection from precomputed future timestamps.
- Dataset-aware losses for event windows.
- Event-window evaluation beyond overall MSE/MAE.

The backbone is not the core contribution. DLinear is fully runnable; GRU/LSTM are lightweight alternatives; PatchTST, iTransformer, and TimesNet remain extension placeholders.

## Data Format

CSV files should use:

```text
date, feature_1, feature_2, ..., target
```

Put datasets under `./data/`, for example:

```text
data/ETTm1.csv
data/ETTh1.csv
data/traffic.csv
```

ETT datasets always use the fixed 12/4/4 month split. If an ETT file is too short, the loader raises an error instead of falling back to 7:1:2. Other datasets use chronological 7:1:2 splitting. Standardization is fit on the train split only.

## Rule JSON

LLM rules live in `llm_rules/example_rules/`. `calendar_periodic` rules should include an `anchor`:

```json
{
  "kind": "calendar_periodic",
  "anchor": "2016-07-01 00:00:00",
  "month_interval": 2,
  "day": 1
}
```

The anchor prevents month-periodic masks from silently assuming January as the start of the cycle.

## Feature Categories

The framework separates three feature families:

- `--use_standard_time_features`: ordinary calendar features such as hour, weekday, day, month, first-day indicator.
- `--use_llm_rule_features`: features directly generated from LLM rule JSON, such as event masks, peak masks, days to event, and distance to peak.
- `--use_oracle_features`: manual/oracle rule features for ablation only.

The legacy `--use_llm_features` is treated as a compatibility alias for LLM rule features.

## Losses

`DatasetAwareLoss` supports:

- `event_weighted_mse`
- `zero_consistency`
- `peak_shape`
- `diff`
- `frequency`

When z-score is enabled, zero consistency targets the scaled value of raw-space zero per target channel:

```text
zero_target = (0 - train_mean) / train_std
```

This avoids pulling predictions toward standardized-space zero.

## Future Rule Use

Two optional future-rule modes are available:

- `--use_rule_adapter`: applies a small MLP correction using known future rule features and event masks.
- `--use_hard_intervention`: forces deterministic zero-rule positions to the scaled zero target. This is an oracle-like intervention and should be reported separately.

Neither mode calls an LLM.

## Run ETTm1

Pure DLinear baseline aligned to common ETTm1 settings:

```bash
bash scripts/run_ettm1_dlinear_baseline.sh
```

This uses `seq_len=336`, `batch_size=8`, `learning_rate=0.0001`, z-score, MSE, and disables RevIN, LLM rule features, dataset-aware losses, rule adapter, and hard intervention.

```bash
python main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model DLinear \
  --data ETTm1 \
  --root_path ./data/ \
  --data_path ETTm1.csv \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --use_zscore 1 \
  --use_revin 1 \
  --use_llm_rule_features 1 \
  --use_dataset_aware_loss 1 \
  --llm_rule_path ./llm_rules/example_rules/ETTm1_rules.json
```

Or:

```bash
bash scripts/run_ettm1.sh
```

## Run Traffic

```bash
bash scripts/run_traffic.sh
```

## Run ETTh1 Baseline

Pure DLinear baseline aligned to common ETTh1 settings:

```bash
bash scripts/run_etth1_dlinear_baseline.sh
```

This uses `seq_len=336`, `batch_size=32`, `learning_rate=0.005`, z-score, MSE, and disables all non-baseline modules.

## Ablations

```bash
bash scripts/run_ablation.sh
```

Dataset-specific DLinear ablations:

```bash
bash scripts/run_ettm1_dlinear_ablation.sh
bash scripts/run_etth1_dlinear_ablation.sh
```

The ETTm1 script covers:

- Pure DLinear, `seq_len=336`
- Pure DLinear, `seq_len=96`
- DLinear + RevIN
- DLinear + standard time features
- DLinear + LLM rule features
- DLinear + dataset-aware loss
- DLinear + LLM rule features + dataset-aware loss
- DLinear + RevIN + LLM rule features + dataset-aware loss
- DLinear + rule adapter
- DLinear + hard intervention

The ETTh1 script only covers rule-free ablations until an ETTh1-specific rule JSON is generated:

- Pure DLinear, `seq_len=336`
- Pure DLinear, `seq_len=96`
- DLinear + RevIN
- DLinear + standard time features

Do not use `ETTm1_rules.json` as an ETTh1 full-method substitute.

## Train-Only LLM Mining

Use `llm_miner/` to create train-only summaries and prompts:

```bash
python -m llm_miner.build_dataset_summary --root_path ./data/ --data_path ETTm1.csv --data ETTm1 --target OT --seq_len 96 --output_path llm_miner/outputs/ETTm1_summary.json
python -m llm_miner.build_llm_prompt --summary_path llm_miner/outputs/ETTm1_summary.json --output_path llm_miner/outputs/ETTm1_prompt.txt
```

Validation and test rows must not be used for rule mining.

## Outputs

Test outputs are saved under `results/<setting>/`:

- `pred.npy` and `true.npy`: original-scale predictions and labels.
- `pred_normalized.npy` and `true_normalized.npy`: normalized-space predictions and labels.
- `metrics_original_scale.json`: main paper/table metrics.
- `metrics_normalized.json`: debugging metrics.
- `event_metrics.json`: event-window metrics in both normalized and original-scale spaces.
- `setting.txt`: run arguments.

Paper tables should use original-scale metrics unless explicitly stated otherwise. Event-window metrics are central for evaluating long-tail behavior.

For DLinear baseline comparison with most LTSF codebases, use `inverse=0` and read `metrics_normalized.json` or the printed `metric_space=normalized` metrics. `metrics_original_scale.json` is saved for original-unit diagnostics.

## Debugging Note: Why Metrics Can Look Larger Than Baseline

Do not compare a full-method run directly to pure DLinear. A run with `seq_len=96`, `use_revin=1`, `use_llm_rule_features=1`, and `use_dataset_aware_loss=1` changes the input dimension, objective, validation loss, and sometimes metric scale. Use the pure DLinear scripts first, then the ablation scripts to isolate whether degradation comes from sequence length, learning rate, RevIN, LLM rule features, or dataset-aware loss.

The current ETTh1 diagnosis is recorded in `docs/experiment_conclusions.md`.
