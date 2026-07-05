# DLinear ETT Diagnosis Conclusions

## Scope

This note records the diagnosis for the large metric gap observed in the earlier DLinear run.

## What Actually Ran

The earlier full-training result was ETTh1, not ETTm1. The saved setting path was:

```text
results/long_term_forecast_DLinear_ETTh1_ftM_sl96_ll48_pl96_etth1_full_es_seq96_0/setting.txt
```

Key parameters from that run:

```text
data=ETTh1
data_path=ETTh1.csv
seq_len=96
label_len=48
pred_len=96
batch_size=32
learning_rate=0.0001
use_revin=1
use_llm_features=1
use_dataset_aware_loss=1
llm_rule_path=./llm_rules/example_rules/ETTm1_rules.json
enc_in=13
raw_input_dim=7
llm_feature_dim=6
c_out=7
```

That run was not comparable to a pure DLinear baseline because it changed sequence length, learning rate, input channels, normalization behavior, and objective.

## ETTh1 Baseline Re-run

A pure ETTh1 DLinear baseline was re-run after the implementation alignment commit. The script-equivalent run with
`train_epochs=10` and `patience=3` is saved at:

```text
results/long_term_forecast_DLinear_ETTh1_ftM_sl336_ll48_pl96_etth1_dlinear_baseline_confirm_0/setting.txt
```

The longer early-stopping probe with `train_epochs=100` and `patience=5` is saved at:

```text
results/long_term_forecast_DLinear_ETTh1_ftM_sl336_ll48_pl96_etth1_baseline_es100_rerun_0/setting.txt
```

The run used:

```text
seq_len=336
label_len=48
pred_len=96
batch_size=32
learning_rate=0.005
train_epochs=100
patience=5
early_stop_metric=base_mse
use_zscore=1
use_revin=0
use_llm_features=0
use_llm_rule_features=0
use_dataset_aware_loss=0
use_rule_adapter=0
use_hard_intervention=0
inverse=0
```

The data/model dimension audit confirms this was a pure DLinear input:

```text
raw_feature_dim=7
standard_time_feature_dim=0
llm_rule_feature_dim=0
oracle_feature_dim=0
enc_in=7
c_out=7
```

Early stopping selected epoch 2 and stopped after epoch 7:

```text
best val_base_mse_loss=0.693798 at epoch 2
test metric_space=normalized
mse=0.4283130
mae=0.4406984
rmse=0.6544563
```

The original-scale diagnostics saved in `metrics_original_scale.json` are:

```text
mse=9.8476877
mae=1.7188561
rmse=3.1381025
```

This is much better than the earlier full-method ETTh1 result:

```text
mse=0.8273810
mae=0.5476723
rmse=0.9096048
```

That earlier result should only be treated as a wrong-configuration example, not as a full-method conclusion.

## Baseline Diagnosis Table

| Dataset | Setting | seq_len | LR | BS | RevIN | LLM rule feat | DA loss | MSE(norm) | MAE(norm) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | old full wrong config | 96 | 0.0001 | 32 | 1 | 1 | 1 | 0.8273810 | 0.5476723 | Used `ETTm1_rules.json`; invalid as DLinear/full-method evidence. |
| ETTh1 | pure baseline script config | 336 | 0.005 | 32 | 0 | 0 | 0 | 0.4283130 | 0.4406984 | `train_epochs=10`, `patience=3`, MSE only. |
| ETTh1 | pure baseline longer early stop | 336 | 0.005 | 32 | 0 | 0 | 0 | 0.4283130 | 0.4406984 | `train_epochs=100`, `patience=5`; selected the same best epoch. |
| ETTh1 | pure baseline avg init | 336 | 0.005 | 32 | 0 | 0 | 0 | 0.4279724 | 0.4407349 | `dlinear_init_avg=1`; only a negligible change. |
| ETTh1 | pure baseline individual | 336 | 0.005 | 32 | 0 | 0 | 0 | 0.4577665 | 0.4598802 | `individual=1`; worse in this run. |
| ETTm1 | pure baseline | 336 | 0.0001 | 8 | 0 | 0 | 0 | 0.3070308 | 0.3528005 | Script-equivalent pure DLinear baseline. |
| ETTm1 | pure baseline seq_len=96 | 96 | 0.0001 | 8 | 0 | 0 | 0 | 0.3429720 | 0.3698014 | Isolates the effect of shorter context. |

Result directories:

```text
results/long_term_forecast_DLinear_ETTh1_ftM_sl96_ll48_pl96_etth1_full_es_seq96_0
results/long_term_forecast_DLinear_ETTh1_ftM_sl336_ll48_pl96_etth1_dlinear_baseline_confirm_0
results/long_term_forecast_DLinear_ETTh1_ftM_sl336_ll48_pl96_etth1_baseline_es100_rerun_0
results/long_term_forecast_DLinear_ETTh1_ftM_sl336_ll48_pl96_etth1_dlinear_initavg1_check_0
results/long_term_forecast_DLinear_ETTh1_ftM_sl336_ll48_pl96_etth1_dlinear_individual1_check_0
results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl96_ettm1_dlinear_baseline_confirm_0
results/long_term_forecast_DLinear_ETTm1_ftM_sl96_ll48_pl96_ettm1_dlinear_seq96_confirm_0
```

## Current Diagnosis

The large `mse=0.8273810` result is not primarily evidence of a code bug. It came from a mismatched experiment
configuration: ETTh1 data, `seq_len=96`, `learning_rate=0.0001`, RevIN enabled, LLM rule features enabled,
dataset-aware loss enabled, and `ETTm1_rules.json` used for ETTh1.

The clean pure-DLinear checks show:

1. The ETTh1 pure baseline recovers to normalized `mse=0.4283130`.
2. The ETTm1 pure baseline reaches normalized `mse=0.3070308`.
3. Reducing ETTm1 `seq_len` from 336 to 96 worsens MSE from `0.3070308` to `0.3429720`.
4. ETTh1 `dlinear_init_avg=1` has negligible impact in this setup.
5. ETTh1 `individual=1` is worse than shared linear layers in this setup.
6. For all pure baseline runs, `event_loss`, `zero_loss`, `peak_loss`, `diff_loss`, and `freq_loss` are zero, so the loss is base MSE only.

If there is still a gap to a specific original DLinear table, the most likely remaining sources are exact upstream
training protocol differences: seed/CUDNN determinism, optimizer schedule, number of repeats, `individual`, data file
revision, metric scale, or small preprocessing differences. Current comparisons should use `metrics_normalized.json`
when matching most LTSF/DLinear reports with `inverse=0`.

## Ablation Plan

ETTm1 is the clean dataset for LLM-rule/full-method experiments because the available rule file is
`llm_rules/example_rules/ETTm1_rules.json`. Run pure baseline first, then change one factor at a time:

```bash
bash scripts/run_ettm1_dlinear_baseline.sh
bash scripts/run_ettm1_dlinear_ablation.sh
```

For ETTh1, do not use `ETTm1_rules.json` as a substitute for full-method evidence. Until an ETTh1-specific rule JSON
is generated, ETTh1 should be limited to pure DLinear, `seq_len=96`, RevIN, and standard-time-feature ablations:

```bash
bash scripts/run_etth1_dlinear_baseline.sh
python main.py --task_name long_term_forecast --is_training 1 --model DLinear --data ETTh1 --root_path ./data/ --data_path ETTh1.csv --features M --target OT --seq_len 336 --label_len 48 --pred_len 96 --batch_size 32 --learning_rate 0.005 --train_epochs 10 --patience 3 --early_stop_metric base_mse --dlinear_init_avg 0 --use_zscore 1 --use_revin 1 --use_llm_features 0 --use_llm_rule_features 0 --use_standard_time_features 0 --use_oracle_features 0 --use_dataset_aware_loss 0 --use_rule_adapter 0 --use_hard_intervention 0 --inverse 0 --des etth1_revin_ablation
python main.py --task_name long_term_forecast --is_training 1 --model DLinear --data ETTh1 --root_path ./data/ --data_path ETTh1.csv --features M --target OT --seq_len 336 --label_len 48 --pred_len 96 --batch_size 32 --learning_rate 0.005 --train_epochs 10 --patience 3 --early_stop_metric base_mse --dlinear_init_avg 0 --use_zscore 1 --use_revin 0 --use_llm_features 0 --use_llm_rule_features 0 --use_standard_time_features 1 --use_oracle_features 0 --use_dataset_aware_loss 0 --use_rule_adapter 0 --use_hard_intervention 0 --inverse 0 --des etth1_standard_time_ablation
```

Generate or provide `ETTh1_rules.json` before running ETTh1 LLM-rule features, dataset-aware loss, rule adapter, or
hard-intervention ablations.

## Primary Cause

The metric gap is primarily a configuration and comparison issue, not evidence that DLinear cannot train:

1. The earlier run used ETTh1 while the expected comparison was assumed to be ETTm1.
2. The earlier run used `seq_len=96`, while common DLinear ETT settings use `seq_len=336`.
3. ETTh1 used `learning_rate=0.0001`, while the common DLinear ETTh1 setting is closer to `0.005`.
4. The earlier run enabled RevIN, LLM rule features, and dataset-aware loss, so it was not a pure DLinear baseline.
5. The validation loss in full-method mode is total dataset-aware loss, not base MSE.

## Fixes Added

- Pure DLinear baseline scripts:
  - `scripts/run_ettm1_dlinear_baseline.sh`
  - `scripts/run_etth1_dlinear_baseline.sh`
- Stepwise ablation scripts:
  - `scripts/run_ettm1_dlinear_ablation.sh`
  - `scripts/run_etth1_dlinear_ablation.sh`
- `--dlinear_init_avg` to control average-weight DLinear initialization.
- `--early_stop_metric base_mse|total_loss` to separate baseline early stopping from full-method early stopping.
- Complete run configuration saving to both `results/<setting>/config.json` and `checkpoints/<setting>/config.json`.
- Clear normalized/original metric outputs.
- Split feature dimension bookkeeping in the actual data pipeline:
  - `raw_feature_dim`
  - `standard_time_feature_dim`
  - `llm_rule_feature_dim`
  - `oracle_feature_dim`
- `pred.npy` and `true.npy` are now fixed original-scale aliases; normalized arrays are saved separately.
- DLinear ablation scripts now cover the 10 required combinations: pure baselines, RevIN, standard time features, LLM rule features, dataset-aware loss, combined feature/loss runs, rule adapter, and hard intervention.

## Recommended Next Runs

Run official-aligned baselines first:

```bash
bash scripts/run_ettm1_dlinear_baseline.sh
bash scripts/run_etth1_dlinear_baseline.sh
```

Then isolate the source of degradation:

```bash
bash scripts/run_ettm1_dlinear_ablation.sh
bash scripts/run_etth1_dlinear_ablation.sh
```

Use `metrics_normalized.json` for comparison with most DLinear/LTSF baseline reports when `inverse=0`.
