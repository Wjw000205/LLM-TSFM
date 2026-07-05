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

## Baseline Check

A pure ETTh1 DLinear baseline smoke run was executed with:

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

Early stopping selected epoch 2 and stopped at epoch 7:

```text
best val_base_mse_loss=0.693798 at epoch 2
test metric_space=normalized
mse=0.4283130
mae=0.4406984
rmse=0.6544563
```

This is much better than the earlier full-method ETTh1 result:

```text
mse=0.8273810
mae=0.5476723
rmse=0.9096048
```

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
