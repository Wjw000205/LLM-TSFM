#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --task_name long_term_forecast
  --is_training 1
  --model DLinear
  --data ETTm1
  --root_path ./data/
  --data_path ETTm1.csv
  --features M
  --target OT
  --seq_len 336
  --label_len 48
  --pred_len 96
  --enc_in 7
  --c_out 7
  --batch_size 8
  --learning_rate 0.0001
  --train_epochs 10
  --patience 3
  --dlinear_init_avg 0
  --use_zscore 1
  --use_revin 0
  --use_standard_time_features 0
  --use_llm_features 0
  --use_llm_rule_features 0
  --use_oracle_features 0
  --use_rule_adapter 0
  --use_hard_intervention 0
  --inverse 0
  --llm_rule_path ./llm_rules/example_rules/ETTm1_rules.json
)

BASELINE_SETTING="long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl96_ettm1_core_pure_dlinear_0"
BASELINE_METRICS="./results/${BASELINE_SETTING}/metrics_normalized.json"

python main.py "${COMMON_ARGS[@]}" --early_stop_metric base_mse --selection_metric guarded_event_mse --overall_mse_tolerance 0.05 --baseline_metric_path "${BASELINE_METRICS}" --use_dataset_aware_loss 1 --use_event_weighted_loss 1 --use_zero_consistency_loss 1 --use_nonevent_preservation_loss 1 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --event_weight 1.0 --zero_weight 0.1 --nonevent_weight 1.0 --des ettm1_longtail_low_weights
python main.py "${COMMON_ARGS[@]}" --early_stop_metric base_mse --selection_metric guarded_event_mse --overall_mse_tolerance 0.05 --baseline_metric_path "${BASELINE_METRICS}" --use_dataset_aware_loss 1 --use_event_weighted_loss 1 --use_zero_consistency_loss 1 --use_nonevent_preservation_loss 1 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --event_weight 2.0 --zero_weight 0.25 --nonevent_weight 1.0 --des ettm1_longtail_medium_weights

python analysis/summarize_core_results.py \
  --filter ettm1_longtail \
  --output_csv artifacts/core_results/ettm1_weight_sweep.csv \
  --output_markdown docs/longtail_guardrail_results.md

python analysis/select_pareto_longtail.py \
  --baseline_metrics "${BASELINE_METRICS}" \
  --sweep_csv artifacts/core_results/ettm1_weight_sweep.csv \
  --output_csv artifacts/core_results/ettm1_pareto_longtail.csv \
  --output_markdown docs/longtail_guardrail_results.md \
  --overall_tolerance 0.05
