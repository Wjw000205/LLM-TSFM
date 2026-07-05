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
  --use_oracle_features 0
  --use_rule_adapter 0
  --inverse 0
  --llm_rule_path ./llm_rules/example_rules/ETTm1_rules.json
)

python main.py "${COMMON_ARGS[@]}" --early_stop_metric base_mse --use_standard_time_features 0 --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 0 --use_event_weighted_loss 0 --use_zero_consistency_loss 0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_hard_intervention 0 --des ettm1_core_pure_dlinear
python main.py "${COMMON_ARGS[@]}" --early_stop_metric base_mse --use_standard_time_features 1 --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 0 --use_event_weighted_loss 0 --use_zero_consistency_loss 0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_hard_intervention 0 --des ettm1_core_standard_time_features
python main.py "${COMMON_ARGS[@]}" --early_stop_metric base_mse --use_standard_time_features 0 --use_llm_features 0 --use_llm_rule_features 1 --use_dataset_aware_loss 0 --use_event_weighted_loss 0 --use_zero_consistency_loss 0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_hard_intervention 0 --des ettm1_core_llm_rule_features
python main.py "${COMMON_ARGS[@]}" --early_stop_metric total_loss --use_standard_time_features 0 --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 1 --use_event_weighted_loss 1 --use_zero_consistency_loss 1 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_hard_intervention 0 --des ettm1_core_dataset_aware_loss
python main.py "${COMMON_ARGS[@]}" --early_stop_metric total_loss --use_standard_time_features 0 --use_llm_features 0 --use_llm_rule_features 1 --use_dataset_aware_loss 1 --use_event_weighted_loss 1 --use_zero_consistency_loss 1 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_hard_intervention 0 --des ettm1_core_llm_rule_features_plus_loss
python main.py "${COMMON_ARGS[@]}" --early_stop_metric total_loss --use_standard_time_features 1 --use_llm_features 0 --use_llm_rule_features 1 --use_dataset_aware_loss 1 --use_event_weighted_loss 1 --use_zero_consistency_loss 1 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_hard_intervention 0 --des ettm1_core_standard_time_plus_llm_rule_plus_loss
python main.py "${COMMON_ARGS[@]}" --early_stop_metric base_mse --use_standard_time_features 0 --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 0 --use_event_weighted_loss 0 --use_zero_consistency_loss 0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_hard_intervention 1 --des ettm1_core_hard_intervention_oracle

