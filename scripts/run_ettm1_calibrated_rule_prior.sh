#!/usr/bin/env bash
set -euo pipefail

BASELINE_SETTING="long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl96_ettm1_rule_prior_pure_dlinear_0"
BASELINE_CHECKPOINT="./checkpoints/${BASELINE_SETTING}/checkpoint.pth"
BASELINE_RESULT_DIR="./results/${BASELINE_SETTING}"
BASELINE_METRICS="./checkpoints/${BASELINE_SETTING}/config.json"
VALIDATED_RULE_PATH="./llm_rules/validated_rules/ETTm1_rules_calibrated.json"
CALIBRATION_REPORT="./artifacts/core_results/ettm1_rule_calibration_report.json"

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
  --use_oracle_features 0
  --inverse 0
  --llm_rule_path ./llm_rules/example_rules/ETTm1_rules.json
  --early_stop_metric base_mse
)

python analysis/verify_and_calibrate_rules.py \
  --data ETTm1 \
  --root_path ./data/ \
  --data_path ETTm1.csv \
  --features M \
  --target OT \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 96 \
  --llm_rule_path ./llm_rules/example_rules/ETTm1_rules.json \
  --baseline_checkpoint "${BASELINE_CHECKPOINT}" \
  --baseline_result_dir "${BASELINE_RESULT_DIR}" \
  --calibration_split val \
  --output_rule_path "${VALIDATED_RULE_PATH}" \
  --output_report_path "${CALIBRATION_REPORT}" \
  --min_prior_improvement 0.01 \
  --near_zero_quantile 0.05 \
  --min_event_precision 0.5

python main.py "${COMMON_ARGS[@]}" --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 0 --use_event_weighted_loss 0 --use_zero_consistency_loss 0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_rule_adapter 0 --use_intervention_layer 0 --use_rule_prior_fusion 0 --use_hard_intervention 0 --des ettm1_calibrated_pure_dlinear
python main.py "${COMMON_ARGS[@]}" --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 0 --use_event_weighted_loss 0 --use_zero_consistency_loss 0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_rule_adapter 0 --use_intervention_layer 0 --use_rule_prior_fusion 1 --rule_prior_mode fixed --rule_prior_alpha 0.25 --use_hard_intervention 0 --des ettm1_calibrated_fixed_zero_prior_alpha_025
python main.py "${COMMON_ARGS[@]}" --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 0 --use_event_weighted_loss 0 --use_zero_consistency_loss 0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_rule_adapter 0 --use_intervention_layer 0 --use_rule_prior_fusion 1 --rule_prior_mode fixed --rule_prior_alpha 0.5 --use_hard_intervention 0 --des ettm1_calibrated_fixed_zero_prior_alpha_05
python main.py "${COMMON_ARGS[@]}" --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 0 --use_event_weighted_loss 0 --use_zero_consistency_loss 0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_rule_adapter 0 --use_intervention_layer 0 --use_rule_prior_fusion 1 --rule_prior_mode calibrated --validated_rule_path "${VALIDATED_RULE_PATH}" --disable_invalid_rules 1 --use_hard_intervention 0 --des ettm1_calibrated_residual_prior
python main.py "${COMMON_ARGS[@]}" --early_stop_metric total_loss --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 1 --use_event_weighted_loss 1 --use_zero_consistency_loss 1 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_rule_adapter 0 --use_intervention_layer 0 --use_rule_prior_fusion 1 --rule_prior_mode calibrated --validated_rule_path "${VALIDATED_RULE_PATH}" --disable_invalid_rules 1 --use_hard_intervention 0 --des ettm1_calibrated_rule_prior_plus_dataset_aware_loss
python main.py "${COMMON_ARGS[@]}" --early_stop_metric base_mse --selection_metric guarded_event_mse --overall_mse_tolerance 0.03 --baseline_metric_path "${BASELINE_METRICS}" --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 1 --use_event_weighted_loss 1 --use_zero_consistency_loss 1 --use_nonevent_preservation_loss 1 --event_weight 1.0 --zero_weight 0.1 --nonevent_weight 1.0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_rule_adapter 0 --use_intervention_layer 0 --use_rule_prior_fusion 0 --use_hard_intervention 0 --des ettm1_calibrated_guarded_longtail_medium_weights
python main.py "${COMMON_ARGS[@]}" --use_llm_features 0 --use_llm_rule_features 0 --use_dataset_aware_loss 0 --use_event_weighted_loss 0 --use_zero_consistency_loss 0 --use_peak_shape_loss 0 --use_diff_loss 0 --use_freq_loss 0 --use_rule_adapter 0 --use_intervention_layer 0 --use_rule_prior_fusion 0 --use_hard_intervention 1 --des ettm1_calibrated_hard_intervention_diagnostic

python analysis/summarize_calibrated_rule_results.py \
  --filter ettm1_calibrated \
  --baseline_experiment ettm1_calibrated_pure_dlinear \
  --output_csv artifacts/core_results/ettm1_calibrated_rule_prior_summary.csv \
  --output_markdown docs/calibrated_rule_prior_results.md
