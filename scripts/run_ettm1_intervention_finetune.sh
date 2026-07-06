#!/usr/bin/env bash
set -euo pipefail

BASELINE_SETTING="long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl96_ettm1_rule_prior_pure_dlinear_0"
BASELINE_CHECKPOINT="./checkpoints/${BASELINE_SETTING}/checkpoint.pth"
BASELINE_METRICS="./checkpoints/${BASELINE_SETTING}/config.json"

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
  --learning_rate 0.0005
  --train_epochs 20
  --patience 5
  --dlinear_init_avg 0
  --use_zscore 1
  --use_revin 0
  --use_standard_time_features 0
  --use_oracle_features 0
  --inverse 0
  --llm_rule_path ./llm_rules/example_rules/ETTm1_rules.json
  --load_pretrained_checkpoint "${BASELINE_CHECKPOINT}"
  --baseline_checkpoint "${BASELINE_CHECKPOINT}"
  --baseline_metric_path "${BASELINE_METRICS}"
  --early_stop_metric base_mse
  --selection_metric guarded_event_mse
  --overall_mse_tolerance 0.03
)

python main.py "${COMMON_ARGS[@]}" \
  --use_llm_features 0 \
  --use_llm_rule_features 1 \
  --use_dataset_aware_loss 1 \
  --use_event_weighted_loss 1 \
  --event_weight 1.0 \
  --use_zero_consistency_loss 0 \
  --zero_weight 0.0 \
  --use_peak_shape_loss 0 \
  --use_diff_loss 0 \
  --use_freq_loss 0 \
  --use_nonevent_preservation_loss 0 \
  --use_baseline_distillation 1 \
  --distill_weight 1.0 \
  --use_rule_adapter 0 \
  --use_intervention_layer 1 \
  --intervention_hidden 32 \
  --intervention_dropout 0.0 \
  --intervention_scale 1.0 \
  --intervention_init_zero 1 \
  --use_intervention_reg 0 \
  --intervention_reg_weight 0.0 \
  --train_only_intervention 1 \
  --use_rule_prior_fusion 0 \
  --use_hard_intervention 0 \
  --des ettm1_intervention_finetune_event_only

python analysis/summarize_intervention_results.py \
  --filter ettm1_intervention \
  --baseline_experiment ettm1_intervention_pure_dlinear \
  --output_csv artifacts/core_results/ettm1_intervention_summary.csv \
  --output_markdown docs/intervention_results.md
