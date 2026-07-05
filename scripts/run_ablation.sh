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
  --seq_len 96
  --label_len 48
  --pred_len 96
  --batch_size 32
  --learning_rate 0.0001
  --train_epochs 10
  --use_gpu 0
  --llm_rule_path ./llm_rules/example_rules/ETTm1_rules.json
)

python main.py "${COMMON_ARGS[@]}" --use_zscore 0 --use_revin 0 --use_dataset_aware_loss 0 --des base_mse
python main.py "${COMMON_ARGS[@]}" --use_standard_time_features 1 --use_dataset_aware_loss 0 --des standard_time
python main.py "${COMMON_ARGS[@]}" --use_llm_rule_features 1 --use_dataset_aware_loss 0 --des llm_rule_features
python main.py "${COMMON_ARGS[@]}" --use_dataset_aware_loss 1 --des llm_rule_loss
python main.py "${COMMON_ARGS[@]}" --use_llm_rule_features 1 --use_dataset_aware_loss 1 --des llm_features_loss
python main.py "${COMMON_ARGS[@]}" --use_llm_rule_features 1 --use_rule_adapter 1 --use_dataset_aware_loss 0 --des rule_adapter
python main.py "${COMMON_ARGS[@]}" --use_hard_intervention 1 --use_dataset_aware_loss 0 --des hard_intervention
python main.py "${COMMON_ARGS[@]}" --use_oracle_features 1 --use_dataset_aware_loss 0 --des oracle_rules
python main.py "${COMMON_ARGS[@]}" --use_zscore 1 --use_revin 0 --use_dataset_aware_loss 0 --des zscore_only
python main.py "${COMMON_ARGS[@]}" --use_zscore 0 --use_revin 1 --use_dataset_aware_loss 0 --des revin_only
python main.py "${COMMON_ARGS[@]}" --use_zscore 1 --use_revin 1 --use_dataset_aware_loss 0 --des zscore_revin
