#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model DLinear \
  --data Traffic \
  --root_path ./data/ \
  --data_path traffic.csv \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --early_stop_metric total_loss \
  --dlinear_init_avg 0 \
  --use_zscore 1 \
  --use_revin 1 \
  --use_llm_features 0 \
  --use_standard_time_features 0 \
  --use_llm_rule_features 1 \
  --use_oracle_features 0 \
  --use_rule_adapter 0 \
  --use_hard_intervention 0 \
  --use_dataset_aware_loss 1 \
  --llm_rule_path ./llm_rules/example_rules/Traffic_rules.json
