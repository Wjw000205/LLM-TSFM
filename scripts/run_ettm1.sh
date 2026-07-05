#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model DLinear \
  --data ETTm1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --c_out 7 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --use_zscore 1 \
  --use_revin 1 \
  --use_llm_features 1 \
  --use_dataset_aware_loss 1 \
  --llm_rule_path ./llm_rules/example_rules/ETTm1_rules.json

