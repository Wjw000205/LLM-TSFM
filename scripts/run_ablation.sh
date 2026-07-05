#!/usr/bin/env bash
set -euo pipefail

for llm_features in 0 1; do
  for aware_loss in 0 1; do
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
      --batch_size 32 \
      --learning_rate 0.0001 \
      --train_epochs 10 \
      --use_zscore 1 \
      --use_revin 1 \
      --use_llm_features "${llm_features}" \
      --use_dataset_aware_loss "${aware_loss}" \
      --use_event_weighted_loss "${aware_loss}" \
      --use_zero_consistency_loss "${aware_loss}" \
      --use_peak_shape_loss 0 \
      --llm_rule_path ./llm_rules/example_rules/ETTm1_rules.json \
      --des "ablation_llmf${llm_features}_loss${aware_loss}"
  done
done

