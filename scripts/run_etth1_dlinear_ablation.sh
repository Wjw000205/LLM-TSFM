#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --task_name long_term_forecast
  --is_training 1
  --model DLinear
  --data ETTh1
  --root_path ./data/
  --data_path ETTh1.csv
  --features M
  --target OT
  --label_len 48
  --pred_len 96
  --enc_in 7
  --c_out 7
  --batch_size 32
  --learning_rate 0.005
  --train_epochs 10
  --patience 3
  --early_stop_metric base_mse
  --dlinear_init_avg 0
  --use_zscore 1
  --inverse 0
)

python main.py "${COMMON_ARGS[@]}" --seq_len 336 --use_revin 0 --use_llm_features 0 --use_standard_time_features 0 --use_llm_rule_features 0 --use_oracle_features 0 --use_dataset_aware_loss 0 --use_rule_adapter 0 --use_hard_intervention 0 --des pure_dlinear_336
python main.py "${COMMON_ARGS[@]}" --seq_len 96 --use_revin 0 --use_llm_features 0 --use_standard_time_features 0 --use_llm_rule_features 0 --use_oracle_features 0 --use_dataset_aware_loss 0 --use_rule_adapter 0 --use_hard_intervention 0 --des pure_dlinear_96
python main.py "${COMMON_ARGS[@]}" --seq_len 336 --use_revin 1 --use_llm_features 0 --use_standard_time_features 0 --use_llm_rule_features 0 --use_oracle_features 0 --use_dataset_aware_loss 0 --use_rule_adapter 0 --use_hard_intervention 0 --des revin
python main.py "${COMMON_ARGS[@]}" --seq_len 336 --use_revin 0 --use_llm_features 0 --use_standard_time_features 1 --use_llm_rule_features 0 --use_oracle_features 0 --use_dataset_aware_loss 0 --use_rule_adapter 0 --use_hard_intervention 0 --des standard_time_features
