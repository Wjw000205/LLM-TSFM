#!/usr/bin/env bash
set -euo pipefail

# Every dataset must get its own LLM analysis before training. This script
# intentionally never falls back to the ETTm1 example rule file.

DATASETS=${DATASETS:-"ETTm1 ETTh1 ETTh2 ETTm2"}
ROOT_PATH=${ROOT_PATH:-"./data/"}
FEATURES=${FEATURES:-"M"}
TARGET=${TARGET:-"OT"}
SEQ_LEN=${SEQ_LEN:-336}
LABEL_LEN=${LABEL_LEN:-48}
PRED_LEN=${PRED_LEN:-96}
BATCH_SIZE=${BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-10}
PATIENCE=${PATIENCE:-3}
OPENAI_BASE_URL=${OPENAI_BASE_URL:-"https://api.ruikon.com/v1"}
OPENAI_MODEL=${OPENAI_MODEL:-"gpt-5.5"}
OPENAI_API_KEY_ENV=${OPENAI_API_KEY_ENV:-"OPENAI_API_KEY"}

for DATA in ${DATASETS}; do
  LOWER=$(printf "%s" "${DATA}" | tr "[:upper:]" "[:lower:]")
  DATA_PATH="${DATA}.csv"
  RULE_PATH="./llm_rules/generated_rules/${DATA}_rules.json"
  RULE_REPORT="./artifacts/core_results/${LOWER}_llm_rule_generation_report.json"
  FREQ="h"
  if [[ "${DATA}" == ETTm* ]]; then
    FREQ="t"
  fi

  python analysis/generate_dataset_llm_rules.py \
    --data "${DATA}" \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --features "${FEATURES}" \
    --target "${TARGET}" \
    --seq_len "${SEQ_LEN}" \
    --profile_split train \
    --base_url "${OPENAI_BASE_URL}" \
    --model "${OPENAI_MODEL}" \
    --api_key_env "${OPENAI_API_KEY_ENV}" \
    --output_rule_path "${RULE_PATH}" \
    --output_report_path "${RULE_REPORT}"

  COMMON_ARGS=(
    --task_name long_term_forecast
    --is_training 1
    --model DLinear
    --data "${DATA}"
    --root_path "${ROOT_PATH}"
    --data_path "${DATA_PATH}"
    --features "${FEATURES}"
    --target "${TARGET}"
    --freq "${FREQ}"
    --seq_len "${SEQ_LEN}"
    --label_len "${LABEL_LEN}"
    --pred_len "${PRED_LEN}"
    --enc_in 7
    --c_out 7
    --batch_size "${BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --train_epochs "${TRAIN_EPOCHS}"
    --patience "${PATIENCE}"
    --dlinear_init_avg 0
    --use_zscore 1
    --use_revin 0
    --use_standard_time_features 0
    --use_oracle_features 0
    --use_rule_adapter 0
    --use_intervention_layer 0
    --use_hard_intervention 0
    --inverse 0
    --llm_rule_path "${RULE_PATH}"
  )

  BASELINE_DES="${LOWER}_rulegate_baseline"
  EVENT_DES="${LOWER}_rulegate_generated_loss"
  BASELINE_SETTING="long_term_forecast_DLinear_${DATA}_ft${FEATURES}_sl${SEQ_LEN}_ll${LABEL_LEN}_pl${PRED_LEN}_${BASELINE_DES}_0"
  EVENT_SETTING="long_term_forecast_DLinear_${DATA}_ft${FEATURES}_sl${SEQ_LEN}_ll${LABEL_LEN}_pl${PRED_LEN}_${EVENT_DES}_0"

  python main.py "${COMMON_ARGS[@]}" \
    --early_stop_metric base_mse \
    --use_llm_features 0 \
    --use_llm_rule_features 0 \
    --use_dataset_aware_loss 0 \
    --use_event_weighted_loss 0 \
    --use_zero_consistency_loss 0 \
    --use_peak_shape_loss 0 \
    --use_diff_loss 0 \
    --use_freq_loss 0 \
    --des "${BASELINE_DES}"

  # Use the LLM-generated loss config from the rule JSON.
  python main.py "${COMMON_ARGS[@]}" \
    --early_stop_metric base_mse \
    --selection_metric guarded_event_mse \
    --overall_mse_tolerance 0.03 \
    --baseline_metric_path "./checkpoints/${BASELINE_SETTING}/validation_history.json" \
    --use_llm_features 0 \
    --use_llm_rule_features 0 \
    --use_dataset_aware_loss 1 \
    --des "${EVENT_DES}"
done
