# LLM Feature False Positive Diagnosis

- Dataset: `ETTm1`.
- Rule path: `./llm_rules/example_rules/ETTm1_rules.json`.
- Near-zero threshold source: `train` quantile `0.05`.
- High false-positive risk: `True`.
- Recommendation: `disable_binary_llm_rule_features_as_deterministic_priors`.

## Split Summary

| Split | Event Precision | Event FP Ratio | Event Recall | Zero Precision | Zero FP Ratio | Zero Recall |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.0570 | 0.9430 | 0.0181 | 0.0570 | 0.9430 | 0.0181 |
| val | 0.0573 | 0.9427 | 0.0140 | 0.0573 | 0.9427 | 0.0140 |
| test | 0.0900 | 0.9100 | 0.0135 | 0.0900 | 0.9100 | 0.0135 |

## Per-channel Test Zero-event Feature

| Channel | Precision | FP Ratio | Recall | Predicted | Actual |
|---|---:|---:|---:|---:|---:|
| HUFL | 0.0417 | 0.9583 | 0.0139 | 192 | 577 |
| HULL | 0.0000 | 1.0000 | 0.0000 | 192 | 2 |
| MUFL | 0.0312 | 0.9688 | 0.0117 | 192 | 514 |
| MULL | 0.0573 | 0.9427 | 0.0235 | 192 | 468 |
| LUFL | 0.0000 | 1.0000 | 0.0000 | 192 | 32 |
| LULL | 0.0000 | 1.0000 | 0.0000 | 192 | 292 |
| OT | 0.5000 | 0.5000 | 0.0136 | 192 | 7084 |

## Test Anchor False Positives

| Date | Predicted Points | True Points | TP | FP | Precision | FP Ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2017-11-01 | 672 | 19 | 19 | 653 | 0.0283 | 0.9717 |
| 2018-01-01 | 672 | 102 | 102 | 570 | 0.1518 | 0.8482 |

## Days-to-event Feature

| Split | Threshold Days | Precision | FP Ratio | Recall |
|---|---:|---:|---:|---:|
| train | 0.0 | 0.3125 | 0.6875 | 0.0196 |
| train | 0.25 | 0.3046 | 0.6954 | 0.0231 |
| train | 1.0 | 0.3065 | 0.6935 | 0.0353 |
| train | 2.0 | 0.2932 | 0.7068 | 0.0491 |
| val | 0.0 | 0.3698 | 0.6302 | 0.0167 |
| val | 0.25 | 0.3500 | 0.6500 | 0.0197 |
| val | 1.0 | 0.2857 | 0.7143 | 0.0258 |
| val | 2.0 | 0.2699 | 0.7301 | 0.0366 |
| test | 0.0 | 0.5938 | 0.4062 | 0.0145 |
| test | 0.25 | 0.5750 | 0.4250 | 0.0175 |
| test | 1.0 | 0.5481 | 0.4519 | 0.0268 |
| test | 2.0 | 0.5484 | 0.4516 | 0.0402 |

## Downstream Enforcement

`analysis/verify_and_calibrate_rules.py` applies a false-positive precision gate before enabling any calibrated rule prior. The actual event mask is defined only from train/val information as `abs(y - train_zero_target) <= train distance quantile`, with default `near_zero_quantile=0.05`.

With `min_event_precision=0.5`, the current ETTm1 periodic zero-day rule is disabled for every channel on validation. Candidate priors are still reported for diagnosis, but `best_prior_type` remains `baseline`, `best_alpha` remains `0.0`, and the calibrated rule JSON has `enabled=false`.
