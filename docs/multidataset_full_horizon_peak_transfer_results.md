# Multidataset GPT-5.5 Peak-Transfer Full-Horizon Results

## Scope

This report extends the gated peak-transfer check to ETTh1, ETTh2, and ETTm2 across pred_len 96, 192, 336, and 720. Each dataset uses its own GPT-5.5 generated dataset-level peak-transfer rule file; no ETTm1 rule is reused.

The evaluated path is still intentionally diagnostic: a pure DLinear baseline is trained first, a dataset-aware loss expert is fine-tuned from that checkpoint, then rule-gated evaluation copies the expert prediction only inside the event mask and keeps baseline predictions elsewhere.

## Main Results

| Dataset | pred_len | Baseline Overall | Gated Overall | Baseline Event | Gated Event | Event Reduction | Event Ratio | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 96 | 0.383827 | 0.383827 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTh1 | 192 | 0.418985 | 0.418985 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTh1 | 336 | 0.445730 | 0.445730 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTh1 | 720 | 0.493335 | 0.493335 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTh2 | 96 | 0.292610 | 0.292610 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTh2 | 192 | 0.373143 | 0.373143 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTh2 | 336 | 0.475399 | 0.475399 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTh2 | 720 | 0.659779 | 0.659779 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTm2 | 96 | 0.170001 | 0.170001 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTm2 | 192 | 0.226063 | 0.226063 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTm2 | 336 | 0.288051 | 0.288051 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |
| ETTm2 | 720 | 0.421043 | 0.421043 | NA | NA | NA | 0.0000% | not_applicable_empty_mask |

## Non-Event Preservation

Gated evaluation should leave the non-event region unchanged. The table below checks that property directly from `pred_normalized.npy` and `true_normalized.npy`.

| Dataset | pred_len | Baseline Non-event MSE | Gated Non-event MSE | Non-event Delta | Expected Overall Delta | Observed Overall Delta |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.383827 | 0.383827 | 0.000e+00 | NA | 0.000e+00 |
| ETTh1 | 192 | 0.418985 | 0.418985 | 0.000e+00 | NA | 0.000e+00 |
| ETTh1 | 336 | 0.445730 | 0.445730 | 0.000e+00 | NA | 0.000e+00 |
| ETTh1 | 720 | 0.493335 | 0.493335 | 0.000e+00 | NA | 0.000e+00 |
| ETTh2 | 96 | 0.292610 | 0.292610 | 0.000e+00 | NA | 0.000e+00 |
| ETTh2 | 192 | 0.373143 | 0.373143 | 0.000e+00 | NA | 0.000e+00 |
| ETTh2 | 336 | 0.475399 | 0.475399 | 0.000e+00 | NA | 0.000e+00 |
| ETTh2 | 720 | 0.659779 | 0.659779 | 0.000e+00 | NA | 0.000e+00 |
| ETTm2 | 96 | 0.170001 | 0.170001 | 0.000e+00 | NA | 0.000e+00 |
| ETTm2 | 192 | 0.226063 | 0.226063 | 0.000e+00 | NA | 0.000e+00 |
| ETTm2 | 336 | 0.288051 | 0.288051 | 0.000e+00 | NA | 0.000e+00 |
| ETTm2 | 720 | 0.421043 | 0.421043 | 0.000e+00 | NA | 0.000e+00 |

## Guardrail Selection

| Dataset | pred_len | selected_reason | selected_epoch | selection_metric | tolerance | event_weight | peak_shape | learning_rate |
|---|---:|---|---:|---|---:|---:|---|---:|
| ETTh1 | 96 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh1 | 192 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh1 | 336 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh1 | 720 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh2 | 96 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh2 | 192 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh2 | 336 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh2 | 720 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTm2 | 96 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTm2 | 192 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTm2 | 336 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTm2 | 720 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |

## Event Coverage

| Dataset | pred_len | event_points | total_prediction_elements | event_mask_warning |
|---|---:|---:|---:|---|
| ETTh1 | 96 | 0 | 1871520 | empty_event_mask |
| ETTh1 | 192 | 0 | 3614016 | empty_event_mask |
| ETTh1 | 336 | 0 | 5985840 | empty_event_mask |
| ETTh1 | 720 | 0 | 10891440 | empty_event_mask |
| ETTh2 | 96 | 0 | 1871520 | empty_event_mask |
| ETTh2 | 192 | 0 | 3614016 | empty_event_mask |
| ETTh2 | 336 | 0 | 5985840 | empty_event_mask |
| ETTh2 | 720 | 0 | 10891440 | empty_event_mask |
| ETTm2 | 96 | 0 | 7677600 | empty_event_mask |
| ETTm2 | 192 | 0 | 15226176 | empty_event_mask |
| ETTm2 | 336 | 0 | 26307120 | empty_event_mask |
| ETTm2 | 720 | 0 | 54437040 | empty_event_mask |

## Interpretation

- The regenerated GPT-5.5 dataset-level rules produced zero test event coverage for all 12 dataset/horizon combinations.
- Because the test event mask is empty, event-window MSE and event reduction are not applicable, and gated evaluation is exactly the baseline in all rows.
- This run resolves the horizon-reuse issue, but it also shows that the current GPT rule mining prompt is too conservative or too train-evidence-specific for transferable test-time event discovery.
- The next method change should target transferable event mining with train-only evidence, then require a nonzero validation/test event coverage diagnostic before running expensive event-loss fine-tuning.

## Artifacts

- `artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.csv`
- `artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.json`
- `docs/multidataset_full_horizon_peak_transfer_results.md`
