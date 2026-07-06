# Multidataset GPT-5.5 Peak-Transfer Full-Horizon Results

## Scope

This report extends the gated peak-transfer check to ETTh1, ETTh2, and ETTm2 across pred_len 96, 192, 336, and 720. Each dataset uses its own GPT-5.5 generated peak-transfer rule file; no ETTm1 rule is reused.

The evaluated path is still intentionally diagnostic: a pure DLinear baseline is trained first, a dataset-aware loss expert is fine-tuned from that checkpoint, then rule-gated evaluation copies the expert prediction only inside the event mask and keeps baseline predictions elsewhere.

## Main Results

| Dataset | pred_len | Baseline Overall | Gated Overall | Baseline Event | Gated Event | Event Reduction | Event Ratio | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 96 | 0.383827 | 0.383802 | 0.136563 | 0.088367 | 35.29% | 0.0513% | guarded |
| ETTh1 | 192 | 0.418985 | 0.418981 | 0.205098 | 0.197510 | 3.70% | 0.0531% | guarded |
| ETTh1 | 336 | 0.445730 | 0.445736 | 0.319931 | 0.330366 | -3.26% | 0.0561% | guarded |
| ETTh1 | 720 | 0.493335 | 0.493358 | 0.583619 | 0.628863 | -7.75% | 0.0492% | guarded |
| ETTh2 | 96 | 0.292610 | 0.292602 | 0.101615 | 0.097232 | 4.31% | 0.1847% | guarded |
| ETTh2 | 192 | 0.373143 | 0.373125 | 0.188925 | 0.179137 | 5.18% | 0.1913% | guarded |
| ETTh2 | 336 | 0.475399 | 0.475348 | 0.298226 | 0.272459 | 8.64% | 0.2021% | guarded |
| ETTh2 | 720 | 0.659779 | 0.659650 | 0.465800 | 0.395876 | 15.01% | 0.1850% | guarded |
| ETTm2 | 96 | 0.170001 | 0.169998 | 0.025562 | 0.018821 | 26.37% | 0.0425% | guarded |
| ETTm2 | 192 | 0.226063 | 0.226059 | 0.029715 | 0.020907 | 29.64% | 0.0429% | fallback_base_mse_no_guardrail_candidate |
| ETTm2 | 336 | 0.288051 | 0.288045 | 0.046052 | 0.032207 | 30.06% | 0.0434% | fallback_base_mse_no_guardrail_candidate |
| ETTm2 | 720 | 0.421043 | 0.421019 | 0.184229 | 0.130819 | 28.99% | 0.0450% | guarded |

## Non-Event Preservation

Gated evaluation should leave the non-event region unchanged. The table below checks that property directly from `pred_normalized.npy` and `true_normalized.npy`.

| Dataset | pred_len | Baseline Non-event MSE | Gated Non-event MSE | Non-event Delta | Expected Overall Delta | Observed Overall Delta |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.383954 | 0.383954 | 0.000e+00 | -2.472e-05 | -2.474e-05 |
| ETTh1 | 192 | 0.419099 | 0.419099 | 0.000e+00 | -4.031e-06 | -4.083e-06 |
| ETTh1 | 336 | 0.445801 | 0.445801 | 0.000e+00 | 5.857e-06 | 5.931e-06 |
| ETTh1 | 720 | 0.493291 | 0.493291 | 0.000e+00 | 2.224e-05 | 2.217e-05 |
| ETTh2 | 96 | 0.292964 | 0.292964 | 0.000e+00 | -8.094e-06 | -8.076e-06 |
| ETTh2 | 192 | 0.373496 | 0.373496 | 0.000e+00 | -1.872e-05 | -1.872e-05 |
| ETTh2 | 336 | 0.475758 | 0.475758 | 0.000e+00 | -5.207e-05 | -5.171e-05 |
| ETTh2 | 720 | 0.660138 | 0.660138 | 0.000e+00 | -1.294e-04 | -1.294e-04 |
| ETTm2 | 96 | 0.170063 | 0.170063 | 0.000e+00 | -2.866e-06 | -2.861e-06 |
| ETTm2 | 192 | 0.226147 | 0.226147 | 0.000e+00 | -3.776e-06 | -3.770e-06 |
| ETTm2 | 336 | 0.288156 | 0.288156 | 0.000e+00 | -6.012e-06 | -6.020e-06 |
| ETTm2 | 720 | 0.421149 | 0.421149 | 0.000e+00 | -2.402e-05 | -2.405e-05 |

## Guardrail Selection

| Dataset | pred_len | selected_reason | selected_epoch | selection_metric | tolerance | event_weight | peak_shape | learning_rate |
|---|---:|---|---:|---|---:|---:|---|---:|
| ETTh1 | 96 | guarded_event_mse | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh1 | 192 | guarded_event_mse | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh1 | 336 | guarded_event_mse | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh1 | 720 | guarded_event_mse | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh2 | 96 | guarded_event_mse | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh2 | 192 | guarded_event_mse | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh2 | 336 | guarded_event_mse | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTh2 | 720 | guarded_event_mse | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTm2 | 96 | guarded_event_mse | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTm2 | 192 | fallback_base_mse_no_guardrail_candidate | 2 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTm2 | 336 | fallback_base_mse_no_guardrail_candidate | 1 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |
| ETTm2 | 720 | guarded_event_mse | 3 | guarded_event_mse | 0.03 | 5.0 | True | 1e-05 |

## Interpretation

- ETTh1 improves event-window MSE at 96 and 192, but the same standard peak-transfer setting degrades event-window MSE at 336 and 720. Its long-horizon rule/loss configuration is therefore not robust without the stricter conservative settings used in the ETTm1 consolidation.
- ETTh2 improves event-window MSE at all four horizons, with stronger gains as the horizon grows in this sweep.
- ETTm2 improves event-window MSE at all four horizons, but 192 and 336 are fallback selections rather than guardrail-selected experts, so those two should be labeled diagnostic rather than fully guarded wins.
- Event ratios are only about 0.04% to 0.20% in this sweep, so overall MSE changes are necessarily tiny. The expected-vs-observed delta check confirms that overall movement is explained by the event-local replacement.

## Artifacts

- `artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.csv`
- `artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.json`
- `docs/multidataset_full_horizon_peak_transfer_results.md`
