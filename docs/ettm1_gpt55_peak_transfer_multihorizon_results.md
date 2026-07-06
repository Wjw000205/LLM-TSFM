# ETTm1 GPT-5.5 Gated Peak-Transfer Multi-Horizon Results

## Main Results

This report consolidates the guarded ETTm1 multi-horizon gated peak-transfer results. All metrics use `metrics_normalized.json`.

| pred_len | Baseline Overall | Gated Overall | Baseline Event | Gated Event | Event Reduction | Status |
|---:|---:|---:|---:|---:|---:|---|
| 96 | 0.307031 | 0.307010 | 0.051324 | 0.025899 | 49.54% | guarded |
| 192 | 0.339500 | 0.339479 | 0.065048 | 0.039033 | 39.99% | guarded |
| 336 | 0.371105 | 0.371076 | 0.071750 | 0.038095 | 46.91% | guarded |
| 720 | 0.431983 | 0.431945 | 0.213412 | 0.169587 | 20.54% | guarded |

## Guardrail Configuration

All four accepted results use `selection_metric=guarded_event_mse` and `overall_mse_tolerance=0.03`.

| pred_len | selected_epoch | event_weight | use_peak_shape_loss | learning_rate |
|---:|---:|---:|---:|---:|
| 96 | 3 | 5.0 | true | 1e-5 |
| 192 | 4 | 5.0 | true | 1e-5 |
| 336 | 9 | 1.0 | 0 | 1e-6 |
| 720 | 1 | 1.0 | 0 | 5e-7 |

The 336 and 720 horizons were rerun with conservative strict settings, so they are no longer fallback diagnostics.

## Event Ratio

The event mask is extremely sparse under the current metric path, where total prediction elements are `samples * pred_len * channels`.

| pred_len | event_points | total_prediction_elements | event_ratio |
|---:|---:|---:|---:|
| 96 | 6,336 | 7,677,600 | 0.0825% |
| 192 | 12,672 | 15,226,176 | 0.0832% |
| 336 | 22,176 | 26,307,120 | 0.0843% |
| 720 | 47,520 | 54,437,040 | 0.0873% |

## Non-Event Preservation

The gated prediction uses the loss expert only inside the event mask and keeps the DLinear baseline outside it.

| pred_len | Baseline Non-event MSE | Gated Non-event MSE | Non-event Delta |
|---:|---:|---:|---:|
| 96 | 0.307242 | 0.307242 | 0.000000 |
| 192 | 0.339729 | 0.339729 | 0.000000 |
| 336 | 0.371358 | 0.371358 | 0.000000 |
| 720 | 0.432175 | 0.432175 | 0.000000 |

## Why Overall MSE Changes Only Slightly

Overall MSE is diluted by the rarity of the event region. The expected overall change from event improvement alone is:

`event_ratio * (gated_event_mse - baseline_event_mse)`

| pred_len | Expected Overall Delta From Event | Observed Overall Delta | Delta Match Error |
|---:|---:|---:|---:|
| 96 | -2.098e-5 | -2.095e-5 | 3.129e-8 |
| 192 | -2.165e-5 | -2.161e-5 | 4.480e-8 |
| 336 | -2.837e-5 | -2.840e-5 | -3.184e-8 |
| 720 | -3.826e-5 | -3.821e-5 | 4.958e-8 |

The observed overall change is almost exactly explained by the event-local change. This confirms that gated peak-transfer is acting in the intended sparse region and is not hiding broad non-event damage.

## Main Conclusion

Across four prediction horizons, gated peak-transfer consistently reduces long-tail event-window MSE by 20% to 50% under a strict overall-MSE guardrail. Since the event ratio is below 0.1%, the resulting overall-MSE improvement is necessarily small, but it is consistent with the event-local nature of the intervention.

中文补充：在四个预测长度下，gated peak-transfer 在严格 overall MSE guardrail 下稳定降低长尾事件误差。由于 event 占比低于 0.1%，overall MSE 的变化天然很小；这说明需要同时报告 event-window MSE、non-event MSE 和 overall guardrail，而不能只看 overall MSE。

## Artifacts

- `artifacts/core_results/ettm1_gpt55_peak_transfer_multihorizon_summary.csv`
- `artifacts/core_results/ettm1_gpt55_peak_transfer_multihorizon_summary.json`
- `docs/ettm1_gpt55_peak_transfer_strict_336_720_results.md`
- `scripts/run_multihorizon_gpt55_peak_transfer.ps1`
