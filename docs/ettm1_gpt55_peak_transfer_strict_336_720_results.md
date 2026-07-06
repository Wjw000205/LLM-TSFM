# ETTm1 Strict GPT-5.5 Peak-Transfer Results for 336 and 720

## Scope

This rerun replaces the earlier fallback diagnostics for pred_len 336 and 720 with strict validation-guarded runs. A result is accepted only when the selected checkpoint has `selected_reason=guarded_event_mse`, meaning validation base MSE stays within the baseline +3% guardrail.

Configuration changes from the earlier fallback run:

- `event_weight=1.0`
- `use_peak_shape_loss=0`
- pred_len 336: `learning_rate=1e-6`
- pred_len 720: `learning_rate=5e-7`

All metrics below use `metrics_normalized.json`.

## Results

| Pred Len | Selected Epoch | Selected Reason | Baseline Overall MSE | Gated Overall MSE | Baseline Event MSE | Gated Event MSE | Event Reduction |
|---:|---:|---|---:|---:|---:|---:|---:|
| 336 | 9 | guarded_event_mse | 0.371105 | 0.371076 | 0.071750 | 0.038095 | 46.91% |
| 720 | 1 | guarded_event_mse | 0.431983 | 0.431945 | 0.213412 | 0.169587 | 20.54% |

## Conclusion

Both long horizons now have strict, non-fallback checkpoints. The conservative loss expert still improves event windows, and the gated final prediction preserves overall MSE at the DLinear baseline level.

The p720 event gain is smaller than the earlier fallback diagnostic, but it is now validation-guarded and therefore more reliable. The earlier p720 fallback result should not be used as the formal conclusion.

## Artifacts

- `artifacts/core_results/ettm1_gpt55_peak_transfer_strict_336_720_summary.csv`
- `artifacts/core_results/ettm1_gpt55_peak_transfer_strict_336_720_summary.json`
- `artifacts/figures/ettm1_gpt55_peak_transfer_strict_regions/p336/manifest.json`
- `artifacts/figures/ettm1_gpt55_peak_transfer_strict_regions/p720/manifest.json`
