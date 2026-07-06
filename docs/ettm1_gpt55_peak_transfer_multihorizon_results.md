# ETTm1 GPT-5.5 Peak-Transfer Multihorizon Results

## Scope

This run checks the current LLM-assisted event method across multiple forecast horizons on ETTm1. The method stays aligned with the current direction:

- GPT-5.5 provides the sparse peak-event hypothesis.
- DLinear is trained as the pure baseline.
- A loss expert is fine-tuned with `DatasetAwareLoss` on the peak-event mask.
- The gated prediction uses the baseline outside event windows and the loss expert inside event windows.

All numbers below use `metrics_normalized.json`.

## Results

| Pred Len | Baseline Overall MSE | Gated Overall MSE | Baseline Event MSE | Gated Event MSE | Event Reduction | Status |
|---:|---:|---:|---:|---:|---:|---|
| 96 | 0.307031 | 0.307010 | 0.051324 | 0.025899 | 49.54% | guarded |
| 192 | 0.339500 | 0.339479 | 0.065048 | 0.039033 | 39.99% | guarded |
| 336 | 0.371105 | 0.371076 | 0.071750 | 0.038095 | 46.91% | guarded |
| 720 | 0.431983 | 0.431945 | 0.213412 | 0.169587 | 20.54% | guarded |

## Conclusion

The multihorizon result supports the current hypothesis: the event/overall conflict is caused by letting the event-focused loss expert affect non-event regions. When event gating is used, overall MSE stays at the DLinear baseline while event MSE drops substantially.

All four horizons now use validation-guarded checkpoints. Pred_len 336 and 720 were rerun with a conservative strict setup (`event_weight=1.0`, `use_peak_shape_loss=0`) so the selected checkpoint has `selected_reason=guarded_event_mse` rather than fallback. The p720 event gain is smaller than the earlier fallback diagnostic, but it is the reliable result to report.

## Artifacts

- `artifacts/core_results/ettm1_gpt55_peak_transfer_multihorizon_summary.csv`
- `artifacts/core_results/ettm1_gpt55_peak_transfer_multihorizon_summary.json`
- `docs/ettm1_gpt55_peak_transfer_strict_336_720_results.md`
- `scripts/run_multihorizon_gpt55_peak_transfer.ps1`
