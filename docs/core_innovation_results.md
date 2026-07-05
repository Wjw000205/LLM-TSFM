# Core Innovation Results

Filter: `ettm1_core`

| Experiment | Overall MSE | Overall MAE | Event MSE | Zero MSE | Rule Score | Notes |
|---|---:|---:|---:|---:|---:|---|
| ettm1_core_dataset_aware_loss | 0.366348 | 0.387346 | 0.448864 | 0.448864 | 0.613012 | dataset_aware_loss |
| ettm1_core_pure_dlinear | 0.307031 | 0.352800 | 0.621358 | 0.621358 | 0.586311 | pure_dlinear |

## Auto-Diagnosis Draft

- Naive dataset-aware loss reduces event-window MSE from `0.621358` to `0.448864`, but overall MSE regresses from `0.307031` to `0.366348` (+19.32%). This is not a deployable win.
- Phase-2 long-tail experiments should keep pure DLinear as the base and use guarded selection, non-event preservation, and weight sweeps to reduce event-window error without breaking overall MSE.
- Current guarded long-tail results are summarized in `docs/longtail_guardrail_results.md`; the best event reduction among accepted candidates is `ettm1_longtail_medium_weights` with overall MSE `0.317920` (+3.55%) and event MSE `0.554032` (-10.84%).
- Treat `hard_intervention_oracle` as an oracle-like upper bound, not a deployable method result.
- If event point counts are zero, regenerate or validate rule masks before interpreting event metrics.
- A deployable long-tail result should stay inside the overall-MSE guardrail while improving event-window metrics.
