# Core Innovation Results

Filter: `ettm1_core`

| Experiment | Overall MSE | Overall MAE | Event MSE | Zero MSE | Rule Score | Notes |
|---|---:|---:|---:|---:|---:|---|
| ettm1_core_dataset_aware_loss | 0.366348 | 0.387346 | 0.448864 | 0.448864 | 0.613012 | dataset_aware_loss |
| ettm1_core_pure_dlinear | 0.307031 | 0.352800 | 0.621358 | 0.621358 | 0.586311 | pure_dlinear |

## Auto-Diagnosis Draft

- Compare `pure_dlinear` against feature/loss variants using both overall and event-window metrics.
- Treat `hard_intervention_oracle` as an oracle-like upper bound, not a deployable method result.
- If event point counts are zero, regenerate or validate rule masks before interpreting event metrics.
- Overall MSE/MAE should not materially degrade while event-window metrics improve.
