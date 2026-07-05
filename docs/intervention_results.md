# Intervention Results

Baseline experiment: `ettm1_intervention_pure_dlinear`

| Experiment | Overall MSE | Overall MAE | Event MSE | Zero MSE | Rule Score | Overall Delta | Event Reduction | Mean Event Gate | Mean Non-event Gate | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ettm1_intervention_dataset_aware_loss | 0.366348 | 0.387346 | 0.448864 | 0.448864 | 0.613012 | 19.32% | 27.76% | 0.000000 | 0.000000 | dataset_aware_loss |
| ettm1_intervention_hard_intervention_oracle | 0.350815 | 0.371079 | 3.236338 | 3.236338 | 0.424820 | 14.26% | -420.85% | 0.000000 | 0.000000 | oracle_like_hard_intervention |
| ettm1_intervention_intermediate | 0.301396 | 0.345392 | 0.705776 | 0.705776 | 0.629684 | -1.84% | -13.59% | 0.019759 | 0.206117 | intermediate_intervention |
| ettm1_intervention_intermediate_plus_event_loss | 0.308224 | 0.348481 | 0.804767 | 0.804767 | 0.632773 | 0.39% | -29.52% | 0.026002 | 0.373188 | intermediate_intervention, dataset_aware_loss |
| ettm1_intervention_intermediate_plus_event_loss_plus_reg | 0.308053 | 0.350247 | 0.723670 | 0.723670 | 0.643515 | 0.33% | -16.47% | 0.026806 | 0.011245 | intermediate_intervention, dataset_aware_loss, intervention_reg |
| ettm1_intervention_intermediate_scale01_plus_event_loss_plus_reg | 0.307827 | 0.349036 | 0.790487 | 0.790487 | 0.594268 | 0.26% | -27.22% | 0.020396 | 0.001470 | intermediate_intervention, dataset_aware_loss, intervention_reg |
| ettm1_intervention_output_rule_adapter | 0.365635 | 0.404235 | 0.589000 | 0.589000 | 0.627091 | 19.09% | 5.21% | 0.000000 | 0.000000 | output_rule_adapter |
| ettm1_intervention_pure_dlinear | 0.307031 | 0.352800 | 0.621358 | 0.621358 | 0.586311 | 0.00% | 0.00% | 0.000000 | 0.000000 | pure_dlinear |

## Interpretation

- Current ETTm1 results show the intermediate intervention preserves or improves overall MSE, but it does not yet improve event-window MSE against pure DLinear.
- Intervention regularization suppresses non-event gate activity, which confirms the regularizer is active, but this run still needs better event supervision or rule quality to improve event MSE.
- In this run, event-window improvements come from `dataset_aware_loss` and `output_rule_adapter`, but both materially hurt overall MSE.
- `output_rule_adapter` is post-prediction residual correction and remains an ablation.
- `intermediate_intervention` is the main timestamp-conditioned rule-gated method.
- `hard_intervention_oracle` is an oracle-like ablation, not a deployable method; with the current ETTm1 zero-event mask it is not an empirical upper bound.
- `dataset_aware_loss` is a diagnostic baseline because it can reduce event MSE while hurting overall MSE.
