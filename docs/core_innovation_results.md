# Core Innovation Results

Filter: `ettm1`

| Experiment | Overall MSE | Overall MAE | Event MSE | Zero MSE | Rule Score | Notes |
|---|---:|---:|---:|---:|---:|---|
| ettm1_calibrated_fixed_zero_prior_alpha_025 | 0.308141 | 0.354217 | 0.685411 | 0.685411 | 0.651143 | pure_dlinear |
| ettm1_calibrated_fixed_zero_prior_alpha_05 | 0.315790 | 0.359282 | 1.139728 | 1.139728 | 0.607571 | pure_dlinear |
| ettm1_calibrated_pure_dlinear | 0.307031 | 0.352800 | 0.621358 | 0.621358 | 0.586311 | pure_dlinear |
| ettm1_calibrated_residual_prior | 0.310095 | 0.354409 | 0.792390 | 0.792390 | 0.674421 | pure_dlinear |
| ettm1_calibrated_rule_prior_plus_dataset_aware_loss | 0.355288 | 0.375175 | 0.693772 | 0.693772 | 0.666208 | dataset_aware_loss |
| ettm1_core_dataset_aware_loss | 0.366348 | 0.387346 | 0.448864 | 0.448864 | 0.613012 | dataset_aware_loss |
| ettm1_core_pure_dlinear | 0.307031 | 0.352800 | 0.621358 | 0.621358 | 0.586311 | pure_dlinear |
| ettm1_dlinear_baseline_confirm | 0.307031 | 0.352800 | 0.000000 | 0.000000 | 0.000000 | pure_dlinear |
| ettm1_guarded_longtail_finetune | 0.305944 | 0.348446 | 0.582883 | 0.582883 | 0.589355 | dataset_aware_loss |
| ettm1_guarded_longtail_valguard_rerun | 0.305944 | 0.348446 | 0.582883 | 0.582883 | 0.589355 | dataset_aware_loss |
| ettm1_intervention_dataset_aware_loss | 0.366348 | 0.387346 | 0.448864 | 0.448864 | 0.613012 | dataset_aware_loss |
| ettm1_intervention_finetune_event_only | 0.309610 | 0.353883 | 0.774808 | 0.774808 | 0.617722 | llm_rule_features, dataset_aware_loss |
| ettm1_intervention_hard_intervention_oracle | 0.350815 | 0.371079 | 3.236338 | 3.236338 | 0.424820 | oracle_like_hard_intervention |
| ettm1_intervention_intermediate | 0.301396 | 0.345392 | 0.705776 | 0.705776 | 0.629684 | llm_rule_features |
| ettm1_intervention_intermediate_plus_event_loss | 0.308224 | 0.348481 | 0.804767 | 0.804767 | 0.632773 | llm_rule_features, dataset_aware_loss |
| ettm1_intervention_intermediate_plus_event_loss_plus_reg | 0.308053 | 0.350247 | 0.723670 | 0.723670 | 0.643515 | llm_rule_features, dataset_aware_loss |
| ettm1_intervention_intermediate_scale01_plus_event_loss_plus_reg | 0.307827 | 0.349036 | 0.790487 | 0.790487 | 0.594268 | llm_rule_features, dataset_aware_loss |
| ettm1_intervention_output_rule_adapter | 0.365635 | 0.404235 | 0.589000 | 0.589000 | 0.627091 | llm_rule_features |
| ettm1_intervention_pure_dlinear | 0.307031 | 0.352800 | 0.621358 | 0.621358 | 0.586311 | pure_dlinear |
| ettm1_longtail_low_weights | 0.312149 | 0.355997 | 0.589348 | 0.589348 | 0.593233 | dataset_aware_loss |
| ettm1_longtail_medium_weights | 0.317920 | 0.359254 | 0.554032 | 0.554032 | 0.597995 | dataset_aware_loss |
| ettm1_mined_calendar_baseline_eval_rawinput | 0.307031 | 0.352800 | 0.070463 | 0.070463 | 0.692819 | llm_rule_features |
| ettm1_mined_calendar_intervention_event_only | 0.307720 | 0.353464 | 0.100909 | 0.100909 | 0.718464 | llm_rule_features, dataset_aware_loss |
| ettm1_regime_residual_prior_eval | 0.310001 | 0.354298 | 0.798098 | 0.798098 | 0.670924 | pure_dlinear |
| ettm1_rule_prior_alpha_025 | 0.308141 | 0.354217 | 0.685411 | 0.685411 | 0.651143 | pure_dlinear |
| ettm1_rule_prior_alpha_05 | 0.315790 | 0.359282 | 1.139728 | 1.139728 | 0.607571 | pure_dlinear |
| ettm1_rule_prior_alpha_05_plus_event_loss | 0.381061 | 0.407714 | 1.062179 | 1.062179 | 0.606391 | dataset_aware_loss |
| ettm1_rule_prior_alpha_075 | 0.330019 | 0.365174 | 1.990496 | 1.990496 | 0.506237 | pure_dlinear |
| ettm1_rule_prior_alpha_10 | 0.350815 | 0.371079 | 3.236338 | 3.236338 | 0.424820 | pure_dlinear |
| ettm1_rule_prior_dataset_aware_loss | 0.366348 | 0.387346 | 0.448864 | 0.448864 | 0.613012 | dataset_aware_loss |
| ettm1_rule_prior_hard_intervention_diagnostic | 0.350815 | 0.371079 | 3.236338 | 3.236338 | 0.424820 | oracle_like_hard_intervention |
| ettm1_rule_prior_intermediate_intervention | 0.301396 | 0.345392 | 0.705776 | 0.705776 | 0.629684 | llm_rule_features |
| ettm1_rule_prior_output_rule_adapter | 0.365635 | 0.404235 | 0.589000 | 0.589000 | 0.627091 | llm_rule_features |
| ettm1_rule_prior_pure_dlinear | 0.307031 | 0.352800 | 0.621358 | 0.621358 | 0.586311 | pure_dlinear |
| ettm1_soft_event_full_dlinear_w1_p2_guarded | 1.106861 | 0.787174 | 1.259900 | 1.259900 | 0.697731 | llm_rule_features, dataset_aware_loss |
| ettm1_soft_event_intervention_finetune_w05_p2_scale01 | 0.307671 | 0.353043 | 0.659466 | 0.659466 | 0.635827 | llm_rule_features, dataset_aware_loss |
| ettm1_dlinear_seq96_confirm | 0.342972 | 0.369801 | 0.000000 | 0.000000 | 0.000000 | pure_dlinear |

## Auto-Diagnosis Draft

- Naive dataset-aware loss can reduce event-window MSE while materially increasing overall MSE; do not treat that as a deployable win.
- Use guarded selection, non-event preservation, and weight sweeps when optimizing long-tail regions.
- Treat `hard_intervention_oracle` as an oracle-like upper bound, not a deployable method result.
- If event point counts are zero, regenerate or validate rule masks before interpreting event metrics.
- A deployable long-tail result should stay inside the overall-MSE guardrail while improving event-window metrics.
