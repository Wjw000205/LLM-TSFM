# Rule Prior Results

Baseline experiment: `ettm1_rule_prior_pure_dlinear`

| Experiment | Overall MSE | Overall MAE | Event MSE | Zero MSE | Rule Score | Overall Delta | Event Reduction | Non-event MSE | Alpha | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ettm1_rule_prior_alpha_025 | 0.308141 | 0.354217 | 0.685411 | 0.685411 | 0.651143 | 0.36% | -10.31% | 0.301693 | 0.25 | rule_prior_fusion |
| ettm1_rule_prior_alpha_05 | 0.315790 | 0.359282 | 1.139728 | 1.139728 | 0.607571 | 2.85% | -83.43% | 0.301706 | 0.5 | rule_prior_fusion |
| ettm1_rule_prior_alpha_05_plus_event_loss | 0.381061 | 0.407714 | 1.062179 | 1.062179 | 0.606391 | 24.11% | -70.94% | 0.369419 | 0.5 | rule_prior_fusion, dataset_aware_loss |
| ettm1_rule_prior_alpha_075 | 0.330019 | 0.365174 | 1.990496 | 1.990496 | 0.506237 | 7.49% | -220.35% | 0.301637 | 0.75 | rule_prior_fusion |
| ettm1_rule_prior_alpha_10 | 0.350815 | 0.371079 | 3.236338 | 3.236338 | 0.424820 | 14.26% | -420.85% | 0.301494 | 1.0 | rule_prior_fusion |
| ettm1_rule_prior_dataset_aware_loss | 0.366348 | 0.387346 | 0.448864 | 0.448864 | 0.613012 | 19.32% | 27.76% | 0.364938 |  | dataset_aware_loss |
| ettm1_rule_prior_hard_intervention_diagnostic | 0.350815 | 0.371079 | 3.236338 | 3.236338 | 0.424820 | 14.26% | -420.85% | 0.301494 |  | hard_intervention_diagnostic |
| ettm1_rule_prior_intermediate_intervention | 0.301396 | 0.345392 | 0.705776 | 0.705776 | 0.629684 | -1.84% | -13.59% | 0.294485 |  | intermediate_intervention |
| ettm1_rule_prior_output_rule_adapter | 0.365635 | 0.404235 | 0.589000 | 0.589000 | 0.627091 | 19.09% | 5.21% | 0.361818 |  | output_rule_adapter |
| ettm1_rule_prior_pure_dlinear | 0.307031 | 0.352800 | 0.621358 | 0.621358 | 0.586311 | 0.00% | 0.00% | 0.301658 |  | pure_dlinear |

## Run Conclusion

- Pure DLinear baseline: overall MSE 0.307031, event MSE 0.621358.
- Best rule-prior event result is alpha=0.25: event MSE 0.685411, event reduction -10.31%.
- Best rule-prior overall result is alpha=0.25: overall MSE 0.308141, delta 0.36%.
- Non-event MSE drift across pure rule-prior runs is at most 0.000164; the degradation is localized to rule-triggered timestamps.
- Mask alignment check: matches_recomputed_masks=True, max_abs_diff=0.0.
- Diagnosis: true values at zero_mask positions are farther from zero_target than baseline predictions; hard zero prior is not a valid oracle on this split.
- Diagnosis: offset scan does not show a strong fixed timestamp shift; inspect rule condition, anchor, and zero_target validity first.
- Success criteria are not met for the current ETTm1 zero_event rule: event MSE does not improve under rule-prior fusion.

## Interpretation

- Rule-prior fusion is a deterministic soft fusion branch, not a trainable MLP adapter.
- If event MSE worsens as alpha increases, diagnose zero masks and zero targets before tuning the model.
- Hard intervention should be called an oracle upper bound only after diagnosis confirms the mask and target are valid.
