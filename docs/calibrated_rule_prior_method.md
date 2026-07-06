# Calibrated Rule Prior Method

Naive zero-prior fusion failed on ETTm1 because the rule-triggered test values were not close to the z-scored zero target. The non-event MSE stayed nearly unchanged, so the fusion mechanism was localized correctly; the invalid assumption was the prior effect itself.

The calibrated rule-prior workflow treats every LLM rule as a hypothesis. Rule/channel/prior type/alpha selection is performed only on train or validation data. The test split is only used for final reporting.

Supported prior templates:

- `zero_target`: fuse toward the scaled zero target.
- `residual_mean`: add the mean calibration residual at rule-triggered points.
- `residual_median`: add the median calibration residual at rule-triggered points.
- `ratio`: scale the baseline prediction by the mean calibration ratio.
- `conditional_mean`: fuse toward the calibration mean true value at rule-triggered points.

Per-channel calibration prevents `affected_variables="all"` from blindly modifying every target variable. A channel is enabled only when its best calibrated prior improves calibration event MSE by at least `min_prior_improvement` and uses `best_alpha > 0`.

False-positive filtering is now part of calibration. The verifier builds the actual event target on train/val only:

- Compute each channel's train-split distance to the z-scored `zero_target`.
- Use `near_zero_quantile=0.05` as the near-zero threshold.
- On the calibration split, compare each LLM rule mask against that actual near-zero mask.
- Enable a channel only when rule-mask precision is at least `min_event_precision` and the calibrated prior also improves baseline MSE.

Current ETTm1 calibration with `min_event_precision=0.5` disables every channel. Validation precision is below the threshold for all channels, with false-positive ratios around `0.90-1.00`. This means the current periodic zero-day LLM feature remains useful only as a diagnostic hypothesis; it is not safe as a deterministic prior.

An additional `rule_window_mse` calibration objective was tested to treat the LLM rule as a regime window instead of a near-zero event detector. It enabled all channels on validation with residual/ratio/conditional-mean priors, but test performance worsened to overall MSE `0.310001` and rule-event MSE `0.798098`. This confirms that validation rule-window residuals are not stable enough to deploy as fixed priors for the current rule.

Current interpretation:

- `dataset-aware loss` remains a diagnostic method: it improves event MSE but damages overall MSE.
- `guarded longtail medium weights` is the current strongest acceptable result: overall MSE `0.317920`, event MSE `0.554032`.
- `hard_intervention` is not an oracle unless prior validity is verified on train/val.
- If calibrated priors fail the false-positive precision gate, the current LLM rule should be disabled and rule mining should be revisited.
