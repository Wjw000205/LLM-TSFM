# LLM Method Improvement Status

Goal: make the LLM-derived rule/features improve event-window forecasting without materially damaging overall MSE.

## Supervision Setup

- Main process goal is active.
- Strategy subagent recommended treating the current LLM rule as a weak regime hypothesis, not a near-zero prior.
- Review subagent flagged these hard constraints:
  - Do not use test to select rule, shift, alpha, threshold, or prior type.
  - Do not treat `zero_target` or hard intervention as oracle unless val proves the prior.
  - Use `guarded_event_mse` with a validation-split baseline for deployable experiments.
  - Report normalized metrics and keep repeated rule-event metrics separate from actual near-zero diagnostics.

## Implemented Changes

- Added `train_only_intervention` to freeze the DLinear backbone and train only the intermediate intervention layer.
- Added validation-first baseline loading for guardrail selection from checkpoint `config.json` / `validation_history.json`.
- Added `calibration_objective=rule_window_mse` and `allowed_prior_types` to rule calibration.
- Added soft LLM event features from `days_to_event`:
  - `soft_event_score_<pattern>`
  - default score: `exp(-days_to_event / soft_event_tau_days)`
- Added `use_soft_event_weighted_loss`, `soft_event_weight`, `soft_event_feature_regex`, and `soft_event_power`.
- Updated scripts so guarded experiments use `selection_metric=guarded_event_mse` with validation baseline config.
- Added `analysis/mine_validated_calendar_windows.py` to mine candidate calendar-window rules from train/val only.
- Added `center_hour` support for `calendar_window` masks.
- Added split-scoped candidate evaluation so train and val precision are computed only inside their own split.
- Added `min_train_precision` to reject rules that only look good on validation but have weak train precision.

## Results So Far

Baseline pure DLinear:

- overall MSE: `0.307031`
- rule-event MSE: `0.621358`

Failed attempts:

| Method | Overall MSE | Rule-event MSE | Status |
|---|---:|---:|---|
| frozen intermediate intervention, hard event loss | 0.309610 | 0.774808 | fails event |
| regime residual calibrated prior | 0.310001 | 0.798098 | fails event |
| frozen intermediate intervention, soft event loss | 0.307671 | 0.659466 | preserves overall but fails event |
| full DLinear with soft event loss | 1.106861 | 1.259900 | fails guardrail and event |

Current validated LLM-based result is guarded longtail with validation guardrail:

- baseline test overall MSE: `0.307031`
- baseline test rule-event MSE: `0.621358`
- guarded test overall MSE: `0.305944`
- guarded test rule-event MSE: `0.582883`
- test event reduction: `6.19%`
- test overall delta: `-0.35%`
- validation guardrail: selected epoch `1`, `val_base_mse=0.398042` versus baseline `0.383451`, within the configured `5%` tolerance
- validation event MSE: `0.279899 -> 0.278254`

Rule-gated ensemble result:

- baseline model handles non-event timestamps.
- aggressive event-specialized model `ettm1_event_push_w20_z0` handles only rule event timestamps.
- event mask is a hard gate, so non-event predictions are exactly identical to baseline.
- test overall MSE: `0.303677`
- test rule-event MSE: `0.421775`
- non-event max absolute prediction difference versus baseline: `0.0`
- interpretation: this preserves the aggressive event improvement while removing the non-event pollution that caused overall MSE to rise.

Cross-dataset note:

- The earlier local ETTh1/ETTh2/ETTm2 rule-gated numbers reused the ETTm1 rule mask and are therefore mechanism checks only.
- They must not be reported as full LLM-method evidence.
- Formal cross-dataset experiments must first call `analysis/generate_dataset_llm_rules.py` for each dataset and then train with `--llm_rule_path ./llm_rules/generated_rules/${DATA}_rules.json`.
- `scripts/run_multidataset_llm_rulegate.sh` implements this required order.

Train/val-mined calendar-window diagnosis:

| Step | Result | Interpretation |
|---|---:|---|
| initial mining without train precision gate | 10 selected candidates | all selected rules target `LULL` around day 28/29; val precision up to `0.474`, but train precision only `0.042-0.087` |
| intervention-only training on those candidates | test overall MSE `0.307720`, event MSE `0.100909` | event worsened versus same-mask baseline event MSE `0.070463` |
| stable mining with `min_train_precision=0.25` | 0 selected candidates | no current calendar-window hypothesis is stable across train and val |

## Current Interpretation

The current `periodic_zero_day` LLM rule is not a reliable event detector and its validation residual pattern does not transfer to test. Soft weighting and zero-init intervention reduce blast radius, but they still learn a split-specific correction from an unstable rule.

The strongest current event/overall tradeoff comes from a rule-gated ensemble: use the event-specialized model only where the LLM rule mask is active and fall back to baseline everywhere else. This directly addresses the diagnosed failure mode: aggressive event training improves the small event subset but damages the much larger non-event subset. Hard gating prevents that non-event damage.

The mined calendar-window route confirmed a separate rule-quality failure mode: without train precision gating it selected validation-specific windows, and training on them made event MSE worse. With train and val precision gating enabled, no ETTm1 calendar-window rule survives. The validated rule output is therefore intentionally empty, which is the correct no-op behavior for this rule family.

The next useful path is not more MLP or alpha tuning. It is rule-quality improvement:

1. Broaden the rule hypothesis beyond fixed monthly calendar windows, while keeping train/val-only selection.
2. Require each selected rule to pass both train and val precision/recall gates.
3. Use selected candidates as weak event windows with guarded training only after they pass those gates.
4. Report test once, with no test-time rule or shift selection.
