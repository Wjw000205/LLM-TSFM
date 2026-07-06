# Split Offset Diagnosis

- Dataset alignment bug: `False`.
- Train best lag: `-10` steps.
- Val best lag: `24` steps.
- Test best lag: `137` steps.
- Split phase shift: `True`.
- Rule stability: `unstable`.
- Offset predictability: `not_predictable_from_train_val`.
- Disable prior recommendation: `True`.
- Recommendation: `case_c_unstable_rule_offset_disable_deterministic_prior_use_guarded_loss_or_remine_rules`.

## Direct Answers

1. Val/test offset exists: `True`.
2. Offset is stable: `False`.
3. Source: `true distribution/rule phase drift, not a Dataset slicing bug`.
4. Rule mask hits true near-zero events weakly: train F1 `0.0263`, val F1 `0.0401`, test F1 `0.0527`.
5. Zero prior is valid on stable channels: `False`.
6. Stable channels: `none`.
7. Disable current deterministic rule prior: `True`.
8. Shift-aware rule recommended: `False`; calendar_window or re-mined rules are preferred when offsets are unstable.
9. Predictability: `test_offsets_form_a_post_hoc_regime_but_are_outside_train_val_range`.

## Horizon-Invariance Update

Update 2026-07-07: event definitions must be horizon-independent. The Dataset now generates rule masks from the full CSV timestamp index first, then slices `global_event_mask[border1:border2]` for each split. Sliding-window `__getitem__` uses the same `r_begin:r_end` indices for `seq_y` and `seq_y_event_masks`, and the prediction mask is `seq_y_event_masks[-pred_len:]`.

This keeps the original split-offset conclusion intact: the weak near-zero alignment reported below is a rule/distribution issue, not a Dataset slicing issue. A separate horizon-invariance diagnostic now checks that changing `pred_len` does not change the unique absolute event timestamps. Empty event masks are treated as `not_applicable_empty_mask`, not as zero-error improvements.

## Offset Predictability

- Predictable from train/val: `False`.
- Post-hoc test regime: `True`.
- Test offsets outside train/val range: `True`.
- Train/val anchor offsets: `[0.0, -1.0, 3.0, 6.0, -1.0, -1.0, 2.0, 0.0]`.
- Test anchor offsets: `[29.0, 34.0]`.
- Best train/val predictor: `train_val_mean` with MAE `30.50` steps.
- Weak train/val-predictable channels by heuristic: `HUFL, MULL`.
- Unpredictable channels: `HULL, MUFL, LUFL, LULL, OT`.

## Per Split

| Split | Rule Mask Sum | True Event Sum | Best Lag Steps | Best Lag Hours | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 4032 | 14428 | -10 | -2.50 | 0.0611 | 0.0168 | 0.0263 |
| val | 1344 | 4344 | 24 | 6.00 | 0.0848 | 0.0262 | 0.0401 |
| test | 1344 | 4572 | 137 | 34.25 | 0.1161 | 0.0341 | 0.0527 |

## Unique Timestamp Metric

- Repeated event points: `129024`.
- Unique event timestamps: `1344`.
- Repeat factor: `96.00`.
- Repeated event MSE: `0.621358`.
- Unique event MSE: `0.551033`.

## Stable Channels

none

## Unstable Channels

HUFL, HULL, MUFL, MULL, LUFL, LULL, OT

## Calendar Anchor Samples

### train
| Expected Timestamp | Nearest True Event | Offset Steps | Offset Hours | True Near-zero Ratio | Zero Prior MSE |
|---|---|---:|---:|---:|---:|
| 2016-07-01 00:00:00 | 2016-07-01 00:00:00 | 0 | 0.00 | 0.0893 | 3.550425 |
| 2016-09-01 00:00:00 | 2016-08-31 23:45:00 | -1 | -0.25 | 0.0208 | 5.500836 |
| 2016-11-01 00:00:00 | 2016-11-01 00:45:00 | 3 | 0.75 | 0.0551 | 1.620110 |
| 2017-01-01 00:00:00 | 2017-01-01 01:30:00 | 6 | 1.50 | 0.0580 | 3.259658 |
| 2017-03-01 00:00:00 | 2017-02-28 23:45:00 | -1 | -0.25 | 0.0610 | 1.722513 |
### val
| Expected Timestamp | Nearest True Event | Offset Steps | Offset Hours | True Near-zero Ratio | Zero Prior MSE |
|---|---|---:|---:|---:|---:|
| 2017-07-01 00:00:00 | 2017-07-01 00:30:00 | 2 | 0.50 | 0.0729 | 2.133592 |
| 2017-09-01 00:00:00 | 2017-09-01 00:00:00 | 0 | 0.00 | 0.0818 | 1.859813 |
### test
| Expected Timestamp | Nearest True Event | Offset Steps | Offset Hours | True Near-zero Ratio | Zero Prior MSE |
|---|---|---:|---:|---:|---:|
| 2017-11-01 00:00:00 | 2017-11-01 07:15:00 | 29 | 7.25 | 0.1146 | 3.132842 |
| 2018-01-01 00:00:00 | 2018-01-01 08:30:00 | 34 | 8.50 | 0.0446 | 3.339835 |
