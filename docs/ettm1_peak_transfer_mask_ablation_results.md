# ETTm1 Peak-Transfer Mask Ablation Results

## Scope

This ablation checks whether the event-window gains come from the correct event mask rather than arbitrary sparse correction. It evaluates pred_len 96 and 192 using existing baseline and loss-expert predictions.

## Results

| Experiment | pred_len | Overall MSE | Event MSE | Non-event MSE | Event Reduction | Non-event Delta | Mask Type |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 96 | 0.307031 | 0.051324 | 0.307242 | 0.00% | 0.000000 | none |
| correct_gated_mask | 96 | 0.307010 | 0.025899 | 0.307242 | 49.54% | 0.000000 | correct |
| shuffled_event_mask | 96 | 0.307020 | 0.051326 | 0.307231 | -0.00% | -0.000011 | shuffled |
| shifted_wrong_mask_24h | 96 | 0.307020 | 0.051324 | 0.307231 | 0.00% | -0.000011 | shifted_by_96_steps |
| random_sparse_mask_same_ratio | 96 | 0.307034 | 0.051310 | 0.307245 | 0.03% | 0.000003 | random_same_ratio |
| no_gate_global_transfer | 96 | 0.310127 | 0.025899 | 0.310362 | 49.54% | 0.003120 | global_no_gate |
| baseline | 192 | 0.339500 | 0.065048 | 0.339729 | 0.00% | 0.000000 | none |
| correct_gated_mask | 192 | 0.339479 | 0.039033 | 0.339729 | 39.99% | 0.000000 | correct |
| shuffled_event_mask | 192 | 0.339493 | 0.065031 | 0.339722 | 0.03% | -0.000008 | shuffled |
| shifted_wrong_mask_24h | 192 | 0.339497 | 0.065048 | 0.339726 | 0.00% | -0.000003 | shifted_by_96_steps |
| random_sparse_mask_same_ratio | 192 | 0.339513 | 0.065023 | 0.339743 | 0.04% | 0.000013 | random_same_ratio |
| no_gate_global_transfer | 192 | 0.354245 | 0.039033 | 0.354507 | 39.99% | 0.014778 | global_no_gate |

## Conclusion

The correct event mask is the only sparse gating variant that substantially improves the true event-window MSE. Shuffled, shifted, and random sparse masks do not produce stable event gains. Global transfer improves event windows but damages non-event regions, which explains why event gating is needed.

## Artifacts

- `analysis/evaluate_peak_transfer_mask_ablation.py`
- `scripts/run_ettm1_peak_transfer_mask_ablation.ps1`
- `artifacts/core_results/ettm1_peak_transfer_mask_ablation_summary.csv`
- `artifacts/core_results/ettm1_peak_transfer_mask_ablation_summary.json`
