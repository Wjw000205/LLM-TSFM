# ETTm1 Peak-Transfer Seed Stability

## Scope

Seed stability is prepared for pred_len 96, 192, and 336 with seeds 2021, 2022, and 2023. Pred_len 720 is optional because its long horizon makes repeated runs more expensive.

## How To Run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_ettm1_peak_transfer_seed_stability.ps1
```

The script trains baseline and loss-expert models for each seed, evaluates gated predictions, and writes:

- `artifacts/core_results/ettm1_peak_transfer_seed_stability.csv`
- `artifacts/core_results/ettm1_peak_transfer_seed_stability.json`

## Output Fields

| Field | Meaning |
|---|---|
| `pred_len` | Forecast horizon |
| `seed` | Random seed |
| `baseline_overall_mse` | Pure DLinear overall MSE |
| `gated_overall_mse` | Gated peak-transfer overall MSE |
| `baseline_event_mse` | Pure DLinear event-window MSE |
| `gated_event_mse` | Gated peak-transfer event-window MSE |
| `event_reduction_pct` | Relative event-window MSE reduction |
| `non_event_delta` | Gated non-event MSE minus baseline non-event MSE |
| `status` | `guarded` when validation selected `guarded_event_mse` |

## Current Status

The script and summarizer are implemented, but the multi-seed training sweep has not been executed in this commit. The next validation step is to run the script and report mean/std for overall MSE, event MSE, event reduction, and non-event delta.

## Artifacts

- `scripts/run_ettm1_peak_transfer_seed_stability.ps1`
- `analysis/summarize_peak_transfer_seed_stability.py`
