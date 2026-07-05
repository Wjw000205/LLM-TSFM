# Long-Tail Guardrail Results

Baseline overall MSE: `0.307031`
Baseline event MSE: `0.621358`
Overall-MSE tolerance: `5.00%`

## Selected Candidate

- `ettm1_longtail_medium_weights`: overall_mse=0.317920, event_mse=0.554032, accepted=True

## All Sweep Runs

| Experiment | Overall MSE | Event MSE | Overall Delta | Event Reduction | Accepted |
|---|---:|---:|---:|---:|---:|
| ettm1_guarded_longtail_finetune | 0.305944 | 0.582883 | -0.35% | 6.19% | True |
| ettm1_longtail_low_weights | 0.312149 | 0.589348 | 1.67% | 5.15% | True |
| ettm1_longtail_medium_weights | 0.317920 | 0.554032 | 3.55% | 10.84% | True |

## Pareto Frontier

| Experiment | Overall MSE | Event MSE | Overall Delta | Event Reduction | Accepted |
|---|---:|---:|---:|---:|---:|
| ettm1_guarded_longtail_finetune | 0.305944 | 0.582883 | -0.35% | 6.19% | True |
| ettm1_longtail_medium_weights | 0.317920 | 0.554032 | 3.55% | 10.84% | True |

## Interpretation

- A candidate is accepted only if it stays within the overall-MSE guardrail and improves event-window MSE.
- For checkpoint selection, prefer a validation-split baseline metrics file; using a test-split baseline can trigger fallback because validation MSE is not directly comparable.
- Hard intervention remains an oracle upper bound, not a deployable long-tail result.
