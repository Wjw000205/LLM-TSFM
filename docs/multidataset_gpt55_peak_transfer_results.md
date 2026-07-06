# Multidataset GPT-5.5 Peak-Transfer Results

## Scope

This run checks whether the current LLM-assisted long-tail loss path transfers beyond ETTm1. Each dataset uses its own GPT-5.5 generated analysis and rules; no ETTm1 rule is reused.

The raw GPT-5.5 rules for ETTh1, ETTh2, and ETTm2 mostly produced train-evidence windows. For test-time diagnostics, the retained transferable signal is the dataset-specific peak-event hypothesis. Zero-event hypotheses are excluded because earlier diagnostics showed that hard zero priors can be false positives and can worsen event MSE.

## Method

- Train a pure DLinear baseline for each dataset.
- Fine-tune a loss expert using the dataset-specific GPT-5.5 peak mask and `DatasetAwareLoss`.
- Build a rule-gated prediction:
  - outside event mask: use pure DLinear baseline;
  - inside event mask: use the loss expert.

This directly tests the target behavior: keep non-event predictions identical to baseline while applying the long-tail specialist only in the intended event region.

## Results

| Dataset | Baseline Overall MSE | Gated Overall MSE | Baseline Event MSE | Gated Event MSE | Event Reduction | Event Points |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.383827 | 0.383802 | 0.136563 | 0.088367 | 35.29% | 960 |
| ETTh2 | 0.292610 | 0.292602 | 0.101615 | 0.097232 | 4.31% | 3456 |
| ETTm2 | 0.170001 | 0.169998 | 0.025562 | 0.018821 | 26.37% | 3264 |

## Conclusion

The event/overall conflict is not inherent. When the loss expert is allowed to affect non-event regions, overall MSE can rise even if event MSE improves. With explicit event gating, non-event predictions stay at the DLinear baseline, while event regions inherit the loss expert improvement.

Across ETTh1, ETTh2, and ETTm2, the gated method improves event MSE and preserves overall MSE. ETTh2 shows the weakest event gain, so its GPT peak hypothesis is less useful than ETTh1 and ETTm2, but it still does not damage overall.

## Artifacts

- `artifacts/core_results/multidataset_gpt55_peak_transfer_gated_summary.csv`
- `artifacts/core_results/multidataset_gpt55_peak_transfer_gated_summary.json`
- `llm_rules/generated_rules/ETTh1_rules.json`
- `llm_rules/generated_rules/ETTh2_rules.json`
- `llm_rules/generated_rules/ETTm2_rules.json`
- `llm_rules/generated_rules/ETTh1_peak_transfer_rules.json`
- `llm_rules/generated_rules/ETTh2_peak_transfer_rules.json`
- `llm_rules/generated_rules/ETTm2_peak_transfer_rules.json`
- `artifacts/figures/etth1_gpt55_peak_transfer_regions/manifest.json`
- `artifacts/figures/etth2_gpt55_peak_transfer_regions/manifest.json`
- `artifacts/figures/ettm2_gpt55_peak_transfer_regions/manifest.json`
