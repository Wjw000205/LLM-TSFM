# Multidataset GPT-5.5 Peak-Transfer Full-Horizon Results

## Status

The previous full-horizon multidataset table has been invalidated.

Reason: the ETTh1/ETTh2/ETTm2 96/192/336/720 sweep reused one dataset-level GPT peak-transfer rule file across multiple prediction horizons. That violates the current experiment contract: each `pred_len` must use its own generated rule/config because the long-tail event hypothesis and loss setting are horizon dependent.

## Updated Contract

For every `(dataset, pred_len)` pair, the training script now expects a horizon-specific rule file:

```text
llm_rules/generated_rules/{DATA}_p{PRED_LEN}_peak_transfer_rules.json
```

Example:

```text
llm_rules/generated_rules/ETTh1_p336_peak_transfer_rules.json
```

The rule generation report is also horizon-specific:

```text
artifacts/core_results/{data}_p{pred_len}_llm_rule_generation_report.json
```

## Enforcement

`scripts/run_multihorizon_gpt55_peak_transfer.ps1` now generates or checks the per-horizon rule file before each baseline/expert/gated run.

`analysis/summarize_multidataset_peak_transfer_full_horizon.py` refuses to summarize stale results whose `config.json` does not point to the expected horizon-specific rule path. This prevents old shared-config runs from being reported as valid full-horizon evidence.

## Next Valid Run

Regenerate rules and rerun the sweep with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& 'scripts\run_multihorizon_gpt55_peak_transfer.ps1' -Datasets @('ETTh1','ETTh2','ETTm2') -PredLens @(96,192,336,720) -GenerateRules 1 -RuleModel 'gpt-5.5'"
```

Then regenerate the summary:

```powershell
& 'C:\Users\33932\.conda\envs\my_fram\python.exe' analysis/summarize_multidataset_peak_transfer_full_horizon.py
```

The summary outputs will be recreated at:

- `artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.csv`
- `artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.json`
- `docs/multidataset_full_horizon_peak_transfer_results.md`
