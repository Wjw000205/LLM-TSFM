# GPT-5.5 Long-tail Loss Results

## Scope

This run corrects the LLM role back to the intended path:

- GPT-5.5 mines sparse long-tail event hypotheses from the train profile.
- GPT-5.5 emits mask-conditioned loss hypotheses.
- Training consumes those hypotheses through `DatasetAwareLoss`.
- GPT-5.5 is not used as a verifier, calibrated prior, oracle, or inference-time prediction gate.

## ETTm1 Results

| Experiment | Rule Scope | Overall MSE | Event MSE | Zero MSE | Peak MSE | Notes |
|---|---:|---:|---:|---:|---:|---|
| `longtail_loss_baseline` | zero+peak | 0.307031 | 0.385221 | 0.496521 | 0.051324 | Pure DLinear on the earlier transferable zero+peak mask |
| `longtail_generated_loss_guarded` | zero+peak | 0.416540 | 0.667853 | 0.874576 | 0.047684 | Raw GPT loss weights fail the overall guardrail |
| `longtail_finetune_w1_guarded` | zero+peak | 0.311460 | 0.465916 | 0.608814 | 0.037223 | Overall is guarded, but zero dominates event degradation |
| `peak_only_baseline` | peak only | 0.307031 | 0.051324 | 0.000000 | 0.051324 | Pure DLinear on the GPT-5.5 peak hypothesis |
| `peak_only_finetune_generated_loss_guarded` | peak only | 0.310127 | 0.025899 | 0.000000 | 0.025899 | Peak event MSE improves by 49.5% within the 3% overall guardrail |

## Conclusion

The useful LLM signal on ETTm1 is the peak long-tail loss hypothesis, not the zero timestamp hypothesis. Zero-event rules are now kept as explicit train evidence windows unless the support is strong enough to justify recurrence; this prevents one-off zero observations from becoming false-positive future events.

The best current result is `peak_only_finetune_generated_loss_guarded`: peak/event MSE improves from 0.051324 to 0.025899, while overall MSE changes from 0.307031 to 0.310127.

Figures:

- `artifacts/figures/gpt55_peak_longtail_loss_regions/manifest.json`
- `artifacts/figures/gpt55_peak_longtail_loss_regions/gpt55_event_examples_top5_OT.png`

## Multidataset Check

The same direction was checked on ETTh1, ETTh2, and ETTm2 with dataset-specific GPT-5.5 peak hypotheses. The gated variant preserves the baseline outside event masks and applies the loss expert only inside event masks.

| Dataset | Baseline Overall MSE | Gated Overall MSE | Baseline Event MSE | Gated Event MSE | Event Reduction |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 0.383827 | 0.383802 | 0.136563 | 0.088367 | 35.29% |
| ETTh2 | 0.292610 | 0.292602 | 0.101615 | 0.097232 | 4.31% |
| ETTm2 | 0.170001 | 0.169998 | 0.025562 | 0.018821 | 26.37% |

Full report: `docs/multidataset_gpt55_peak_transfer_results.md`.
