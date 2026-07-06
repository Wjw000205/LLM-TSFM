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
