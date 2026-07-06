# Long-Tail Temporal Event Problem Statement

## Scope

This project does not aim to claim general forecasting SOTA. The target problem is long-tail temporal events that are easy for average-loss training to ignore because they occupy a tiny fraction of all prediction elements.

## Motivation

In the ETTm1 gated peak-transfer experiments, event elements account for about 0.08% of all forecast elements under the current metric path. A model can therefore fail on important event windows while still looking acceptable under overall MSE. This makes event-window MSE, non-event MSE, event ratio, and overall guardrails necessary parts of the evaluation.

Event definition is horizon-independent. A rule defines absolute event timestamps from the raw timeline; `pred_len` only changes how many sliding windows repeatedly predict those timestamps and how many event elements appear under the `samples * pred_len * channels` metric denominator.

## Method Positioning

Gated peak-transfer uses an LLM/GPT-derived rule to identify sparse event windows. The event-specialized prediction is applied only inside those windows; all non-event predictions fall back to the same DLinear baseline. This isolates long-tail correction from the much larger non-event region.

## Evaluation Requirements

Core evaluation should include:

- Event-window MSE
- Non-event MSE
- Event ratio
- Unique event timestamp count
- Repeated event point count
- Horizon-invariance diagnostics for event masks
- Overall MSE guardrail
- Correct-mask ablation against shuffled, shifted, and random masks
- Seed stability across repeated runs

If `event_mask.sum() == 0`, event-window metrics are not applicable and should be reported as `NaN`/`not_applicable_empty_mask`. Empty masks must not be recorded as zero-error improvements.

SOTA comparison is not the main line of evidence. If included, it should be treated as an appendix or sanity check rather than the central claim.

## Main Claim

The main claim is that, under a fixed backbone and strict overall-MSE guardrail, sparse event gating can substantially reduce long-tail event-window error without damaging the non-event region. The current ETTm1 multi-horizon result supports that claim across pred_len 96, 192, 336, and 720 after the horizon-invariance audit. ETTh1/ETTh2/ETTm2 results produced with horizon-specific rule files should be treated as invalid until rerun with dataset-level horizon-independent event rules.
