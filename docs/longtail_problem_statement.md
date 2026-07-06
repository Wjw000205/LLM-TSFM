# Long-Tail Temporal Event Problem Statement

## Scope

This project does not aim to claim general forecasting SOTA. The target problem is long-tail temporal events that are easy for average-loss training to ignore because they occupy a tiny fraction of all prediction elements.

## Motivation

In the ETTm1 gated peak-transfer experiments, event elements account for about 0.08% of all forecast elements under the current metric path. A model can therefore fail on important event windows while still looking acceptable under overall MSE. This makes event-window MSE, non-event MSE, event ratio, and overall guardrails necessary parts of the evaluation.

## Method Positioning

Gated peak-transfer uses an LLM/GPT-derived rule to identify sparse event windows. The event-specialized prediction is applied only inside those windows; all non-event predictions fall back to the same DLinear baseline. This isolates long-tail correction from the much larger non-event region.

## Evaluation Requirements

Core evaluation should include:

- Event-window MSE
- Non-event MSE
- Event ratio
- Overall MSE guardrail
- Correct-mask ablation against shuffled, shifted, and random masks
- Seed stability across repeated runs

SOTA comparison is not the main line of evidence. If included, it should be treated as an appendix or sanity check rather than the central claim.

## Main Claim

The main claim is that, under a fixed backbone and strict overall-MSE guardrail, sparse event gating can substantially reduce long-tail event-window error without damaging the non-event region. The current ETTm1 multi-horizon result supports that claim across pred_len 96, 192, 336, and 720.
