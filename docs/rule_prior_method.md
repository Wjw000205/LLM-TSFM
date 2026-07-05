# Rule Prior Fusion Method

LLM rule prior fusion treats the offline LLM output as executable temporal prior, not as generic trainable features.

The forecasting backbone first produces its normal prediction everywhere. At rule-triggered timestamps, a deterministic branch softly fuses the compiled prior into the forecast. For zero-event rules:

```text
pred_fused = pred_base + zero_mask * alpha * (zero_target - pred_base)
```

When `zero_mask=0`, the prediction is exactly unchanged. When `alpha=0`, the branch is equivalent to the baseline. When `alpha=1`, the branch becomes a hard zero prior at zero-event positions.

This path does not call an LLM during training or inference, and it does not add a trainable MLP. The LLM's role is to convert sparse temporal observations into executable priors: where an event occurs, what rule type applies, which variables are affected, and what prior target should be fused.

In Chinese terms: LLM 的作用不是提供一个普通 feature，而是把低样本长尾规律转化为可执行先验。模型不再从少量 event 样本中学习“什么时候发生事件”，而是在 LLM 指定的位置融合规则先验。

The following remain ablations:

- `output_rule_adapter`: trainable post-prediction residual correction.
- `intermediate_intervention`: trainable hidden-state intervention.
- `hard_intervention`: hard prior diagnostic, only an oracle upper bound if diagnosis verifies that masks and targets are valid.
- `dataset_aware_loss`: diagnostic baseline, because it can reduce event error while damaging overall error.
