"""Factory for loss functions."""

from __future__ import annotations

from argparse import Namespace

from llm_rules.rule_parser import loss_config_from_rules, parse_llm_rules
from losses.dataset_aware_loss import DatasetAwareLoss


def build_loss(args: Namespace) -> DatasetAwareLoss:
    """Build the configured training criterion."""
    args_config = vars(args).copy()
    rules = parse_llm_rules(getattr(args, "llm_rule_path", None))
    config = loss_config_from_rules(rules)
    for key, value in args_config.items():
        if value is not None:
            config[key] = value

    if not _flag(config.get("use_dataset_aware_loss", False)):
        config.update(
            {
                "use_event_weighted_loss": False,
                "use_zero_consistency_loss": False,
                "use_peak_shape_loss": False,
                "use_diff_loss": False,
                "use_freq_loss": False,
            }
        )
    return DatasetAwareLoss(config)


def _flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)
