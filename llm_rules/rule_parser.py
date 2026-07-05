"""Read and validate offline LLM rule JSON files."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from llm_rules.rule_schema import LLMRules


LOSS_KEY_TO_WEIGHT = {
    "event_weighted_mse": "event_weight",
    "zero_consistency": "zero_weight",
    "peak_shape": "peak_weight",
    "diff": "diff_weight",
    "frequency": "freq_weight",
}


def parse_llm_rules(source: str | Path | dict[str, Any] | LLMRules | None) -> LLMRules | None:
    """Parse rule JSON from a path, dict, existing object, or ``None``."""
    if source is None or source == "":
        return None
    if isinstance(source, LLMRules):
        return source
    if isinstance(source, dict):
        return LLMRules.from_dict(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"LLM rule file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LLMRules.from_dict(payload)


def loss_config_from_rules(rules: LLMRules | None) -> dict[str, float | bool]:
    """Extract enabled loss switches and weights from parsed rules."""
    config: dict[str, float | bool] = {}
    if rules is None:
        return config

    for pattern in rules.patterns:
        for loss_name, payload in pattern.losses.items():
            if not payload.get("enabled", False):
                continue
            weight = float(payload.get("weight", 1.0))
            if loss_name == "event_weighted_mse":
                config["use_event_weighted_loss"] = True
                config["event_weight"] = max(float(config.get("event_weight", 0.0)), weight)
            elif loss_name == "zero_consistency":
                config["use_zero_consistency_loss"] = True
                config["zero_weight"] = max(float(config.get("zero_weight", 0.0)), weight)
            elif loss_name == "peak_shape":
                config["use_peak_shape_loss"] = True
                config["peak_weight"] = max(float(config.get("peak_weight", 0.0)), weight)
            elif loss_name == "diff":
                config["use_diff_loss"] = True
                config["diff_weight"] = max(float(config.get("diff_weight", 0.0)), weight)
            elif loss_name == "frequency":
                config["use_freq_loss"] = True
                config["freq_weight"] = max(float(config.get("freq_weight", 0.0)), weight)
            else:
                warnings.warn(
                    f"Unsupported loss '{loss_name}' in rule '{pattern.name}' was ignored.",
                    UserWarning,
                    stacklevel=2,
                )
    return config
