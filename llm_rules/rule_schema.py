"""Typed schema objects for LLM-generated dataset rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RulePattern:
    """One dataset-level pattern mined before training."""

    name: str
    type: str
    condition: dict[str, Any]
    affected_variables: str | list[str] = "all"
    time_range: str = "single_step"
    losses: dict[str, dict[str, Any]] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RulePattern":
        return cls(
            name=str(payload["name"]),
            type=str(payload.get("type", "event")),
            condition=dict(payload.get("condition", {})),
            affected_variables=payload.get("affected_variables", "all"),
            time_range=str(payload.get("time_range", "single_step")),
            losses=dict(payload.get("losses", {})),
            features=dict(payload.get("features", {})),
        )


@dataclass
class LLMRules:
    """Container for all rules associated with one dataset."""

    dataset_name: str
    patterns: list[RulePattern] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LLMRules":
        return cls(
            dataset_name=str(payload.get("dataset_name", "unknown")),
            patterns=[RulePattern.from_dict(item) for item in payload.get("patterns", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "patterns": [
                {
                    "name": pattern.name,
                    "type": pattern.type,
                    "condition": pattern.condition,
                    "affected_variables": pattern.affected_variables,
                    "time_range": pattern.time_range,
                    "losses": pattern.losses,
                    "features": pattern.features,
                }
                for pattern in self.patterns
            ],
        }

