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
    description: str | None = None
    support_count: int | None = None
    evidence_windows: list[list[str]] = field(default_factory=list)
    confidence: float | None = None

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
            description=payload.get("description"),
            support_count=payload.get("support_count"),
            evidence_windows=list(payload.get("evidence_windows", [])),
            confidence=payload.get("confidence"),
        )


@dataclass
class LLMRules:
    """Container for all rules associated with one dataset."""

    dataset_name: str
    patterns: list[RulePattern] = field(default_factory=list)
    analysis_scope: str | None = None
    warnings: list[str] = field(default_factory=list)
    confidence: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LLMRules":
        return cls(
            dataset_name=str(payload.get("dataset_name", "unknown")),
            patterns=[RulePattern.from_dict(item) for item in payload.get("patterns", [])],
            analysis_scope=payload.get("analysis_scope"),
            warnings=list(payload.get("warnings", [])),
            confidence=payload.get("confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "dataset_name": self.dataset_name,
            "patterns": [
                _without_none(
                    {
                    "name": pattern.name,
                    "type": pattern.type,
                    "description": pattern.description,
                    "condition": pattern.condition,
                    "affected_variables": pattern.affected_variables,
                    "time_range": pattern.time_range,
                    "support_count": pattern.support_count,
                    "evidence_windows": pattern.evidence_windows,
                    "confidence": pattern.confidence,
                    "losses": pattern.losses,
                    "features": pattern.features,
                    }
                )
                for pattern in self.patterns
            ],
        }
        if self.analysis_scope is not None:
            payload["analysis_scope"] = self.analysis_scope
        if self.warnings:
            payload["warnings"] = self.warnings
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


def _without_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None and value != []}
