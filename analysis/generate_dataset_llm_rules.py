"""Generate dataset-specific LLM rule files before training.

This script builds a compact profile for the current dataset and asks an
OpenAI-compatible chat completion endpoint to return rule JSON accepted by the
local ``llm_rules`` parser. The CLI defaults to train-only profiling so rules
are not selected with validation/test information.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_miner.common import train_borders
from llm_rules.rule_parser import parse_llm_rules


SUPPORTED_CONDITION_KINDS = {"calendar_periodic", "calendar_window", "hourly", "weekday"}
SUPPORTED_LOSS_NAMES = {"event_weighted_mse", "zero_consistency", "peak_shape", "diff", "frequency"}
LOSS_ALIASES = {
    "event_window_mse": "event_weighted_mse",
    "event_masked_mse": "event_weighted_mse",
    "event_masked_mae": "event_weighted_mse",
    "peak_window_mse": "peak_shape",
    "peak_window_mae": "peak_shape",
    "zero_event_mse": "zero_consistency",
}


def build_dataset_profile(
    data: str,
    root_path: str,
    data_path: str,
    features: str = "M",
    target: str = "OT",
    split: str = "all",
    seq_len: int = 96,
    pred_len: int = 96,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Build a dataset-specific profile for LLM rule generation.

    ``split='train'`` is used by the CLI to avoid leakage. ``split='all'`` is
    kept for local diagnostics and small unit tests.
    """
    csv_path = Path(root_path) / data_path
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    value_columns = _select_value_columns(frame, date_col, features, target)

    if split == "train":
        start, end = train_borders(data, len(frame), seq_len)
        frame = frame.iloc[start:end].copy()
    elif split != "all":
        raise ValueError("split must be 'train' or 'all'")

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[:max_rows].copy()

    numeric = frame[value_columns].apply(pd.to_numeric, errors="coerce")
    dates = frame[date_col]
    freq = pd.infer_freq(dates)

    profile = {
        "dataset_name": data,
        "analysis_scope": "train_only" if split == "train" else "all_rows_profile",
        "split": split,
        "source_csv": str(csv_path),
        "row_count": int(len(frame)),
        "columns": value_columns,
        "target": target,
        "features": features,
        "forecast_horizon": {
            "seq_len": int(seq_len),
            "pred_len": int(pred_len),
        },
        "time": {
            "date_column": date_col,
            "start": str(dates.iloc[0]),
            "end": str(dates.iloc[-1]),
            "freq_inferred": _normalize_freq(freq),
        },
        "statistics": _statistics(numeric),
        "calendar_target_mean": _calendar_target_mean(frame, date_col, target),
        "candidate_windows": _candidate_windows(frame, date_col, numeric, target),
    }
    return profile


def extract_json_object(content: str) -> str:
    """Extract the first JSON object from a chat response."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    return text[start : end + 1]


def normalize_llm_rule_payload(
    payload: dict[str, Any],
    dataset_name: str,
    allow_zero_consistency: bool = False,
) -> dict[str, Any]:
    """Normalize LLM JSON into the local rule schema."""
    normalized = dict(payload)
    normalized["dataset_name"] = dataset_name
    normalized["analysis_scope"] = "llm_dataset_specific_pretraining"
    warnings_list = list(normalized.get("warnings", []))
    patterns = []
    for idx, item in enumerate(normalized.get("patterns", [])):
        pattern = dict(item)
        pattern.setdefault("name", f"dataset_specific_rule_{idx}")
        pattern.setdefault("type", "event")
        pattern.setdefault("condition", {"kind": "calendar_periodic", "anchor": "1970-01-01 00:00:00"})
        pattern.setdefault("affected_variables", "all")
        pattern.setdefault("time_range", "single_step")

        features = _coerce_feature_mapping(pattern.get("features", {}))
        features["event_mask"] = True
        features["days_to_event"] = True
        pattern["features"] = features

        losses = _coerce_loss_mapping(pattern.get("losses", {}))
        if "zero_consistency" in losses and not allow_zero_consistency:
            losses.pop("zero_consistency")
            warnings_list.append(
                f"Rule '{pattern['name']}' emitted zero_consistency, which is disabled for GPT-generated "
                "loss hypotheses by default; use event_weighted_mse for near-zero long-tail supervision."
            )
        if not losses:
            losses["event_weighted_mse"] = {"enabled": True, "weight": 5.0}
        pattern["losses"] = losses
        _downgrade_weak_zero_recurrence(pattern, warnings_list)

        kind = pattern.get("condition", {}).get("kind")
        if kind not in SUPPORTED_CONDITION_KINDS:
            warnings_list.append(f"Rule '{pattern['name']}' uses unsupported condition kind '{kind}'.")
        patterns.append(pattern)
    normalized["patterns"] = patterns
    normalized["warnings"] = warnings_list
    parse_llm_rules(normalized)
    return normalized


def _coerce_feature_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return {value: True}
    if isinstance(value, list):
        return {str(item): True for item in value}
    return {}


def _coerce_loss_mapping(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        losses = {}
        for key, payload in value.items():
            name = _canonical_loss_name(str(key))
            if name is not None:
                losses[name] = _coerce_loss_payload(name, payload)
        return losses
    if isinstance(value, str):
        name = _canonical_loss_name(value)
        return {name: _default_loss_payload(name)} if name is not None else {}
    if isinstance(value, list):
        losses = {}
        for item in value:
            if isinstance(item, Mapping):
                name = _canonical_loss_name(str(item.get("name") or item.get("loss") or item.get("type") or ""))
                if not name:
                    continue
                losses[name] = _coerce_loss_payload(name, item)
            else:
                name = _canonical_loss_name(str(item))
                if name is not None:
                    losses[name] = _default_loss_payload(name)
        return losses
    return {}


def _canonical_loss_name(name: str) -> str | None:
    normalized = str(name).strip()
    if normalized in SUPPORTED_LOSS_NAMES:
        return normalized
    return LOSS_ALIASES.get(normalized)


def _coerce_loss_payload(name: str, payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        result = {str(key): value for key, value in payload.items() if key not in {"name", "loss", "type"}}
    else:
        result = {}
    result["enabled"] = bool(result.get("enabled", True))
    result["weight"] = float(result.get("weight", _default_loss_payload(name)["weight"]))
    return result


def _default_loss_payload(name: str) -> dict[str, Any]:
    return {"enabled": True, "weight": 5.0 if name == "event_weighted_mse" else 1.0}


def _downgrade_weak_zero_recurrence(pattern: dict[str, Any], warnings_list: list[str]) -> None:
    """Keep one-off zero hypotheses as train-evidence windows instead of recurrence."""
    if pattern.get("type") != "zero_event":
        return
    support_count = int(pattern.get("support_count", 0) or 0)
    if support_count >= 2:
        return
    condition = pattern.get("condition", {})
    if not isinstance(condition, Mapping):
        return
    if condition.get("kind") != "calendar_window" or "windows" in condition:
        return
    evidence_windows = _coerce_evidence_windows(pattern.get("evidence_windows"))
    if not evidence_windows:
        return
    pattern["condition"] = {"kind": "calendar_window", "windows": evidence_windows}
    warnings_list.append(
        f"Rule '{pattern['name']}' is a low-support zero_event; it is kept as explicit evidence windows "
        "instead of a recurring calendar rule."
    )


def _coerce_evidence_windows(value: Any) -> list[dict[str, str]]:
    windows: list[dict[str, str]] = []
    if not isinstance(value, list):
        return windows
    for item in value:
        if isinstance(item, Mapping):
            start = item.get("start") or item.get("date_start")
            end = item.get("end") or item.get("date_end")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue
        if start and end:
            windows.append({"start": str(start), "end": str(end)})
    return windows


def call_llm_rule_generator(
    profile: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 120,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat completion API and return normalized rules."""
    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate dataset-specific sparse long-tail event and loss-hypothesis rule JSON. "
                    "Use only the provided profile and obey the requested schema exactly. "
                    "Do not verify, calibrate, accept, or reject hypotheses. "
                    "Return JSON only."
                ),
            },
            {"role": "user", "content": _build_prompt(profile)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM rule generation failed with HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM rule generation failed: {exc}") from exc

    content = raw["choices"][0]["message"]["content"]
    payload = json.loads(extract_json_object(content))
    return normalize_llm_rule_payload(payload, dataset_name=str(profile["dataset_name"]))


def write_rule_outputs(
    rules: dict[str, Any],
    profile: dict[str, Any],
    output_rule_path: str,
    output_report_path: str,
) -> None:
    """Persist generated rules and an audit report without storing secrets."""
    rule_path = Path(output_rule_path)
    report_path = Path(output_report_path)
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rule_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    report = {
        "dataset_name": profile["dataset_name"],
        "analysis_scope": profile["analysis_scope"],
        "split": profile["split"],
        "source_csv": profile["source_csv"],
        "output_rule_path": str(rule_path),
        "num_patterns": len(rules.get("patterns", [])),
        "warnings": rules.get("warnings", []),
        "profile": profile,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _select_value_columns(frame: pd.DataFrame, date_col: str, features: str, target: str) -> list[str]:
    columns = [col for col in frame.columns if col != date_col]
    if features == "S":
        return [target]
    if target not in columns:
        raise ValueError(f"Target column '{target}' not found.")
    return [target] + [col for col in columns if col != target]


def _statistics(numeric: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in numeric.columns:
        series = numeric[col].dropna().astype(float)
        if series.empty:
            stats[col] = {}
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "q05": float(series.quantile(0.05)),
            "q50": float(series.quantile(0.50)),
            "q95": float(series.quantile(0.95)),
            "max": float(series.max()),
            "zero_ratio": float((series == 0).mean()),
        }
    return stats


def _calendar_target_mean(frame: pd.DataFrame, date_col: str, target: str) -> dict[str, dict[str, float]]:
    if target not in frame.columns:
        return {}
    values = pd.to_numeric(frame[target], errors="coerce")
    return {
        "hour": _group_mean(values, frame[date_col].dt.hour),
        "weekday": _group_mean(values, frame[date_col].dt.weekday),
        "month": _group_mean(values, frame[date_col].dt.month),
        "day": _group_mean(values, frame[date_col].dt.day),
    }


def _candidate_windows(
    frame: pd.DataFrame,
    date_col: str,
    numeric: pd.DataFrame,
    target: str,
    limit: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    windows: dict[str, list[dict[str, Any]]] = {"near_zero": [], "peaks": []}
    for col in numeric.columns:
        series = numeric[col].astype(float)
        std = float(series.std(ddof=0))
        eps = max(1e-6, std * 0.02)
        zero_hits = np.flatnonzero(series.abs().to_numpy() <= eps)
        for idx in zero_hits[:limit]:
            windows["near_zero"].append(
                {"variable": col, "timestamp": str(frame[date_col].iloc[idx]), "value": float(series.iloc[idx])}
            )
    if target in numeric:
        target_values = numeric[target].astype(float)
        high = float(target_values.quantile(0.95))
        low = float(target_values.quantile(0.05))
        hits = np.flatnonzero((target_values >= high).to_numpy() | (target_values <= low).to_numpy())
        for idx in hits[:limit]:
            windows["peaks"].append(
                {
                    "variable": target,
                    "timestamp": str(frame[date_col].iloc[idx]),
                    "value": float(target_values.iloc[idx]),
                    "low_threshold": low,
                    "high_threshold": high,
                }
            )
    return windows


def _group_mean(values: pd.Series, keys: pd.Series) -> dict[str, float]:
    grouped = values.groupby(keys).mean()
    return {str(key): float(value) for key, value in grouped.items()}


def _normalize_freq(freq: str | None) -> str | None:
    if freq is None:
        return None
    lowered = freq.lower()
    if lowered in {"h", "1h"}:
        return "h"
    return lowered


def _build_prompt(profile: dict[str, Any]) -> str:
    schema_example = {
        "dataset_name": str(profile.get("dataset_name", "DATASET")),
        "analysis_scope": "train_only",
        "patterns": [
            {
                "name": "periodic_zero_day",
                "type": "zero_event",
                "description": "Values are close to zero on a recurring sparse calendar window.",
                "condition": {"kind": "calendar_periodic", "anchor": "YYYY-MM-DD HH:MM:SS", "month_interval": 2, "day": 1},
                "affected_variables": "all",
                "time_range": "whole_day",
                "support_count": 1,
                "evidence_windows": [["YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS"]],
                "confidence": 0.8,
                "losses": {"event_weighted_mse": {"enabled": True, "weight": 5.0}},
                "features": {"event_mask": True, "days_to_event": True},
            }
        ],
        "warnings": [],
        "confidence": 0.8,
    }
    return (
        "Create sparse long-tail event and loss hypotheses for this dataset only. Do not reuse rules from any other dataset. "
        "The output must align with the repository's legacy rule contract used by the earlier ETTm1 rules.\n\n"
        "Hard constraints:\n"
        "- You are a hypothesis miner, not a verifier or final judge.\n"
        "- Use only train-profile evidence. Do not use validation or test information.\n"
        "- Treat forecast_horizon.pred_len as part of the experiment configuration; do not assume one horizon's rule config is valid for another horizon.\n"
        "- Do not validate, accept, reject, calibrate, or rank hypotheses inside the LLM output.\n"
        "- Return at most 3 patterns.\n"
        "- Each pattern must be sparse and localized; prefer window_hours <= 4, and use <= 8 only when the evidence window is wider.\n"
        "- Each pattern must name explicit affected_variables; do not use \"all\" unless all columns are directly present in the candidate evidence.\n"
        "- Generate candidate loss hypotheses even when confidence is low.\n"
        "- The trainer will consume your loss hypotheses as mask-conditioned dataset-aware losses; metrics are reported after training.\n"
        "- Do not output deterministic priors, alpha values, oracle labels, recommended enable/disable decisions, or calibrated values.\n"
        "- Do not output zero_consistency for GPT-generated hypotheses. It is a prior-like ablation, not the default LLM loss route.\n"
        "- Do not return an empty patterns list when near_zero or peak candidate windows exist in the profile.\n"
        "- Prefer timestamp-transferable templates beyond the train split only when the train evidence supports transfer.\n"
        "- Do not invent recurring calendar rules from one-off near_zero evidence.\n"
        "- For one-off near_zero evidence, use explicit calendar_window windows from evidence_windows so it acts as a train loss hypothesis only.\n"
        "- Prefer the previous ETTm1-style calendar_periodic zero_event contract when there is recurring or suspected recurring near-zero/shutdown evidence.\n"
        "- Do not encode ordinary seasonality, hour-of-day, weekday, or month regimes as events.\n"
        "- Only create a pattern when it marks a sparse, localized, actionable event window such as near-zero shutdowns, sensor outages, maintenance-like intervals, or rare peaks.\n"
        "- If the profile only supports normal calendar regimes and has no near_zero or peak candidate windows, return an empty patterns list and explain this in warnings.\n"
        "- Do not put recommended, weighting, rationale, or free-form metadata inside losses or features.\n"
        "- Put explanations in description, warnings, confidence, support_count, and evidence_windows only.\n"
        "- losses keys must be chosen only from: event_weighted_mse, zero_consistency, peak_shape, diff, frequency.\n"
        "- features keys must be chosen only from: event_mask, days_to_event, peak_mask, zero_event_mask, rule_confidence, support_count.\n"
        "- Prefer event_weighted_mse only for weak hypotheses. Add zero_consistency only when labels are truly close to raw zero inside the event.\n"
        "- For near_zero candidate windows, use type zero_event and usually enable event_weighted_mse; optionally enable zero_consistency only if the raw values are actually near zero.\n"
        "- For peak candidate windows, use type peak_event and usually enable event_weighted_mse plus peak_shape.\n\n"
        "Allowed condition templates:\n"
        "- calendar_periodic: {\"kind\": \"calendar_periodic\", \"anchor\": \"YYYY-MM-DD HH:MM:SS\", \"month_interval\": 2, \"day\": 1}\n"
        "- calendar_window: {\"kind\": \"calendar_window\", \"anchor\": \"YYYY-MM-DD HH:MM:SS\", \"center_day\": 1, \"center_hour\": 0, \"month_interval\": 2, \"window_hours\": 24}\n"
        "- calendar_window explicit: {\"kind\": \"calendar_window\", \"windows\": [{\"start\": \"YYYY-MM-DD HH:MM:SS\", \"end\": \"YYYY-MM-DD HH:MM:SS\"}]}\n"
        "- hourly: {\"kind\": \"hourly\", \"hour\": 13}\n"
        "- weekday: {\"kind\": \"weekday\", \"weekday\": 6}\n"
        "Do not use high_hours, low_hours, high_weekdays, low_weekdays, periods, calendar_field, month_sin, hour_sin, season_bucket, or calendar_target_mean.\n\n"
        "Legacy field snippets to copy exactly when applicable:\n"
        "- \"type\": \"zero_event\"\n"
        "- \"condition\": {\"kind\": \"calendar_periodic\", \"anchor\": \"YYYY-MM-DD HH:MM:SS\", \"month_interval\": 2, \"day\": 1}\n"
        "- \"losses\": {\"event_weighted_mse\": {\"enabled\": true, \"weight\": 5.0}}\n"
        "- \"features\": {\"event_mask\": true, \"days_to_event\": true}\n\n"
        "Return JSON only in this exact shape:\n"
        f"{json.dumps(schema_example, indent=2)}\n\n"
        f"Dataset profile:\n{json.dumps(profile, indent=2)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset-specific LLM rule JSON before training.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--profile_split", default="train", choices=["train", "all"])
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.2"))
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry_run_profile_only", type=int, default=0)
    parser.add_argument("--output_rule_path", default=None)
    parser.add_argument("--output_report_path", default=None)
    args = parser.parse_args()

    profile = build_dataset_profile(
        data=args.data,
        root_path=args.root_path,
        data_path=args.data_path,
        features=args.features,
        target=args.target,
        split=args.profile_split,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    output_rule_path = args.output_rule_path or f"./llm_rules/generated_rules/{args.data}_rules.json"
    output_report_path = (
        args.output_report_path
        or f"./artifacts/core_results/{args.data.lower()}_llm_rule_generation_report.json"
    )

    if args.dry_run_profile_only:
        empty_rules = normalize_llm_rule_payload(
            {"dataset_name": args.data, "patterns": [], "warnings": ["dry_run_profile_only"]},
            dataset_name=args.data,
        )
        write_rule_outputs(empty_rules, profile, output_rule_path, output_report_path)
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key env var '{args.api_key_env}'. Set it before generating dataset-specific LLM rules."
        )
    rules = call_llm_rule_generator(
        profile=profile,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
    )
    write_rule_outputs(rules, profile, output_rule_path, output_report_path)


if __name__ == "__main__":
    main()
