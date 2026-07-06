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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_miner.common import train_borders
from llm_rules.rule_parser import parse_llm_rules


SUPPORTED_CONDITION_KINDS = {"calendar_periodic", "calendar_window", "hourly", "weekday"}


def build_dataset_profile(
    data: str,
    root_path: str,
    data_path: str,
    features: str = "M",
    target: str = "OT",
    split: str = "all",
    seq_len: int = 96,
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


def normalize_llm_rule_payload(payload: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    """Normalize LLM JSON into the local rule schema."""
    normalized = dict(payload)
    normalized["dataset_name"] = dataset_name
    normalized["analysis_scope"] = "llm_dataset_specific_pretraining"
    patterns = []
    for idx, item in enumerate(normalized.get("patterns", [])):
        pattern = dict(item)
        pattern.setdefault("name", f"dataset_specific_rule_{idx}")
        pattern.setdefault("type", "event")
        pattern.setdefault("condition", {"kind": "calendar_periodic", "anchor": "1970-01-01 00:00:00"})
        pattern.setdefault("affected_variables", "all")
        pattern.setdefault("time_range", "single_step")

        features = dict(pattern.get("features", {}))
        features["event_mask"] = True
        features["days_to_event"] = True
        pattern["features"] = features

        losses = dict(pattern.get("losses", {}))
        if not losses:
            losses["event_weighted_mse"] = {"enabled": True, "weight": 5.0}
        pattern["losses"] = losses

        kind = pattern.get("condition", {}).get("kind")
        if kind not in SUPPORTED_CONDITION_KINDS:
            warnings = list(normalized.get("warnings", []))
            warnings.append(f"Rule '{pattern['name']}' uses unsupported condition kind '{kind}'.")
            normalized["warnings"] = warnings
        patterns.append(pattern)
    normalized["patterns"] = patterns
    parse_llm_rules(normalized)
    return normalized


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
                    "You generate dataset-specific time-series rule JSON. "
                    "Use only the provided profile. Return JSON only."
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
    return (
        "Create rules for this dataset only. Do not reuse rules from any other dataset, "
        "including ETTm1, unless the profile itself supports the same timestamps and "
        "pattern. Supported condition kinds are calendar_periodic, calendar_window, "
        "hourly, and weekday. Include event_mask and days_to_event features. Avoid "
        "zero_consistency unless the profile shows true near-zero behavior.\n\n"
        "Return a JSON object with dataset_name, analysis_scope, patterns, warnings, "
        "and confidence. Each pattern must include name, type, condition, "
        "affected_variables, time_range, losses, features, confidence, and evidence_windows.\n\n"
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
