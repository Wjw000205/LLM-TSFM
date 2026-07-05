"""Create an offline LLM prompt from train-only mining artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_TEMPLATE = """# Train-Only Dataset Rule Mining Prompt

You are a dataset-level characteristic miner for time-series forecasting.

Hard constraints:
- Do not predict future values.
- Do not use validation or test information.
- Only infer dataset-level temporal rules from training evidence.
- Prefer structured rules that can be converted into masks, features, and differentiable loss.
- If evidence is weak, lower confidence or reject the rule.
- Return JSON only.

Expected JSON shape:

```json
{{
  "dataset_name": "{dataset_name}",
  "analysis_scope": "train_only",
  "patterns": [
    {{
      "name": "periodic_zero_day",
      "type": "zero_event",
      "description": "Values are close to zero on a recurring calendar window.",
      "condition": {{
        "kind": "calendar_periodic",
        "anchor": "YYYY-MM-DD HH:MM:SS",
        "month_interval": 2,
        "day": 1
      }},
      "affected_variables": "all",
      "time_range": "whole_day",
      "support_count": 1,
      "evidence_windows": [["YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS"]],
      "confidence": 0.8,
      "losses": {{
        "event_weighted_mse": {{"enabled": true, "weight": 5.0}},
        "zero_consistency": {{"enabled": true, "weight": 1.0}}
      }},
      "features": {{
        "event_mask": true,
        "days_to_event": true,
        "rule_confidence": true
      }}
    }}
  ],
  "selected_loss_templates": ["event_weighted_mse", "zero_consistency"],
  "selected_rule_features": ["event_mask", "days_to_event", "rule_confidence"],
  "warnings": [],
  "confidence": 0.8
}}
```

Train-only dataset summary:

```json
{summary_json}
```

Heuristic candidate rules for evidence, not final truth:

```json
{candidate_json}
```

Train-only figure paths:

{figure_paths}
"""


def build_prompt(
    summary_path: str,
    output_path: str | None = None,
    candidate_rules_path: str | None = None,
    figures_dir: str | None = None,
):
    """Build a prompt from summary, optional candidate JSON, and optional figure paths."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    candidates = _read_json(candidate_rules_path)
    figures = _figure_paths(figures_dir)
    dataset_name = summary.get("dataset_name") or summary.get("dataset") or "unknown"
    prompt = PROMPT_TEMPLATE.format(
        dataset_name=dataset_name,
        summary_json=json.dumps(summary, indent=2),
        candidate_json=json.dumps(candidates, indent=2),
        figure_paths="\n".join(f"- {path}" for path in figures) if figures else "- No figures generated.",
    )
    output = Path(output_path) if output_path else Path(summary_path).with_name("llm_prompt.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prompt, encoding="utf-8")
    return prompt


def _read_json(path: str | None):
    if not path:
        return {}
    candidate_path = Path(path)
    if not candidate_path.exists():
        return {}
    return json.loads(candidate_path.read_text(encoding="utf-8"))


def _figure_paths(figures_dir: str | None):
    if not figures_dir:
        return []
    path = Path(figures_dir)
    if not path.exists():
        return []
    return [str(item) for item in sorted(path.glob("*.png"))]


def main():
    parser = argparse.ArgumentParser(description="Build an offline LLM prompt from train-only artifacts.")
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--candidate_rules_path", default=None)
    parser.add_argument("--figures_dir", default=None)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    build_prompt(**vars(args))


if __name__ == "__main__":
    main()

