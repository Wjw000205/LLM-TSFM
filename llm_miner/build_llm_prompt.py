"""Create an offline LLM prompt from train-only summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_TEMPLATE = """You are analyzing only the TRAIN split of a time-series dataset.
Do not infer anything from validation or test data.

Return JSON rules for long-tail events using this schema:
{{
  "dataset_name": "...",
  "patterns": [
    {{
      "name": "...",
      "type": "zero_event|peak_event|calendar_event",
      "condition": {{"kind": "calendar_periodic|hourly|weekday"}},
      "affected_variables": "all",
      "time_range": "single_step|whole_day",
      "losses": {{}},
      "features": {{}}
    }}
  ]
}}

Train-only summary:
{summary}
"""


def build_prompt(summary_path: str, output_path: str):
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    prompt = PROMPT_TEMPLATE.format(summary=json.dumps(summary, indent=2))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prompt, encoding="utf-8")
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Build an offline LLM prompt from train-only summary JSON.")
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    build_prompt(**vars(args))


if __name__ == "__main__":
    main()

