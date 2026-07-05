"""Parse an offline LLM JSON response into a rule JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_rules.rule_schema import LLMRules


def parse_response(
    response_path: str,
    output_rule_path: str | None = None,
    output_path: str | None = None,
    error_report_path: str | None = None,
):
    """Extract and validate rule JSON from a saved LLM response file."""
    destination = output_rule_path or output_path
    text = Path(response_path).read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(_strip_code_fence(text))
        rules = LLMRules.from_dict(payload)
        output = Path(destination) if destination else Path("llm_rules") / "generated_rules" / f"{rules.dataset_name}_rules.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rules.to_dict(), indent=2), encoding="utf-8")
        return rules
    except Exception as exc:
        report = Path(error_report_path) if error_report_path else Path(response_path).with_name("error_report.txt")
        report.write_text(f"Failed to parse LLM response: {exc}\n", encoding="utf-8")
        raise


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser(description="Parse offline LLM response into validated rule JSON.")
    parser.add_argument("--response_path", required=True)
    parser.add_argument("--output_rule_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--error_report_path", default=None)
    args = parser.parse_args()
    parse_response(**vars(args))


if __name__ == "__main__":
    main()

