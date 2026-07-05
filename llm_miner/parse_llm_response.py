"""Parse an offline LLM JSON response into a rules file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_rules.rule_schema import LLMRules


def parse_response(response_path: str, output_path: str):
    """Extract and validate rule JSON from a response file."""
    text = Path(response_path).read_text(encoding="utf-8").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    payload = json.loads(text)
    rules = LLMRules.from_dict(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rules.to_dict(), indent=2), encoding="utf-8")
    return rules


def main():
    parser = argparse.ArgumentParser(description="Parse offline LLM response into validated rule JSON.")
    parser.add_argument("--response_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    parse_response(**vars(args))


if __name__ == "__main__":
    main()

