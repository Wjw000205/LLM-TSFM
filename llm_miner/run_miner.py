"""Unified train-only offline LLM mining pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_miner.build_dataset_summary import build_summary
from llm_miner.build_llm_prompt import build_prompt
from llm_miner.build_visualization import build_visualizations
from llm_miner.common import default_output_dir
from llm_miner.detect_rule_candidates import detect_candidates
from llm_miner.parse_llm_response import parse_response
from llm_miner.validate_rules import validate_rules


def run_miner(
    data: str,
    root_path: str,
    data_path: str,
    features: str = "M",
    target: str = "OT",
    seq_len: int = 96,
    pred_len: int = 96,
    output_dir: str | None = None,
    near_zero_eps: float = 1e-5,
    llm_response_path: str | None = None,
):
    """Run the train-only miner without calling any LLM API."""
    out_dir = default_output_dir(data, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "dataset_summary.json"
    candidate_path = out_dir / "candidate_rules.json"
    prompt_path = out_dir / "llm_prompt.md"

    build_summary(
        root_path=root_path,
        data_path=data_path,
        data=data,
        features=features,
        target=target,
        seq_len=seq_len,
        pred_len=pred_len,
        output_path=str(summary_path),
        near_zero_eps=near_zero_eps,
    )
    detect_candidates(
        root_path=root_path,
        data_path=data_path,
        data=data,
        features=features,
        target=target,
        seq_len=seq_len,
        pred_len=pred_len,
        output_dir=str(out_dir),
        near_zero_eps=near_zero_eps,
    )
    figure_paths = []
    try:
        figure_paths = build_visualizations(
            root_path=root_path,
            data_path=data_path,
            data=data,
            features=features,
            target=target,
            seq_len=seq_len,
            output_dir=str(out_dir),
            candidate_rules_path=str(candidate_path),
        )
    except Exception as exc:
        (out_dir / "figure_generation_warning.txt").write_text(str(exc), encoding="utf-8")

    build_prompt(
        summary_path=str(summary_path),
        candidate_rules_path=str(candidate_path),
        figures_dir=str(out_dir / "figures"),
        output_path=str(prompt_path),
    )

    outputs = {
        "summary_path": summary_path,
        "candidate_rules_path": candidate_path,
        "prompt_path": prompt_path,
        "figures_dir": out_dir / "figures",
        "figure_paths": figure_paths,
    }

    if llm_response_path:
        rule_path = Path("llm_rules") / "generated_rules" / f"{data}_rules.json"
        parse_response(llm_response_path, output_rule_path=str(rule_path))
        validate_rules(
            rule_path=str(rule_path),
            data=data,
            root_path=root_path,
            data_path=data_path,
            features=features,
            target=target,
            seq_len=seq_len,
            output_dir=str(out_dir),
        )
        outputs["rule_path"] = rule_path
        outputs["validation_report_path"] = out_dir / "rule_validation_report.json"
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run train-only offline LLM miner.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--near_zero_eps", type=float, default=1e-5)
    parser.add_argument("--llm_response_path", default=None)
    args = parser.parse_args()
    run_miner(**vars(args))


if __name__ == "__main__":
    main()

