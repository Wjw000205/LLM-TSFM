"""Command-line entry point for LLM-guided time-series forecasting."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from exp.exp_long_term_forecasting import ExpLongTermForecasting
from utils.tools import bool_flag, set_seed


def main():
    args = get_args()
    set_seed(args.seed)

    for itr in range(args.itr):
        run_args = copy.deepcopy(args)
        setting = (
            f"{run_args.task_name}_{run_args.model}_{run_args.data}_"
            f"ft{run_args.features}_sl{run_args.seq_len}_ll{run_args.label_len}_"
            f"pl{run_args.pred_len}_{run_args.des}_{itr}"
        )
        exp = ExpLongTermForecasting(run_args)
        if bool_flag(run_args.is_training):
            exp.train(setting)
            exp.test(setting, load_best=True)
        else:
            exp.test(setting, load_best=False)


def get_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser], description="LLM-guided dataset-aware forecasting")
    add = parser.add_argument
    add("--task_name", type=str, default=config.get("task_name", "long_term_forecast"))
    add("--is_training", type=int, default=config.get("is_training", 1))
    add("--model", type=str, default=config.get("model", "DLinear"))
    add("--data", type=str, default=config.get("data", "custom"))
    add("--root_path", type=str, default=config.get("root_path", "./dataset/"))
    add("--data_path", type=str, default=config.get("data_path", "data.csv"))
    add("--features", type=str, default=config.get("features", "M"), choices=["M", "S", "MS"])
    add("--target", type=str, default=config.get("target", "OT"))
    add("--freq", type=str, default=config.get("freq", "h"))
    add("--timeenc", type=int, default=config.get("timeenc", 0))

    add("--seq_len", type=int, default=config.get("seq_len", 96))
    add("--label_len", type=int, default=config.get("label_len", 48))
    add("--pred_len", type=int, default=config.get("pred_len", 96))
    add("--enc_in", type=int, default=config.get("enc_in", 7))
    add("--c_out", type=int, default=config.get("c_out", 7))
    add("--individual", type=int, default=config.get("individual", 0))
    add("--moving_avg", type=int, default=config.get("moving_avg", 25))
    add("--dlinear_init_avg", type=int, default=config.get("dlinear_init_avg", 0))
    add("--rnn_hidden_size", type=int, default=config.get("rnn_hidden_size", 64))
    add("--rnn_layers", type=int, default=config.get("rnn_layers", 1))
    add("--dropout", type=float, default=config.get("dropout", 0.0))

    add("--batch_size", type=int, default=config.get("batch_size", 32))
    add("--num_workers", type=int, default=config.get("num_workers", 0))
    add("--learning_rate", type=float, default=config.get("learning_rate", 0.0001))
    add("--train_epochs", type=int, default=config.get("train_epochs", 10))
    add("--patience", type=int, default=config.get("patience", 3))
    add("--early_stop_metric", type=str, default=config.get("early_stop_metric", "base_mse"), choices=["base_mse", "total_loss"])
    add("--itr", type=int, default=config.get("itr", 1))
    add("--des", type=str, default=config.get("des", "exp"))

    add("--use_zscore", type=int, default=config.get("use_zscore", 1))
    add("--use_revin", type=int, default=config.get("use_revin", 0))
    add("--use_llm_features", type=int, default=config.get("use_llm_features", 0))
    add("--use_standard_time_features", type=int, default=config.get("use_standard_time_features", 0))
    add("--use_llm_rule_features", type=int, default=config.get("use_llm_rule_features", None))
    add("--use_oracle_features", type=int, default=config.get("use_oracle_features", 0))
    add("--use_rule_adapter", type=int, default=config.get("use_rule_adapter", 0))
    add("--rule_adapter_hidden", type=int, default=config.get("rule_adapter_hidden", 32))
    add("--use_hard_intervention", type=int, default=config.get("use_hard_intervention", 0))
    add("--use_dataset_aware_loss", type=int, default=config.get("use_dataset_aware_loss", 0))
    add("--use_event_weighted_loss", type=int, default=config.get("use_event_weighted_loss", None))
    add("--use_zero_consistency_loss", type=int, default=config.get("use_zero_consistency_loss", None))
    add("--use_peak_shape_loss", type=int, default=config.get("use_peak_shape_loss", None))
    add("--use_diff_loss", type=int, default=config.get("use_diff_loss", None))
    add("--use_freq_loss", type=int, default=config.get("use_freq_loss", None))
    add("--event_weight", type=float, default=config.get("event_weight", None))
    add("--zero_weight", type=float, default=config.get("zero_weight", None))
    add("--peak_weight", type=float, default=config.get("peak_weight", None))
    add("--diff_weight", type=float, default=config.get("diff_weight", None))
    add("--freq_weight", type=float, default=config.get("freq_weight", None))
    add("--peak_window_size", type=int, default=config.get("peak_window_size", 2))
    add("--llm_rule_path", type=str, default=config.get("llm_rule_path", None))

    add("--use_gpu", type=int, default=config.get("use_gpu", 1))
    add("--device", type=str, default=config.get("device", "cuda:0"))
    add("--use_amp", type=int, default=config.get("use_amp", 0))
    add("--inverse", type=int, default=config.get("inverse", 0))
    add("--seed", type=int, default=config.get("seed", 2024))
    add("--checkpoints", type=str, default=config.get("checkpoints", "./checkpoints/"))
    add("--results", type=str, default=config.get("results", "./results/"))
    return parser.parse_args()


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML to use --config YAML files.") from exc
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return payload or {}


if __name__ == "__main__":
    main()
