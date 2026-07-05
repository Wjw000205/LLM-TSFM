"""Training, validation, and testing loop for long-term forecasting."""

from __future__ import annotations

import copy
import csv
import json
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from losses.loss_factory import build_loss
from models.model_factory import build_model
from models.layers.RulePriorFusion import RulePriorFusion
from models.rule_adapter import RuleAdapter, apply_hard_intervention
from utils.early_stopping import EarlyStopping
from utils.metrics import metric
from utils.tools import bool_flag, ensure_dir


class ExpLongTermForecasting:
    """End-to-end experiment wrapper for one setting."""

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.train_data, self.train_loader = data_provider(args, "train")
        self.val_data, self.val_loader = data_provider(args, "val")
        self.test_data, self.test_loader = data_provider(args, "test")
        self.model = build_model(args).to(self.device)
        self.rule_adapter = self._build_rule_adapter()
        self.rule_prior_fusion = self._build_rule_prior_fusion()
        if bool_flag(getattr(args, "use_rule_prior_fusion", False)) and bool_flag(
            getattr(args, "use_hard_intervention", False)
        ):
            warnings.warn(
                "use_rule_prior_fusion and use_hard_intervention are both enabled; "
                "hard_intervention will overwrite the soft prior at zero-event timestamps.",
                UserWarning,
                stacklevel=2,
            )
        self.training_module = nn.ModuleDict({"model": self.model})
        if self.rule_adapter is not None:
            self.training_module["rule_adapter"] = self.rule_adapter
        self._load_pretrained_checkpoint()
        self.baseline_model = self._build_baseline_model()
        self.criterion = build_loss(args)
        print_run_config(self.args)

    def train(self, setting: str):
        """Train with validation and early stopping."""
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty. Check data length and window sizes.")

        checkpoint_dir = Path(self.args.checkpoints) / setting
        save_run_config(self.args, checkpoint_dir)
        save_loss_config(self.criterion, checkpoint_dir)
        optimizer = optim.Adam(self.training_module.parameters(), lr=float(self.args.learning_rate))
        early_stopping = EarlyStopping(patience=int(self.args.patience))
        baseline_mse = load_baseline_mse(getattr(self.args, "baseline_metric_path", None))
        selector = GuardedCheckpointSelector(
            selection_metric=getattr(self.args, "selection_metric", "base_mse"),
            baseline_mse=baseline_mse,
            overall_mse_tolerance=float(getattr(self.args, "overall_mse_tolerance", 0.05)),
        )
        early_stop_dir = checkpoint_dir / "_early_stop_monitor"
        amp_enabled = bool_flag(getattr(self.args, "use_amp", False)) and self.device.type == "cuda"
        amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        validation_history: list[dict[str, float | int]] = []

        for epoch in range(int(self.args.train_epochs)):
            self.training_module.train()
            component_sums = _empty_components()
            steps = 0
            for batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                with self._autocast(amp_enabled):
                    pred, true, masks, baseline_pred = self._process_batch(batch)
                    loss_dict = self.criterion(pred, true, batch_masks=masks, baseline_pred=baseline_pred)
                    loss_dict = self._add_intervention_components(loss_dict)
                    loss = loss_dict["loss"]

                if amp_enabled:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                _accumulate(component_sums, loss_dict)
                steps += 1

            train_components = _average_components(component_sums, steps)
            val_components = self.validate()
            validation_history.append(_validation_history_row(epoch + 1, val_components))
            save_validation_history(validation_history, checkpoint_dir)
            learning_rate = optimizer.param_groups[0]["lr"]
            self._print_epoch(epoch + 1, train_components, val_components, learning_rate, self.args.early_stop_metric)
            selector.update(epoch + 1, val_components, self.training_module, checkpoint_dir)
            stop_value = select_early_stop_value(val_components, self.args.early_stop_metric)
            early_stopping(stop_value, self.training_module, early_stop_dir)
            if early_stopping.early_stop:
                break

        selected = selector.finalize()
        for key, value in selected.items():
            setattr(self.args, key, value)
        save_run_config(self.args, checkpoint_dir)
        save_loss_config(self.criterion, checkpoint_dir)
        best_model_path = checkpoint_dir / "checkpoint.pth"
        if best_model_path.exists():
            self.training_module.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def validate(self) -> dict[str, float]:
        """Evaluate validation loss components."""
        self.training_module.eval()
        component_sums = _empty_components()
        preds, trues, masks_all = [], [], []
        steps = 0
        with torch.no_grad():
            for batch in self.val_loader:
                pred, true, masks, baseline_pred = self._process_batch(batch)
                loss_dict = self.criterion(pred, true, batch_masks=masks, baseline_pred=baseline_pred)
                loss_dict = self._add_intervention_components(loss_dict)
                _accumulate(component_sums, loss_dict)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                masks_all.append(masks.detach().cpu().numpy())
                steps += 1
        components = _average_components(component_sums, steps)
        if preds:
            val_metrics = metric(
                np.concatenate(preds, axis=0),
                np.concatenate(trues, axis=0),
                masks=np.concatenate(masks_all, axis=0),
            )
            components["base_loss"] = float(val_metrics.get("mse", components["base_loss"]))
            components["event_mse"] = float(val_metrics.get("event_window_mse", 0.0))
            components["zero_mse"] = float(val_metrics.get("zero_event_mse", 0.0))
            components["rule_score"] = float(val_metrics.get("rule_consistency_score", 0.0))
            components["val_base_mse"] = components["base_loss"]
            components["val_event_mse"] = components["event_mse"]
            components["val_zero_mse"] = components["zero_mse"]
            components["val_rule_score"] = components["rule_score"]
            components["val_total_loss"] = components["loss"]
        return components

    def test(self, setting: str, load_best: bool = True):
        """Run test split and save predictions, labels, and metrics."""
        if load_best:
            checkpoint = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            if checkpoint.exists():
                self.training_module.load_state_dict(torch.load(checkpoint, map_location=self.device))

        self.training_module.eval()
        preds, trues, masks_all, intervention_stats = [], [], [], []
        with torch.no_grad():
            for batch in self.test_loader:
                pred, true, masks, _ = self._process_batch(batch)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                masks_all.append(masks.detach().cpu().numpy())
                intervention_stats.append(self._intervention_stats())

        if not preds:
            raise ValueError("Test loader is empty. Check data length and window sizes.")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks_all = np.concatenate(masks_all, axis=0)

        metrics_normalized = metric(preds, trues, masks=masks_all)
        if bool_flag(getattr(self.args, "use_zscore", False)):
            preds_original = self.test_data.inverse_transform_target(preds)
            trues_original = self.test_data.inverse_transform_target(trues)
        else:
            preds_original = preds
            trues_original = trues

        metrics_original = metric(preds_original, trues_original, masks=masks_all)
        if bool_flag(getattr(self.args, "inverse", False)):
            metrics = metrics_original
            metric_space = "original"
        else:
            metrics = metrics_normalized
            metric_space = "normalized"
        result_dir = ensure_dir(Path(self.args.results) / setting)
        save_run_config(self.args, result_dir)
        save_loss_config(self.criterion, result_dir)
        checkpoint_history = Path(self.args.checkpoints) / setting / "validation_history.csv"
        checkpoint_history_json = Path(self.args.checkpoints) / setting / "validation_history.json"
        if checkpoint_history.exists():
            shutil.copy2(checkpoint_history, result_dir / "validation_history.csv")
        if checkpoint_history_json.exists():
            shutil.copy2(checkpoint_history_json, result_dir / "validation_history.json")
        for filename, array in build_prediction_output_payload(preds, trues, preds_original, trues_original).items():
            np.save(result_dir / filename, array)
        np.save(result_dir / "metrics.npy", metrics)
        (result_dir / "metrics_normalized.json").write_text(json.dumps(metrics_normalized, indent=2), encoding="utf-8")
        (result_dir / "metrics_original_scale.json").write_text(json.dumps(metrics_original, indent=2), encoding="utf-8")
        event_metrics = {
            "normalized": filter_event_metrics(metrics_normalized),
            "original_scale": filter_event_metrics(metrics_original),
        }
        (result_dir / "event_metrics.json").write_text(json.dumps(event_metrics, indent=2), encoding="utf-8")
        (result_dir / "intervention_stats.json").write_text(
            json.dumps(_average_stat_rows(intervention_stats), indent=2),
            encoding="utf-8",
        )
        print(f"test metric_space={metric_space} metrics: {metrics}")
        return metrics

    def _process_batch(self, batch):
        seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_llm, seq_y_llm, seq_y_masks = batch
        seq_x = seq_x.float().to(self.device)
        seq_y = seq_y.float().to(self.device)
        seq_x_llm = seq_x_llm.float().to(self.device)
        seq_y_llm = seq_y_llm.float().to(self.device)
        seq_y_masks = seq_y_masks.float().to(self.device)
        true = seq_y[:, -int(self.args.pred_len) :, : int(self.args.c_out)]
        masks = seq_y_masks[:, -int(self.args.pred_len) :, :]
        future_llm = seq_y_llm[:, -int(self.args.pred_len) :, :]

        baseline_pred = None
        if self.baseline_model is not None:
            baseline_enc_in = int(getattr(self.baseline_model, "enc_in", seq_x.shape[-1]))
            baseline_input = seq_x[:, :, :baseline_enc_in]
            with torch.no_grad():
                baseline_pred = self.baseline_model(baseline_input)

        if seq_x_llm.shape[-1] > 0 and not bool_flag(getattr(self.args, "use_intervention_layer", False)):
            seq_x = torch.cat([seq_x, seq_x_llm], dim=-1)

        if bool_flag(getattr(self.args, "use_intervention_layer", False)):
            pred = self.model(seq_x, future_features=future_llm, future_masks=masks)
        else:
            pred = self.model(seq_x)
        if self.rule_prior_fusion is not None:
            pred = self.rule_prior_fusion(
                pred_base=pred,
                future_masks=masks,
                future_features=future_llm,
                zero_target=getattr(self.train_data, "zero_target", getattr(self.args, "zero_target", [0.0])),
            )
        if self.rule_adapter is not None:
            pred = self.rule_adapter(pred, future_llm, masks)
        if bool_flag(getattr(self.args, "use_hard_intervention", False)):
            pred = apply_hard_intervention(pred, masks, getattr(self.args, "zero_target", [0.0] * int(self.args.c_out)))
        return pred, true, masks, baseline_pred

    def _add_intervention_components(self, loss_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        reg_loss = self._intervention_reg_loss()
        if bool_flag(getattr(self.args, "use_intervention_reg", False)):
            loss_dict["loss"] = loss_dict["loss"] + float(getattr(self.args, "intervention_reg_weight", 0.0)) * reg_loss
        loss_dict["intervention_reg_loss"] = reg_loss
        for key, value in self._intervention_stats().items():
            loss_dict[key] = reg_loss.new_tensor(value)
        return loss_dict

    def _intervention_reg_loss(self) -> torch.Tensor:
        if not hasattr(self.model, "get_intervention_reg_loss"):
            return next(self.training_module.parameters()).new_tensor(0.0)
        return self.model.get_intervention_reg_loss()

    def _intervention_stats(self) -> dict[str, float]:
        if not hasattr(self.model, "get_intervention_stats"):
            return _zero_intervention_stats()
        return self.model.get_intervention_stats()

    def _acquire_device(self):
        use_gpu = bool_flag(getattr(self.args, "use_gpu", True))
        if use_gpu and torch.cuda.is_available():
            return torch.device(getattr(self.args, "device", "cuda:0"))
        return torch.device("cpu")

    def _autocast(self, enabled: bool):
        if enabled:
            return torch.autocast(device_type="cuda")
        return nullcontext()

    def _build_rule_adapter(self):
        if not bool_flag(getattr(self.args, "use_rule_adapter", False)):
            return None
        feature_dim = int(getattr(self.args, "llm_feature_dim", 0))
        if feature_dim <= 0:
            return None
        hidden_dim = int(getattr(self.args, "rule_adapter_hidden", 32))
        return RuleAdapter(feature_dim=feature_dim, c_out=int(self.args.c_out), hidden_dim=hidden_dim).to(self.device)

    def _build_rule_prior_fusion(self):
        if not bool_flag(getattr(self.args, "use_rule_prior_fusion", False)):
            return None
        return RulePriorFusion(
            alpha=float(getattr(self.args, "rule_prior_alpha", 0.5)),
            use_confidence=bool_flag(getattr(self.args, "rule_prior_use_confidence", False)),
            rule_prior_types=getattr(self.args, "rule_prior_types", "zero_event"),
        ).to(self.device)

    def _load_pretrained_checkpoint(self):
        checkpoint = getattr(self.args, "load_pretrained_checkpoint", None)
        if not checkpoint:
            return
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
        state = _unwrap_state_dict(torch.load(path, map_location=self.device))
        if any(str(key).startswith(("model.", "rule_adapter.")) for key in state):
            self.training_module.load_state_dict(state, strict=False)
        else:
            self.model.load_state_dict(state, strict=False)
        print(f"loaded_pretrained_checkpoint: {path}")

    def _build_baseline_model(self):
        if not bool_flag(getattr(self.args, "use_baseline_distillation", False)):
            return None
        checkpoint = getattr(self.args, "baseline_checkpoint", None)
        if not checkpoint:
            warnings.warn(
                "use_baseline_distillation=1 but baseline_checkpoint is empty; distillation loss will be disabled.",
                UserWarning,
                stacklevel=2,
            )
            return None
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found: {path}")

        baseline_args = copy.deepcopy(self.args)
        raw_dim = int(getattr(self.args, "raw_input_dim", getattr(self.args, "raw_feature_dim", self.args.enc_in)))
        baseline_args.enc_in = raw_dim
        baseline_args.llm_feature_dim = 0
        baseline_args.standard_time_feature_dim = 0
        baseline_args.llm_rule_feature_dim = 0
        baseline_args.oracle_feature_dim = 0
        baseline_args.use_llm_features = 0
        baseline_args.use_llm_rule_features = 0
        baseline_args.use_standard_time_features = 0
        baseline_args.use_oracle_features = 0
        baseline_args.use_rule_adapter = 0
        baseline_args.use_intervention_layer = 0
        baseline_args.use_intervention_reg = 0
        baseline_args.use_hard_intervention = 0

        baseline_model = build_model(baseline_args).to(self.device)
        baseline_model.load_state_dict(_extract_model_state(torch.load(path, map_location=self.device)), strict=False)
        baseline_model.eval()
        for param in baseline_model.parameters():
            param.requires_grad_(False)
        print(f"loaded_baseline_distillation_checkpoint: {path}")
        return baseline_model

    @staticmethod
    def _print_epoch(epoch, train_components, val_components, learning_rate, early_stop_metric):
        fields = {
            "epoch": epoch,
            "train_total_loss": train_components["loss"],
            "val_total_loss": val_components["loss"],
            "train_base_mse_loss": train_components["base_loss"],
            "val_base_mse_loss": val_components["base_loss"],
            "event_loss": train_components["event_loss"],
            "zero_loss": train_components["zero_loss"],
            "peak_loss": train_components["peak_loss"],
            "diff_loss": train_components["diff_loss"],
            "freq_loss": train_components["freq_loss"],
            "nonevent_loss": train_components["nonevent_loss"],
            "distill_loss": train_components["distill_loss"],
            "intervention_reg_loss": train_components["intervention_reg_loss"],
            "mean_event_gate": train_components["mean_event_gate"],
            "mean_non_event_gate": train_components["mean_non_event_gate"],
            "mean_event_delta_norm": train_components["mean_event_delta_norm"],
            "mean_non_event_delta_norm": train_components["mean_non_event_delta_norm"],
            "val_event_mse": val_components.get("event_mse", 0.0),
            "val_zero_mse": val_components.get("zero_mse", 0.0),
            "val_rule_score": val_components.get("rule_score", 0.0),
            "total_loss": train_components["loss"],
            "early_stop_value": select_early_stop_value(val_components, early_stop_metric),
            "learning_rate": learning_rate,
        }
        print(" ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in fields.items()))


def _empty_components():
    return {
        "loss": 0.0,
        "base_loss": 0.0,
        "event_loss": 0.0,
        "zero_loss": 0.0,
        "peak_loss": 0.0,
        "diff_loss": 0.0,
        "freq_loss": 0.0,
        "nonevent_loss": 0.0,
        "distill_loss": 0.0,
        "intervention_reg_loss": 0.0,
        "mean_event_gate": 0.0,
        "mean_non_event_gate": 0.0,
        "mean_event_delta_norm": 0.0,
        "mean_non_event_delta_norm": 0.0,
        "event_mse": 0.0,
        "zero_mse": 0.0,
        "rule_score": 0.0,
        "val_base_mse": 0.0,
        "val_event_mse": 0.0,
        "val_zero_mse": 0.0,
        "val_rule_score": 0.0,
        "val_total_loss": 0.0,
    }


def _accumulate(target, loss_dict):
    for key in target:
        value = loss_dict.get(key)
        if value is not None:
            target[key] += float(value.detach().cpu())


def _average_components(component_sums, steps: int):
    if steps == 0:
        return _empty_components()
    return {key: value / steps for key, value in component_sums.items()}


def _zero_intervention_stats() -> dict[str, float]:
    return {
        "mean_event_gate": 0.0,
        "mean_non_event_gate": 0.0,
        "mean_event_delta_norm": 0.0,
        "mean_non_event_delta_norm": 0.0,
    }


def _average_stat_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return _zero_intervention_stats()
    keys = _zero_intervention_stats().keys()
    return {key: float(np.mean([row.get(key, 0.0) for row in rows])) for key in keys}


def select_early_stop_value(losses: dict[str, float], metric: str) -> float:
    """Choose validation loss used by early stopping."""
    if metric == "base_mse":
        return float(losses["base_loss"])
    if metric == "total_loss":
        return float(losses["loss"])
    raise ValueError("early_stop_metric must be 'base_mse' or 'total_loss'.")


class GuardedCheckpointSelector:
    """Select the persisted checkpoint independently of patience-based early stopping."""

    def __init__(self, selection_metric: str = "base_mse", baseline_mse: float | None = None, overall_mse_tolerance: float = 0.05):
        self.selection_metric = selection_metric
        self.baseline_mse = baseline_mse
        self.overall_mse_tolerance = overall_mse_tolerance
        self.best_score: tuple[float, ...] | None = None
        self.selected_epoch: int | None = None
        self.selected_reason: str | None = None
        self.fallback_score: float | None = None
        self.fallback_epoch: int | None = None
        self.checkpoint_dir: Path | None = None
        self.selected_metrics: dict[str, float] = {}
        self.fallback_metrics: dict[str, float] = {}
        if self.selection_metric in {"guarded_event_mse", "pareto"} and self.baseline_mse is None:
            warnings.warn(
                f"selection_metric={self.selection_metric} requested without baseline_mse; falling back to base_mse.",
                UserWarning,
                stacklevel=2,
            )

    def update(self, epoch: int, metrics: dict[str, float], model: torch.nn.Module, checkpoint_dir: str | Path):
        checkpoint_dir = ensure_dir(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        base_mse = _metric_value(metrics, "base_loss", "val_base_mse")
        event_mse = _metric_value(metrics, "event_mse", "val_event_mse")
        total_loss = _metric_value(metrics, "loss", "val_total_loss")

        if self.fallback_score is None or base_mse < self.fallback_score:
            self.fallback_score = base_mse
            self.fallback_epoch = epoch
            self.fallback_metrics = _selection_metrics(metrics)
            torch.save(model.state_dict(), checkpoint_dir / "fallback_checkpoint.pth")

        result = self._score(base_mse=base_mse, event_mse=event_mse, total_loss=total_loss)
        if result is None:
            return False
        score, reason = result
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.selected_epoch = epoch
            self.selected_reason = reason
            self.selected_metrics = _selection_metrics(metrics)
            torch.save(model.state_dict(), checkpoint_dir / "checkpoint.pth")
            return True
        return False

    def finalize(self) -> dict[str, Any]:
        if self.selected_epoch is not None:
            return {
                "selected_epoch": self.selected_epoch,
                "selected_reason": self.selected_reason,
                "selected_metrics": self.selected_metrics,
            }

        if self.checkpoint_dir is not None and self.fallback_epoch is not None:
            fallback_path = self.checkpoint_dir / "fallback_checkpoint.pth"
            checkpoint_path = self.checkpoint_dir / "checkpoint.pth"
            if fallback_path.exists():
                shutil.copy2(fallback_path, checkpoint_path)
            warnings.warn(
                "No validation checkpoint satisfied the guarded_event_mse overall-MSE guardrail; "
                "falling back to the lowest validation base MSE checkpoint.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "selected_epoch": self.fallback_epoch,
                "selected_reason": "fallback_base_mse_no_guardrail_candidate",
                "selected_metrics": self.fallback_metrics,
            }

        return {
            "selected_epoch": None,
            "selected_reason": "no_validation_checkpoint",
            "selected_metrics": {},
        }

    def _score(self, base_mse: float, event_mse: float, total_loss: float) -> tuple[float, ...] | None:
        if self.selection_metric == "base_mse":
            return (base_mse,), "base_mse"
        if self.selection_metric == "total_loss":
            return (total_loss,), "total_loss"
        if self.selection_metric == "event_mse":
            return (event_mse,), "event_mse"
        if self.selection_metric in {"guarded_event_mse", "pareto"}:
            if self.baseline_mse is None:
                return (base_mse,), "fallback_base_mse_missing_baseline"
            threshold = self.baseline_mse * (1.0 + self.overall_mse_tolerance)
            if base_mse <= threshold:
                return (event_mse, base_mse), self.selection_metric
            return None
        raise ValueError(f"Unknown selection_metric: {self.selection_metric}")


def load_baseline_mse(path: str | None) -> float | None:
    """Load the baseline MSE used by guarded checkpoint selection."""
    if not path:
        return None
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("mse", "overall_mse", "Overall MSE"):
        if key in payload:
            return float(payload[key])
    raise KeyError(f"Baseline metrics file must contain one of mse/overall_mse/Overall MSE: {metrics_path}")


def save_validation_history(rows: list[dict[str, float | int]], output_dir: str | Path):
    """Persist validation metrics observed during training."""
    output_dir = ensure_dir(output_dir)
    json_path = output_dir / "validation_history.json"
    csv_path = output_dir / "validation_history.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = [
        "epoch",
        "val_base_mse",
        "val_event_mse",
        "val_zero_mse",
        "val_rule_score",
        "val_total_loss",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validation_history_row(epoch: int, metrics: dict[str, float]) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "val_base_mse": float(metrics.get("val_base_mse", metrics.get("base_loss", 0.0))),
        "val_event_mse": float(metrics.get("val_event_mse", metrics.get("event_mse", 0.0))),
        "val_zero_mse": float(metrics.get("val_zero_mse", metrics.get("zero_mse", 0.0))),
        "val_rule_score": float(metrics.get("val_rule_score", metrics.get("rule_score", 0.0))),
        "val_total_loss": float(metrics.get("val_total_loss", metrics.get("loss", 0.0))),
    }


def _metric_value(metrics: dict[str, float], *keys: str) -> float:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"Missing metric, expected one of: {keys}")


def _selection_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "base_loss",
        "loss",
        "event_mse",
        "zero_mse",
        "rule_score",
        "val_base_mse",
        "val_event_mse",
        "val_zero_mse",
        "val_rule_score",
        "val_total_loss",
    ]
    return {key: float(metrics[key]) for key in keys if key in metrics}


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _extract_model_state(state):
    state = _unwrap_state_dict(state)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a state dict or contain a state_dict.")
    if any(str(key).startswith("model.") for key in state):
        return {str(key)[len("model.") :]: value for key, value in state.items() if str(key).startswith("model.")}
    return state


def save_run_config(args, output_dir: str | Path):
    """Save complete run configuration as text and JSON without unserializable objects."""
    output_dir = ensure_dir(output_dir)
    payload = serializable_args(args)
    (output_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"{key}: {value}" for key, value in payload.items()]
    (output_dir / "setting.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_loss_config(criterion, output_dir: str | Path):
    """Save the resolved dataset-aware loss configuration for auditability."""
    config = getattr(criterion, "config", None)
    if not config:
        return None
    output_dir = ensure_dir(output_dir)
    payload = {key: _jsonable_value(value) for key, value in sorted(config.items())}
    path = output_dir / "loss_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_run_config(args):
    """Print the dimensions and run arguments that affect reproducibility."""
    payload = serializable_args(args)
    print("run_config:")
    for key in [
        "data",
        "data_path",
        "root_path",
        "model",
        "seq_len",
        "label_len",
        "pred_len",
        "batch_size",
        "learning_rate",
        "train_epochs",
        "patience",
        "features",
        "target",
        "raw_feature_dim",
        "raw_input_dim",
        "llm_feature_dim",
        "standard_time_feature_dim",
        "llm_rule_feature_dim",
        "oracle_feature_dim",
        "enc_in",
        "target_dim",
        "c_out",
        "use_zscore",
        "use_revin",
        "use_llm_features",
        "use_llm_rule_features",
        "use_standard_time_features",
        "use_oracle_features",
        "use_dataset_aware_loss",
        "use_event_weighted_loss",
        "use_zero_consistency_loss",
        "use_peak_shape_loss",
        "use_diff_loss",
        "use_freq_loss",
        "use_nonevent_preservation_loss",
        "use_baseline_distillation",
        "event_weight",
        "zero_weight",
        "peak_weight",
        "diff_weight",
        "freq_weight",
        "nonevent_weight",
        "distill_weight",
        "peak_window_size",
        "selection_metric",
        "overall_mse_tolerance",
        "baseline_metric_path",
        "baseline_checkpoint",
        "load_pretrained_checkpoint",
        "finetune_learning_rate",
        "finetune_epochs",
        "finetune_patience",
        "use_rule_adapter",
        "use_intervention_layer",
        "intervention_hidden",
        "intervention_dropout",
        "intervention_scale",
        "intervention_init_zero",
        "use_intervention_reg",
        "intervention_reg_weight",
        "use_rule_prior_fusion",
        "rule_prior_alpha",
        "rule_prior_use_confidence",
        "rule_prior_types",
        "use_hard_intervention",
        "dlinear_init_avg",
        "inverse",
        "early_stop_metric",
        "selected_epoch",
        "selected_reason",
    ]:
        if key in payload:
            print(f"  {key}: {payload[key]}")


def serializable_args(args) -> dict[str, Any]:
    """Convert argparse Namespace to stable JSON-serializable metadata."""
    skip = {"scaler"}
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        elif isinstance(value, (list, tuple)):
            payload[key] = list(value)
        elif isinstance(value, dict):
            payload[key] = value
        else:
            payload[key] = str(value)
    return dict(sorted(payload.items()))


def _jsonable_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    return str(value)


def build_prediction_output_payload(pred_normalized, true_normalized, pred_original, true_original):
    """Return arrays saved by test(), with pred.npy/true.npy fixed to original scale."""
    return {
        "pred.npy": pred_original,
        "true.npy": true_original,
        "pred_normalized.npy": pred_normalized,
        "true_normalized.npy": true_normalized,
        "pred_original.npy": pred_original,
        "true_original.npy": true_original,
    }


def filter_event_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Keep only event-window metric entries."""
    return {key: value for key, value in metrics.items() if "event" in key or "rule" in key or "peak" in key}
