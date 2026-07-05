"""Training, validation, and testing loop for long-term forecasting."""

from __future__ import annotations

import json
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
        self.training_module = nn.ModuleDict({"model": self.model})
        if self.rule_adapter is not None:
            self.training_module["rule_adapter"] = self.rule_adapter
        self.criterion = build_loss(args)
        print_run_config(self.args)

    def train(self, setting: str):
        """Train with validation and early stopping."""
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty. Check data length and window sizes.")

        checkpoint_dir = Path(self.args.checkpoints) / setting
        save_run_config(self.args, checkpoint_dir)
        optimizer = optim.Adam(self.training_module.parameters(), lr=float(self.args.learning_rate))
        early_stopping = EarlyStopping(patience=int(self.args.patience))
        amp_enabled = bool_flag(getattr(self.args, "use_amp", False)) and self.device.type == "cuda"
        amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        for epoch in range(int(self.args.train_epochs)):
            self.training_module.train()
            component_sums = _empty_components()
            steps = 0
            for batch in self.train_loader:
                optimizer.zero_grad(set_to_none=True)
                with self._autocast(amp_enabled):
                    pred, true, masks = self._process_batch(batch)
                    loss_dict = self.criterion(pred, true, batch_masks=masks)
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
            learning_rate = optimizer.param_groups[0]["lr"]
            self._print_epoch(epoch + 1, train_components, val_components, learning_rate, self.args.early_stop_metric)
            stop_value = select_early_stop_value(val_components, self.args.early_stop_metric)
            early_stopping(stop_value, self.training_module, checkpoint_dir)
            if early_stopping.early_stop:
                break

        best_model_path = checkpoint_dir / "checkpoint.pth"
        if best_model_path.exists():
            self.training_module.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def validate(self) -> dict[str, float]:
        """Evaluate validation loss components."""
        self.training_module.eval()
        component_sums = _empty_components()
        steps = 0
        with torch.no_grad():
            for batch in self.val_loader:
                pred, true, masks = self._process_batch(batch)
                loss_dict = self.criterion(pred, true, batch_masks=masks)
                _accumulate(component_sums, loss_dict)
                steps += 1
        return _average_components(component_sums, steps)

    def test(self, setting: str, load_best: bool = True):
        """Run test split and save predictions, labels, and metrics."""
        if load_best:
            checkpoint = Path(self.args.checkpoints) / setting / "checkpoint.pth"
            if checkpoint.exists():
                self.training_module.load_state_dict(torch.load(checkpoint, map_location=self.device))

        self.training_module.eval()
        preds, trues, masks_all = [], [], []
        with torch.no_grad():
            for batch in self.test_loader:
                pred, true, masks = self._process_batch(batch)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                masks_all.append(masks.detach().cpu().numpy())

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
            pred_selected = preds_original
            true_selected = trues_original
            metric_space = "original"
        else:
            metrics = metrics_normalized
            pred_selected = preds
            true_selected = trues
            metric_space = "normalized"
        result_dir = ensure_dir(Path(self.args.results) / setting)
        save_run_config(self.args, result_dir)
        np.save(result_dir / "pred.npy", pred_selected)
        np.save(result_dir / "true.npy", true_selected)
        np.save(result_dir / "pred_normalized.npy", preds)
        np.save(result_dir / "true_normalized.npy", trues)
        np.save(result_dir / "pred_original.npy", preds_original)
        np.save(result_dir / "true_original.npy", trues_original)
        np.save(result_dir / "metrics.npy", metrics)
        (result_dir / "metrics_normalized.json").write_text(json.dumps(metrics_normalized, indent=2), encoding="utf-8")
        (result_dir / "metrics_original_scale.json").write_text(json.dumps(metrics_original, indent=2), encoding="utf-8")
        event_metrics = {k: v for k, v in metrics.items() if "event" in k or "rule" in k or "peak" in k}
        (result_dir / "event_metrics.json").write_text(json.dumps(event_metrics, indent=2), encoding="utf-8")
        print(f"test metric_space={metric_space} metrics: {metrics}")
        return metrics

    def _process_batch(self, batch):
        seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_llm, seq_y_llm, seq_y_masks = batch
        seq_x = seq_x.float().to(self.device)
        seq_y = seq_y.float().to(self.device)
        seq_x_llm = seq_x_llm.float().to(self.device)
        seq_y_llm = seq_y_llm.float().to(self.device)
        seq_y_masks = seq_y_masks.float().to(self.device)

        if seq_x_llm.shape[-1] > 0:
            seq_x = torch.cat([seq_x, seq_x_llm], dim=-1)

        pred = self.model(seq_x)
        true = seq_y[:, -int(self.args.pred_len) :, : int(self.args.c_out)]
        masks = seq_y_masks[:, -int(self.args.pred_len) :, :]
        future_llm = seq_y_llm[:, -int(self.args.pred_len) :, :]
        if self.rule_adapter is not None:
            pred = self.rule_adapter(pred, future_llm, masks)
        if bool_flag(getattr(self.args, "use_hard_intervention", False)):
            pred = apply_hard_intervention(pred, masks, getattr(self.args, "zero_target", [0.0] * int(self.args.c_out)))
        return pred, true, masks

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
    }


def _accumulate(target, loss_dict):
    for key in target:
        target[key] += float(loss_dict[key].detach().cpu())


def _average_components(component_sums, steps: int):
    if steps == 0:
        return _empty_components()
    return {key: value / steps for key, value in component_sums.items()}


def select_early_stop_value(losses: dict[str, float], metric: str) -> float:
    """Choose validation loss used by early stopping."""
    if metric == "base_mse":
        return float(losses["base_loss"])
    if metric == "total_loss":
        return float(losses["loss"])
    raise ValueError("early_stop_metric must be 'base_mse' or 'total_loss'.")


def save_run_config(args, output_dir: str | Path):
    """Save complete run configuration as text and JSON without unserializable objects."""
    output_dir = ensure_dir(output_dir)
    payload = serializable_args(args)
    (output_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [f"{key}: {value}" for key, value in payload.items()]
    (output_dir / "setting.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "llm_feature_dim",
        "enc_in",
        "target_dim",
        "c_out",
        "use_zscore",
        "use_revin",
        "use_llm_features",
        "use_llm_rule_features",
        "use_dataset_aware_loss",
        "inverse",
        "early_stop_metric",
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
