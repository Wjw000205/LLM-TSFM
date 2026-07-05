"""Training, validation, and testing loop for long-term forecasting."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import optim

from data_provider.data_factory import data_provider
from losses.loss_factory import build_loss
from models.model_factory import build_model
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
        self.criterion = build_loss(args)

    def train(self, setting: str):
        """Train with validation and early stopping."""
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty. Check data length and window sizes.")

        checkpoint_dir = Path(self.args.checkpoints) / setting
        optimizer = optim.Adam(self.model.parameters(), lr=float(self.args.learning_rate))
        early_stopping = EarlyStopping(patience=int(self.args.patience))
        amp_enabled = bool_flag(getattr(self.args, "use_amp", False)) and self.device.type == "cuda"
        amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        for epoch in range(int(self.args.train_epochs)):
            self.model.train()
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
            self._print_epoch(epoch + 1, train_components, val_components, learning_rate)
            early_stopping(val_components["loss"], self.model, checkpoint_dir)
            if early_stopping.early_stop:
                break

        best_model_path = checkpoint_dir / "checkpoint.pth"
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def validate(self) -> dict[str, float]:
        """Evaluate validation loss components."""
        self.model.eval()
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
                self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

        self.model.eval()
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

        if bool_flag(getattr(self.args, "inverse", False)):
            preds = self.test_data.inverse_transform_target(preds)
            trues = self.test_data.inverse_transform_target(trues)

        metrics = metric(preds, trues, masks=masks_all)
        result_dir = ensure_dir(Path(self.args.results) / setting)
        np.save(result_dir / "pred.npy", preds)
        np.save(result_dir / "true.npy", trues)
        np.save(result_dir / "metrics.npy", metrics)
        event_metrics = {k: v for k, v in metrics.items() if "event" in k or "rule" in k or "peak" in k}
        (result_dir / "event_metrics.json").write_text(json.dumps(event_metrics, indent=2), encoding="utf-8")
        (result_dir / "setting.txt").write_text(str(self.args), encoding="utf-8")
        print(f"test metrics: {metrics}")
        return metrics

    def _process_batch(self, batch):
        seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_llm, seq_y_llm, seq_y_masks = batch
        seq_x = seq_x.float().to(self.device)
        seq_y = seq_y.float().to(self.device)
        seq_x_llm = seq_x_llm.float().to(self.device)
        seq_y_masks = seq_y_masks.float().to(self.device)

        if bool_flag(getattr(self.args, "use_llm_features", False)) and seq_x_llm.shape[-1] > 0:
            seq_x = torch.cat([seq_x, seq_x_llm], dim=-1)

        pred = self.model(seq_x)
        true = seq_y[:, -int(self.args.pred_len) :, : int(self.args.c_out)]
        masks = seq_y_masks[:, -int(self.args.pred_len) :, :]
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

    @staticmethod
    def _print_epoch(epoch, train_components, val_components, learning_rate):
        fields = {
            "epoch": epoch,
            "train_loss": train_components["loss"],
            "val_loss": val_components["loss"],
            "base_loss": train_components["base_loss"],
            "event_loss": train_components["event_loss"],
            "zero_loss": train_components["zero_loss"],
            "peak_loss": train_components["peak_loss"],
            "diff_loss": train_components["diff_loss"],
            "freq_loss": train_components["freq_loss"],
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
