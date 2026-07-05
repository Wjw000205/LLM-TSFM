"""Model factory."""

from __future__ import annotations

from models.DLinear import DLinear
from models.PatchTST import PatchTST
from models.RNN import RecurrentForecastModel
from models.TimesNet import TimesNet
from models.iTransformer import iTransformer


def build_model(args):
    """Instantiate a forecasting backbone from ``args.model``."""
    name = getattr(args, "model", "DLinear")
    if name == "DLinear":
        return DLinear(args)
    if name in {"GRU", "LSTM"}:
        return RecurrentForecastModel(args)
    if name == "PatchTST":
        return PatchTST(args)
    if name == "iTransformer":
        return iTransformer(args)
    if name == "TimesNet":
        return TimesNet(args)
    raise ValueError(f"Unknown model: {name}")
