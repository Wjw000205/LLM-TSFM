"""TimesNet placeholder."""

from __future__ import annotations

import torch.nn as nn


class TimesNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        raise NotImplementedError("TimesNet is a placeholder. Use --model DLinear or implement this class.")

