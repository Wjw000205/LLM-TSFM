"""PatchTST placeholder.

The framework keeps this class as a clear extension point. DLinear is the
fully runnable backbone in this scaffold.
"""

from __future__ import annotations

import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(self, args):
        super().__init__()
        raise NotImplementedError("PatchTST is a placeholder. Use --model DLinear or implement this class.")

