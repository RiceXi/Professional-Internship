from __future__ import annotations

import torch
from torch import nn

from .ops import fused_add_rmsnorm as _fused_add_rmsnorm
from .ops import rmsnorm as _rmsnorm


class RMSNorm(nn.Module):
    """RMS 归一化层，支持带残差融合的 fused 路径。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """无残差时直接 RMSNorm；有残差时走 fused add + RMSNorm（原地）。"""
        if residual is None:
            return _rmsnorm(x, self.weight, self.eps), x
        _fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
