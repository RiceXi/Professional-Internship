from __future__ import annotations
import torch
from torch import nn


class SiluAndMul(nn.Module):
    """SiLU 与门控乘法的融合激活（对应 fused gate/up MLP 结构）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.silu_and_mul(x)
