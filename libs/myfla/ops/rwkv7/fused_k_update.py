from __future__ import annotations

import torch


def fused_k_rwkv7(k: torch.Tensor, a: torch.Tensor, k_a: torch.Tensor) -> torch.Tensor:
    """PyTorch fallback for libs.fla.ops.rwkv7.fused_k_update."""
    return k * (1.0 + (a - 1.0) * k_a)
