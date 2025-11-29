# Source: libs/fla/modules/l2norm.py (L2NormFunction, lines 240-312)
# 以純 PyTorch 計算 L2 正規化，避免依賴 Triton kernel。

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch fallback for `libs.fla.modules.l2norm.l2norm_fwd`.

    Args:
        x: [..., D] tensor
        eps: numerical stability constant
        output_dtype: override dtype for normalized output

    Returns:
        y: L2-normalized tensor（使用指定 dtype）
        rstd: reciprocal std（float32），shape = x.shape[:-1]
    """

    x_float = x.to(torch.float32)
    norm_sq = x_float.pow(2).sum(dim=-1, keepdim=True)
    rstd = torch.rsqrt(norm_sq + eps)
    y = x_float * rstd
    target_dtype = x.dtype if output_dtype is None else output_dtype
    y = y.to(target_dtype)
    return y, rstd.squeeze(-1)


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pure PyTorch fallback for `libs.fla.modules.l2norm.l2norm_bwd`.

    Args:
        y: normalized tensor（[..., D]）
        rstd: reciprocal std（[...,]）
        dy: gradient of y
        eps: numerical stability constant（保留介面相容性，實際由 forward 控制）
    """

    y_float = y.to(torch.float32)
    dy_float = dy.to(torch.float32)
    if rstd.dim() == y_float.dim():
        rstd_view = rstd
    else:
        rstd_view = rstd.unsqueeze(-1)
    inner = torch.sum(dy_float * y_float, dim=-1, keepdim=True)
    dx = (dy_float - inner * y_float) * rstd_view
    return dx.to(y.dtype)


def l2_norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """前向便捷介面，與官方 `l2_norm` 行為一致。"""

    y, _ = l2norm_fwd(x, eps, output_dtype)
    return y


class L2Norm(nn.Module):
    """模擬 libs.fla.modules.l2norm.L2Norm 的 nn.Module 介面。"""

    def __init__(
        self,
        eps: float = 1e-6,
        output_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_norm(x, self.eps, self.output_dtype)
