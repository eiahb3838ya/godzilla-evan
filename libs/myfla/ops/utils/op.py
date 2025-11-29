"""Elementwise op helpers（移植自 `libs/fla/ops/utils/op.py`)."""

from __future__ import annotations

import math

import torch


def exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def exp2(x: torch.Tensor) -> torch.Tensor:
    base = torch.tensor(2.0, dtype=x.dtype, device=x.device)
    return base.pow(x)


def log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


def log2(x: torch.Tensor) -> torch.Tensor:
    return torch.log2(x)


def safe_exp(x: torch.Tensor) -> torch.Tensor:
    """Matches `exp(tl.where(x <= 0, x, -inf))` 的行為。"""

    neg_inf = torch.full_like(x, -math.inf)
    return torch.exp(torch.where(x <= 0, x, neg_inf))


class TensorDescriptor:
    """Minimal PyTorch fallback for Triton TMA descriptor API."""

    def __init__(self, tensor: torch.Tensor, shape, strides, block_shape):
        self.tensor = tensor
        self.block_shape = block_shape

    def load(self, offset) -> torch.Tensor:
        row, col = offset
        h, w = self.block_shape
        return self.tensor[..., row:row + h, col:col + w]

    def store(self, offset, value: torch.Tensor) -> None:
        row, col = offset
        h, w = self.block_shape
        self.tensor[..., row:row + h, col:col + w] = value


def make_tensor_descriptor(
    tensor: torch.Tensor,
    shape,
    strides,
    block_shape,
) -> TensorDescriptor:
    return TensorDescriptor(tensor.view(*shape), shape, strides, block_shape)


__all__ = ['exp', 'exp2', 'log', 'log2', 'safe_exp', 'make_tensor_descriptor']
