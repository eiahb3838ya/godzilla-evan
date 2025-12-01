"""Compatibility layer matching libs.fla.modules.layernorm."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    upcast: bool = False,
):
    dtype = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        if residual is not None:
            residual = residual.float()
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(dtype)
    return out if not prenorm else (out, x)


def rms_norm_ref(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    upcast: bool = False,
    residual_in_fp32: bool = False,
):
    dtype = x.dtype
    working = x.float() if (upcast or residual_in_fp32) else x
    weight_tensor = weight
    bias_tensor = bias
    if upcast:
        weight_tensor = weight_tensor.float() if weight_tensor is not None else None
        bias_tensor = bias_tensor.float() if bias_tensor is not None else None
    if residual is not None:
        residual_to_add = residual.float() if (upcast or residual_in_fp32) else residual
        working = working + residual_to_add
    residual_out = working if prenorm else None
    rstd = 1 / torch.sqrt((working.square()).mean(dim=-1, keepdim=True) + eps)
    if weight_tensor is None:
        weight_tensor = torch.ones(
            working.shape[-1],
            dtype=working.dtype,
            device=working.device,
        )
    out = working * rstd * weight_tensor
    if bias_tensor is not None:
        out = out + bias_tensor
    out = out.to(dtype)
    if prenorm:
        res = residual_out if residual_in_fp32 or upcast else residual_out.to(dtype)
        return out, res
    return out


def group_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    num_groups: int,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    is_rms_norm: bool = False,
    prenorm: bool = False,
    upcast: bool = False,
):
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        if residual is not None:
            residual = residual.float()
    dtype = x.dtype
    if x.shape[-1] % num_groups != 0:
        raise ValueError("hidden_size must be divisible by num_groups")
    if residual is not None:
        x = (x + residual).to(x.dtype)
    residual_out = x
    weight = weight if weight is not None else torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    grouped = rearrange(x, "... (g d) -> ... g d", g=num_groups)
    weight_grouped = rearrange(weight, "(g d) -> g d", g=num_groups)
    bias_grouped = rearrange(bias, "(g d) -> g d", g=num_groups) if bias is not None else None
    if not is_rms_norm:
        grouped = grouped - grouped.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    out = grouped * rstd * weight_grouped
    if bias_grouped is not None:
        out = out + bias_grouped
    out = rearrange(out, "... g d -> ... (g d)").to(dtype)
    return out if not prenorm else (out, residual_out)


class GroupNormRef(nn.Module):
    """Pure PyTorch GroupNorm, mirroring fla.modules.layernorm.GroupNormRef."""

    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        is_rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.is_rms_norm = is_rms_norm

        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
    ) -> torch.Tensor:
        return group_norm_ref(
            x,
            self.weight,
            self.bias,
            num_groups=self.num_groups,
            residual=residual,
            eps=self.eps,
            is_rms_norm=self.is_rms_norm,
            prenorm=prenorm,
        )


class GroupNorm(GroupNormRef):
    """Alias to match fla.modules.layernorm.GroupNorm."""


class RMSNorm(nn.Module):
    """Pure PyTorch RMSNorm 實作（對齊 fla.modules.layernorm.RMSNorm 介面）。"""

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ):
        return rms_norm_ref(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


def rms_norm_gated_ref(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    activation: str = 'swish',
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS norm + gated activation.
    Replicates libs.fla.modules.fused_norm_gate.rms_norm_gated using pure PyTorch.

    Source: libs/fla/modules/fused_norm_gate.py:L838-L860

    Args:
        x: Input tensor [..., D]
        g: Gate tensor [..., D]
        weight: RMSNorm weight [D]
        bias: RMSNorm bias [D] or None
        activation: 'swish'/'silu' or 'sigmoid'
        residual: Optional residual to add before norm
        eps: RMSNorm epsilon
        prenorm: If True, return (output, residual_out)
        residual_in_fp32: Store residual in fp32

    Returns:
        output: Normalized and gated output
        or (output, residual_out) if prenorm=True
    """
    if activation not in ['swish', 'silu', 'sigmoid']:
        raise ValueError(f"Unsupported activation: {activation}")

    normed = rms_norm_ref(
        x,
        weight,
        bias,
        residual=residual,
        eps=eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
    )
    if prenorm:
        x_normalized, residual_out = normed
    else:
        x_normalized = normed
        residual_out = None

    x_float = x_normalized.float()
    g_float = g.float()
    if activation in ['swish', 'silu']:
        gated = x_float * g_float * torch.sigmoid(g_float)
    else:  # sigmoid
        gated = x_float * torch.sigmoid(g_float)
    output = gated.to(x_normalized.dtype)

    if prenorm:
        return output, residual_out
    return output


class FusedRMSNormGated(nn.Module):
    """
    PyTorch version of FusedRMSNormGated.

    Source: libs/fla/modules/fused_norm_gate.py:L985-L1046

    This module performs RMSNorm followed by gated activation in a fused manner.
    The official implementation uses Triton kernels for performance, but this
    version uses pure PyTorch for compatibility.
    """

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = 'swish',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ['swish', 'silu', 'sigmoid']:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += f", activation={self.activation}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        return rms_norm_gated_ref(
            x,
            g,
            self.weight,
            self.bias,
            self.activation,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


__all__ = [
    'layer_norm_ref',
    'rms_norm_ref',
    'rms_norm_gated_ref',
    'group_norm_ref',
    'GroupNormRef',
    'GroupNorm',
    'RMSNorm',
    'FusedRMSNormGated',
]
