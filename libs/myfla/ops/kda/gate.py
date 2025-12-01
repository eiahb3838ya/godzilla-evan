# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# -*- coding: utf-8 -*-

"""
KDA Gate Operations - PyTorch Implementation
============================================

Source: libs/fla/ops/kda/gate.py

Perfect replication of official FLA KDA gate ops in pure PyTorch.
All Triton kernels are converted to PyTorch implementations.

Exported APIs:
- kda_gate_ref: Reference implementation (pure PyTorch)
- kda_gate_fwd: Forward pass
- kda_gate_bwd: Backward pass
- KDAGateFunction: torch.autograd.Function wrapper
- fused_kda_gate: User-facing API
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


def kda_gate_ref(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Torch reference implementation for KDA gate computation.

    Source: libs/fla/ops/kda/gate.py:L17-L55

    Computes: g = -A.exp().unsqueeze(-1) * softplus(rearrange(g, '... (h d) -> ... h d', d=head_k_dim))

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]

    Args:
        g: Input tensor of shape [..., num_heads * head_k_dim]
        A: Parameter tensor of shape [num_heads] or [1, 1, num_heads, 1]
        head_k_dim: Dimension of each head
        g_bias: Optional bias tensor added to g before activation, shape [num_heads * head_k_dim]
        b: Optional tensor to compute sigmoid gate, shape [..., num_heads]
        beta: softplus beta parameter (default: 1.0)
        threshold: softplus threshold parameter (default: 20.0)

    Returns:
        g_out: Output tensor of shape [..., num_heads, head_k_dim]
        b_sigmoid: Sigmoid of b if b is not None, else None
    """
    # Rearrange g to separate heads: [..., H*D] -> [..., H, D]
    A = A.view(-1)  # Flatten A to [num_heads] to handle any input shape
    if g_bias is not None:
        g = g + g_bias
    g = rearrange(g, '... (h d) -> ... h d', d=head_k_dim)

    # Apply the gate computation: -A.exp().unsqueeze(-1) * softplus(g)
    # A: [H] -> [H, 1] for broadcasting
    A_exp = -A.float().exp().unsqueeze(-1)  # [H, 1]
    g_softplus = F.softplus(g.float(), beta, threshold)  # [..., H, D]

    return A_exp * g_softplus, b.float().sigmoid() if b is not None else None


def kda_gate_fwd(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Forward pass for KDA gate (PyTorch implementation of Triton kernel).

    Source: libs/fla/ops/kda/gate.py:L284-L336 (kda_gate_fwd function)

    This is a pure PyTorch implementation that replicates the Triton kernel behavior.
    Official Triton kernel: libs/fla/ops/kda/gate.py:L58-L152

    Args:
        g: Input tensor [..., H*D]
        A: Parameter [H] or [1, 1, H, 1]
        head_k_dim: Dimension of each head
        g_bias: Optional bias [H*D]
        b: Optional tensor [..., H]
        beta: softplus beta parameter
        threshold: softplus threshold parameter

    Returns:
        y: Output tensor [..., H, D] (fp32)
        b_sigmoid: Sigmoid of b [..., H] (fp32) if b is not None, else None
    """
    # Use reference implementation (pure PyTorch equivalent)
    # The official uses Triton for performance, but the math is identical
    return kda_gate_ref(g, A, head_k_dim, g_bias, b, beta, threshold)


def kda_gate_bwd(
    grad_output: torch.Tensor,  # [..., H, D]
    g: torch.Tensor,            # [..., H*D]
    A: torch.Tensor,            # [H]
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    gb: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Backward pass for KDA gate (PyTorch implementation of Triton kernel).

    Source: libs/fla/ops/kda/gate.py:L339-L396 (kda_gate_bwd function)

    This is a pure PyTorch implementation that replicates the Triton kernel behavior.
    Official Triton kernel: libs/fla/ops/kda/gate.py:L154-L282

    Args:
        grad_output: Gradient of output [..., H, D]
        g: Input tensor [..., H*D]
        A: Parameter [H]
        head_k_dim: Dimension of each head
        g_bias: Optional bias [H*D]
        b: Optional tensor [..., H]
        gb: Optional gradient w.r.t. b output [..., H]
        beta: softplus beta parameter
        threshold: softplus threshold parameter

    Returns:
        dg: Gradient w.r.t. g [..., H*D]
        dA: Gradient w.r.t. A (same shape as A)
        dgbias: Gradient w.r.t. g_bias [H*D] if g_bias is not None, else None
        db: Gradient w.r.t. b [..., H] if b is not None, else None
    """
    orig_shape = g.shape
    g_flat = g.view(-1, g.shape[-1])
    T = g_flat.shape[0]

    A_ori_shape = A.shape
    H = A.numel()
    D = head_k_dim

    # Flatten grad_output: [..., H, D] -> [T, H*D]
    dy = grad_output.view(T, H * D)

    # Prepare output gradients
    dg = torch.zeros_like(g_flat, dtype=torch.float32)
    dA = torch.zeros((T, H), dtype=torch.float32, device=g.device)

    # Process b gradient if needed
    db = None
    if b is not None and gb is not None:
        b_flat = b.view(-1, H)
        gb_flat = gb.view(-1, H)
        # db = gb * sigmoid(b) * (1 - sigmoid(b))
        b_sig = torch.sigmoid(b_flat.float())
        db = gb_flat.float() * b_sig * (1.0 - b_sig)
        db = db.view(b.shape).type_as(b)

    # Compute gradients for each head
    A_flat = A.view(-1)  # [H]

    for h in range(H):
        # Extract this head's data: [T, D]
        g_h = g_flat[:, h * D:(h + 1) * D].float()
        dy_h = dy[:, h * D:(h + 1) * D].float()

        # Add bias if present
        if g_bias is not None:
            g_bias_h = g_bias[h * D:(h + 1) * D].float()
            g_h = g_h + g_bias_h[None, :]

        # Compute softplus: softplus(beta * g) = (1/beta) * log(1 + exp(beta * g))
        # When beta * g > threshold, use linear approximation g
        g_scaled = g_h * beta
        use_linear = g_scaled > threshold
        sp = torch.where(
            use_linear,
            g_h,
            (1.0 / beta) * torch.log(1.0 + torch.exp(g_scaled))
        )

        # Compute sigmoid for gradient: sigmoid(beta * g)
        sig = torch.sigmoid(g_scaled)

        # Gradient w.r.t. g: dg = dy * (-exp(A[h])) * sigmoid(beta * g)
        neg_exp_a = -torch.exp(A_flat[h].float())
        dg_h = dy_h * (neg_exp_a * sig)
        dg[:, h * D:(h + 1) * D] = dg_h

        # Gradient w.r.t. A: dA = sum(dy * (-exp(A[h]) * softplus(g)))
        contrib = dy_h * (neg_exp_a * sp)
        dA[:, h] = contrib.sum(dim=1)

    # Sum over T dimension for dA
    dA = dA.sum(0).view(A_ori_shape).type_as(A)

    # Sum over T dimension for g_bias gradient
    dgbias = dg.sum(0).type_as(g_bias) if g_bias is not None else None

    # Reshape dg back to original shape
    dg = dg.view(orig_shape).type_as(g)

    return dg, dA, dgbias, db


class KDAGateFunction(torch.autograd.Function):
    """
    Autograd function for KDA gate computation.

    Source: libs/fla/ops/kda/gate.py:L399-L436

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]
    """

    @staticmethod
    def forward(
        ctx,
        g: torch.Tensor,
        A: torch.Tensor,
        head_k_dim: int,
        g_bias: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ctx.save_for_backward(g, A)
        ctx.g_bias = g_bias
        ctx.b = b
        ctx.head_k_dim = head_k_dim
        ctx.beta = beta
        ctx.threshold = threshold

        return kda_gate_fwd(g, A, head_k_dim, g_bias, b, beta, threshold)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        gb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, ...]:
        g, A = ctx.saved_tensors
        head_k_dim = ctx.head_k_dim
        beta = ctx.beta
        threshold = ctx.threshold
        g_bias = ctx.g_bias
        b = ctx.b

        grad_g, grad_A, grad_gbias, grad_b = kda_gate_bwd(
            grad_output, g, A, head_k_dim, g_bias, b, gb, beta, threshold
        )
        return grad_g, grad_A, None, grad_gbias, grad_b, None, None


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Fused KDA gate computation with autograd support.

    Source: libs/fla/ops/kda/gate.py:L438-L461

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]

    Args:
        g: Input tensor of shape [..., num_heads * head_k_dim]
        A: Parameter tensor of shape [num_heads] or [1, 1, num_heads, 1]
        head_k_dim: Dimension of each head
        g_bias: Optional bias tensor [num_heads * head_k_dim]
        b: Optional tensor [..., num_heads]
        beta: softplus beta parameter (default: 1.0)
        threshold: softplus threshold parameter (default: 20.0)

    Returns:
        g_out: Output tensor of shape [..., num_heads, head_k_dim]
        b_sigmoid: (optional) Sigmoid of b if b is not None
    """
    g_out, b_sigmoid = KDAGateFunction.apply(g, A, head_k_dim, g_bias, b, beta, threshold)
    return (g_out, b_sigmoid) if b is not None else g_out


__all__ = [
    'kda_gate_ref',
    'kda_gate_fwd',
    'kda_gate_bwd',
    'KDAGateFunction',
    'fused_kda_gate',
]
