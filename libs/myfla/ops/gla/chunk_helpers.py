# Copyright (c) 2023-2025, myfla project
# Helper functions for GLA chunk operations

"""
Helper utilities for GLA chunk-based computations.

Includes:
- reshape_qkv: Reshape and validate q/k/v tensors for multi-head processing (Task 3.3.2.a)
"""

from typing import Tuple
import torch


def reshape_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reshape q/k/v tensors to ensure correct memory layout for chunk processing.

    Args:
        q: Query tensor, shape [B, T, H, K] or [B, T, H*K]
        k: Key tensor, same shape as q
        v: Value tensor, shape [B, T, H, V] or [B, T, H*V]
        num_heads: Number of attention heads (H)
        head_dim: Dimension per head for q/k (K)

    Returns:
        q_reshaped: [B, T, H, K], contiguous
        k_reshaped: [B, T, H, K], contiguous
        v_reshaped: [B, T, H, V], contiguous

    Raises:
        ValueError: If shapes are inconsistent

    Note:
        - 確保所有張量都是 contiguous(),避免後續 view/reshape 錯誤
        - 若輸入已是 4D 且 head 維度正確,僅確保 contiguous
        - 若輸入是 3D,自動 reshape 到 [B, T, H, K/V]

    Example:
        >>> q = torch.randn(2, 16, 8, 64)  # [B, T, H, K]
        >>> k = torch.randn(2, 16, 8, 64)
        >>> v = torch.randn(2, 16, 8, 128)  # [B, T, H, V]
        >>> q_r, k_r, v_r = reshape_qkv(q, k, v, num_heads=8, head_dim=64)
        >>> assert q_r.shape == (2, 16, 8, 64)
        >>> assert q_r.is_contiguous()
    """
    B, T = q.shape[0], q.shape[1]

    # Validate q/k have same shape
    if q.shape != k.shape:
        raise ValueError(f"q and k must have same shape, got q={q.shape}, k={k.shape}")

    # Handle q/k reshaping
    if q.ndim == 3:
        # [B, T, H*K] -> [B, T, H, K]
        total_dim = q.shape[2]
        if total_dim != num_heads * head_dim:
            raise ValueError(
                f"Expected q.shape[2]={total_dim} to equal num_heads * head_dim = {num_heads * head_dim}"
            )
        q_reshaped = q.view(B, T, num_heads, head_dim).contiguous()
        k_reshaped = k.view(B, T, num_heads, head_dim).contiguous()

    elif q.ndim == 4:
        # Already [B, T, H, K], verify dimensions
        H, K = q.shape[2], q.shape[3]
        if H != num_heads or K != head_dim:
            raise ValueError(
                f"Expected q.shape[2:] = ({num_heads}, {head_dim}), got ({H}, {K})"
            )
        q_reshaped = q.contiguous()
        k_reshaped = k.contiguous()

    else:
        raise ValueError(f"q must be 3D or 4D, got shape {q.shape}")

    # Handle v reshaping
    if v.shape[0] != B or v.shape[1] != T:
        raise ValueError(
            f"v batch/seq dims must match q: expected ({B}, {T}), got ({v.shape[0]}, {v.shape[1]})"
        )

    if v.ndim == 3:
        # [B, T, H*V] -> [B, T, H, V]
        total_v_dim = v.shape[2]
        if total_v_dim % num_heads != 0:
            raise ValueError(
                f"v.shape[2]={total_v_dim} must be divisible by num_heads={num_heads}"
            )
        value_dim = total_v_dim // num_heads
        v_reshaped = v.view(B, T, num_heads, value_dim).contiguous()

    elif v.ndim == 4:
        # Already [B, T, H, V], verify H matches
        H_v = v.shape[2]
        if H_v != num_heads:
            raise ValueError(
                f"v.shape[2]={H_v} must equal num_heads={num_heads}"
            )
        v_reshaped = v.contiguous()

    else:
        raise ValueError(f"v must be 3D or 4D, got shape {v.shape}")

    return q_reshaped, k_reshaped, v_reshaped
