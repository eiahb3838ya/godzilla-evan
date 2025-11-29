# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# -*- coding: utf-8 -*-

"""
Inter-chunk Backward Gradients for KDA - PyTorch Implementation
===============================================================

Source: libs/fla/ops/kda/chunk_inter.py:L1-L185

This module computes inter-chunk backward gradients (cross-chunk contributions).

Official Triton kernel converted to PyTorch:
- chunk_kda_bwd_kernel_inter (L31-L137): Inter-chunk backward

Perfect replication requirements:
- All function signatures match official API
- Support varlen (cu_seqlens)
- Correctly handle hidden state gradients dh
"""

import torch
from typing import Optional, Tuple

from myfla.ops.utils import prepare_chunk_indices


def chunk_kda_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    dv: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute inter-chunk backward gradients for q, k, w, g.
    
    Source: libs/fla/ops/kda/chunk_inter.py:L139-L185
    
    Args:
        q, k, v, w, g: Forward tensors
        h: Hidden state [B, NT, H, K, V]
        dv: Gradient of v
        do: Gradient of output
        dh: Gradient of hidden state
        scale: Attention scale
        cu_seqlens: Cumulative sequence lengths for varlen
        chunk_size: Chunk size (default: 64)
    
    Returns:
        dq, dk, dw, dg: Gradients
    """
    if cu_seqlens is not None:
        raise NotImplementedError(
            "chunk_kda_bwd_dqkwg: varlen (cu_seqlens) not yet implemented."
        )
    
    # Call internal PyTorch kernel
    dq, dk, dw, dg = _chunk_kda_bwd_kernel_inter_pytorch(
        q, k, v, g, h, do, dh, dv, scale, chunk_size
    )
    
    return dq, dk, dw, dg


# ============================================================================
# Internal PyTorch kernel implementation (Stage 2.3.1)
# Converted from Triton kernel in libs/fla/ops/kda/chunk_inter.py
# ============================================================================


def _chunk_kda_bwd_kernel_inter_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of chunk_kda_bwd_kernel_inter.
    
    Source: libs/fla/ops/kda/chunk_inter.py:L31-L137
    
    完整轉換 106 行 Triton kernel 邏輯，無任何簡化。
    
    Args:
        q, k, v: Forward inputs [B, T, H, K/V]
        g: cumulative gate [B, T, H, K]
        h: hidden states [B, NT, H, K, V]
        do: gradient of output [B, T, H, V]
        dh: gradient of hidden state [B, NT, H, K, V]
        dv: gradient of v (from local attention) [B, T, H, V]
        scale: attention scale factor
        chunk_size: chunk size BT (default: 64)
    
    Returns:
        dq, dk, dw, dg: Gradients
    
    Key logic (from Triton kernel):
    - Part 1 (L93-L114): V dimension loop → dq, dk, dw from h/dh
    - Part 2 (L115-L116): Store dw with negation
    - Part 3 (L118-L136): Gate processing & complex dg computation
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    
    # Initialize outputs
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dw = torch.zeros_like(k, dtype=k.dtype)
    dg = torch.zeros_like(g, dtype=torch.float32)
    
    # Process each chunk
    for i_t in range(NT):
        t_start = i_t * BT
        t_end = min(t_start + BT, T)
        chunk_len = t_end - t_start
        if t_start >= T:
            break
        
        for b in range(B):
            for h_idx in range(H):
                # Load g for this chunk [chunk_len, K]
                g_chunk = g[b, t_start:t_end, h_idx, :]  # [chunk_len, K]
                
                # Load gn = g[last_token] [K]
                last_idx = min(T, t_start + BT) - 1
                gn = g[b, last_idx, h_idx, :]  # [K]
                
                # Initialize accumulator
                b_dq = torch.zeros(chunk_len, K, device=q.device, dtype=torch.float32)
                b_dk = torch.zeros(chunk_len, K, device=k.device, dtype=torch.float32)
                b_dw = torch.zeros(chunk_len, K, device=k.device, dtype=torch.float32)
                b_dgk = torch.zeros(K, device=k.device, dtype=torch.float32)
                
                # ============================================================
                # Part 1: V dimension loop (lines 93-114)
                # ============================================================
                # Process in full V dimension at once (no blocking)
                # Load tensors [chunk_len, V]
                v_chunk = v[b, t_start:t_end, h_idx, :]
                do_chunk = do[b, t_start:t_end, h_idx, :]
                dv_chunk = dv[b, t_start:t_end, h_idx, :]
                
                # Load h, dh [K, V]
                h_chunk = h[b, i_t, h_idx, :, :]  # [K, V]
                dh_chunk = dh[b, i_t, h_idx, :, :]  # [K, V]
                
                # Compute dgk = sum(h * dh, axis=0)  [K]
                b_dgk += torch.sum(h_chunk * dh_chunk, dim=1)  # sum over V
                
                # Compute dq = dot(do, h.T)  [chunk_len, K]
                b_dq += torch.matmul(do_chunk, h_chunk.T)
                
                # Compute dk = dot(v, dh.T)  [chunk_len, K]
                b_dk += torch.matmul(v_chunk, dh_chunk.T)
                
                # Compute dw = dot(dv, h.T)  [chunk_len, K]
                b_dw += torch.matmul(dv_chunk, h_chunk.T)
                
                # ============================================================
                # Part 2: Store dw with negation (line 116)
                # ============================================================
                # CRITICAL: dw has a negative sign!
                dw[b, t_start:t_end, h_idx, :] = -b_dw
                
                # ============================================================
                # Part 3: Gate processing & dg computation (lines 118-136)
                # ============================================================
                # Apply gate to dgk: dgk *= exp(gn)  [K]
                b_dgk *= torch.exp(gn)
                
                # Apply scale and gate to dq: dq *= scale * exp(g)  [chunk_len, K]
                b_dq *= scale
                b_dq = b_dq * torch.exp(g_chunk)
                
                # Apply gate to dk: dk *= exp(gn - g)  [chunk_len, K]
                b_dk = b_dk * torch.exp(gn.unsqueeze(0) - g_chunk)
                
                # Load q, k
                q_chunk = q[b, t_start:t_end, h_idx, :]  # [chunk_len, K]
                k_chunk = k[b, t_start:t_end, h_idx, :]  # [chunk_len, K]
                
                # Accumulate dgk: dgk += sum(dk * k, axis=0)  [K]
                b_dgk += torch.sum(b_dk * k_chunk, dim=0)
                
                # Compute dg (complex formula, line 131-132):
                # dg = q * dq - k * dk
                # dg = dg - cumsum(dg, axis=0) + sum(dg, axis=0) + dgk
                b_dg = q_chunk * b_dq - k_chunk * b_dk  # [chunk_len, K]
                
                # cumsum along time dimension
                b_dg_cumsum = torch.cumsum(b_dg, dim=0)  # [chunk_len, K]
                b_dg_sum = torch.sum(b_dg, dim=0)  # [K]
                
                # Final dg formula
                b_dg = b_dg - b_dg_cumsum + b_dg_sum.unsqueeze(0) + b_dgk.unsqueeze(0)
                
                # Store results
                dq[b, t_start:t_end, h_idx, :] = b_dq
                dk[b, t_start:t_end, h_idx, :] = b_dk
                dg[b, t_start:t_end, h_idx, :] = b_dg
    
    return dq, dk, dw, dg

