# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# -*- coding: utf-8 -*-

"""
Intra-chunk Local Attention for KDA - PyTorch Implementation
=============================================================

Source: libs/fla/ops/kda/chunk_intra.py:L1-L574

This module computes intra-chunk local attention matrices Aqk and Akk.

Official Triton kernels converted to PyTorch:
1. chunk_kda_fwd_kernel_intra_sub_inter (L27-L115): Inter-block attention (i > j)
2. chunk_kda_fwd_kernel_intra_sub_intra (L117-L191): Intra-block attention (i == j)
3. chunk_kda_bwd_kernel_intra (L193-L385): Backward gradients

Key computations:
- Aqk = q·exp(g-gn) @ (k·exp(gn-gk))^T * scale
- Akk = k·exp(g-gn) @ (k·exp(gn-gk))^T * beta

Perfect replication requirements:
- All function signatures match official API
- Support varlen (cu_seqlens)
- Support output_dtype conversion
- Numerically stable exp(g - gn) gating
"""

import torch
from typing import List, Optional, Tuple


def _build_sequence_infos(
    batch_size: int,
    seq_len: int,
    cu_seqlens: Optional[torch.LongTensor],
) -> List[Tuple[int, int, int]]:
    """
    Return list of (batch_index, base_offset, effective_length).
    Varlen 模式假定 batch=1，並沿 `cu_seqlens` 展開每條序列。
    """
    if cu_seqlens is None:
        return [(b, 0, seq_len) for b in range(batch_size)]
    if batch_size != 1:
        raise ValueError("Varlen 模式需先將 batch flatten（q.shape[0] 應為 1）")
    starts = cu_seqlens[:-1].tolist()
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    seq_infos = []
    for start, length in zip(starts, lengths):
        seq_infos.append((0, int(start), int(length)))
    return seq_infos


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    output_dtype: torch.dtype = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute intra-chunk local attention matrices Aqk and Akk.
    
    Source: libs/fla/ops/kda/chunk_intra.py:L387-L476
    
    Args:
        q (torch.Tensor): queries of shape [B, T, H, K]
        k (torch.Tensor): keys of shape [B, T, H, K]
        gk (torch.Tensor): cumulative gate (log space) of shape [B, T, H, K]
        beta (torch.Tensor): beta weights of shape [B, T, H]
        scale (float): attention scale factor (typically 1 / sqrt(K))
        cu_seqlens (Optional[torch.LongTensor]): cumulative sequence lengths for varlen
        output_dtype (torch.dtype): output data type (default: q.dtype)
    
    Returns:
        Aqk (torch.Tensor): q-k attention matrix of shape [B, H, T, BT]
        Akk (torch.Tensor): k-k similarity matrix of shape [B, H, T, BT]
    """
    # Set default output dtype
    if output_dtype is None:
        output_dtype = q.dtype
    
    if cu_seqlens is None:
        Aqk_inter, Akk_inter = _chunk_kda_fwd_kernel_intra_sub_inter_pytorch(
            q, k, gk, beta, scale, chunk_size=64
        )
        Aqk_intra, Akk_intra = _chunk_kda_fwd_kernel_intra_sub_intra_pytorch(
            q, k, gk, beta, scale, chunk_size=64
        )
        Aqk = Aqk_inter + Aqk_intra
        Akk = Akk_inter + Akk_intra
    else:
        if q.shape[0] != 1:
            raise ValueError("Varlen 模式需先 flatten batch (q.shape[0] 應為 1)")
        Aqk_inter, Akk_inter = _chunk_kda_fwd_kernel_intra_sub_inter_pytorch(
            q, k, gk, beta, scale, chunk_size=64, cu_seqlens=cu_seqlens
        )
        Aqk_intra, Akk_intra = _chunk_kda_fwd_kernel_intra_sub_intra_pytorch(
            q, k, gk, beta, scale, chunk_size=64, cu_seqlens=cu_seqlens
        )
        Aqk = Aqk_inter + Aqk_intra
        Akk = Akk_inter + Akk_intra

    # Convert to output dtype (Triton keeps fp32, then converts)
    Aqk = Aqk.to(output_dtype)
    Akk = Akk.to(output_dtype)
    
    return Aqk, Akk


def _chunk_kda_bwd_kernel_intra_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dg: torch.Tensor,
    db: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of chunk_kda_bwd_kernel_intra.
    
    Source: libs/fla/ops/kda/chunk_intra.py:L193-L385
    
    完整轉換 193 行 Triton kernel 邏輯，無任何簡化。
    
    Args:
        q, k, g, beta: Forward inputs [B, T, H, K], [B, T, H, K], [B, T, H, K], [B, T, H]
        dAqk, dAkk: Gradients of attention matrices [B, H, T, BT]
        dq, dk, dg, db: Gradient accumulators (will be updated in-place)
        chunk_size: Chunk size BC (default: 64)
    
    Returns:
        dq, dk, dg, db: Updated gradients
    
    Key logic (from Triton kernel):
    - Part 1 (L255-277): Inter-block backward (i > j)
    - Part 2 (L290-315): Intra-block diagonal backward + dbeta + dg(q)
    - Part 3 (L318-384): dk backward from later blocks + dg(k)
    """
    B, T, H, K = q.shape
    BC = chunk_size
    BT = 64  # chunk size
    NC = BT // BC  # number of sub-chunks per chunk
    dq2 = torch.zeros_like(q)
    dk2 = torch.zeros_like(k)
    seq_infos = _build_sequence_infos(B, T, cu_seqlens)
    
    for b, base, length in seq_infos:
        if length <= 0:
            continue
        seq_end = base + length
        NT_local = (length + BT - 1) // BT
        for i_t in range(NT_local):
            rel_chunk_start = i_t * BT
            if rel_chunk_start >= length:
                break
            t_start = base + rel_chunk_start
            t_end = min(t_start + BT, seq_end)
            for h in range(H):
                # Loop over sub-chunks (i_i)
                for i_i in range(NC):
                    i_start = t_start + i_i * BC
                    i_end = min(i_start + BC, seq_end)
                    if i_start >= seq_end:
                        continue
                    
                    # Extract blocks
                    g_block_i = g[b, i_start:i_end, h, :]  # [BC, K]
                    beta_block = beta[b, i_start:i_end, h]  # [BC]
                    q_block = q[b, i_start:i_end, h, :]  # [BC, K]
                    k_block_i = k[b, i_start:i_end, h, :]  # [BC, K]
                    
                    b_dq2 = torch.zeros(i_end - i_start, K, device=q.device, dtype=torch.float32)
                    b_dk2 = torch.zeros(i_end - i_start, K, device=k.device, dtype=torch.float32)
                    
                    # ============================================================
                    # Part 1: Inter-block backward (i_i > 0, lines 257-277)
                    # ============================================================
                    if i_i > 0:
                        # gn = g[first_token_of_block_i]  [K]
                        gn = g[b, i_start, h, :]
                        
                        for i_j in range(0, i_i):
                            j_start = t_start + i_j * BC
                            if j_start >= seq_end:
                                break
                            j_end = min(j_start + BC, seq_end)
                            # Load blocks for j
                            k_block_j = k[b, j_start:j_end, h, :]  # [BC, K]
                            g_block_j = g[b, j_start:j_end, h, :]  # [BC, K]
                            
                            # Load gradients dAqk[i,j], dAkk[i,j]（列索引用 chunk 內偏移）
                            col_offset = j_start - t_start
                            col_slice = slice(col_offset, col_offset + (j_end - j_start))
                            dAqk_block = dAqk[b, h, i_start:i_end, col_slice]  # [BC, BC]
                            dAkk_block = dAkk[b, h, i_start:i_end, col_slice]  # [BC, BC]
                            
                            # Compute k_gated = k * exp(gn - gk)  [BC, K]
                            k_gated = k_block_j * torch.exp(gn.unsqueeze(0) - g_block_j)
                            
                            # Accumulate gradients: dq2 += dAqk @ k_gated, dk2 += dAkk @ k_gated
                            actual_i_len = i_end - i_start
                            actual_j_len = j_end - j_start
                            b_dq2[:actual_i_len, :] += torch.matmul(
                                dAqk_block[:actual_i_len, :actual_j_len], k_gated[:actual_j_len, :]
                            )
                            b_dk2[:actual_i_len, :] += torch.matmul(
                                dAkk_block[:actual_i_len, :actual_j_len], k_gated[:actual_j_len, :]
                            )
                        
                        # Apply gate: dq2 *= exp(g - gn), dk2 *= exp(g - gn)
                        b_dq2 *= torch.exp(g_block_i - gn.unsqueeze(0))
                        b_dk2 *= torch.exp(g_block_i - gn.unsqueeze(0))
                    
                    # ============================================================
                    # Part 2: Intra-block (diagonal) backward (lines 290-315)
                    # ============================================================
                    # Loop over tokens within the block (causal mask)
                    for j in range(min(BC, seq_end - i_start)):
                        actual_idx = i_start + j
                        if actual_idx >= seq_end:
                            break
                        
                        # Load single token gradients（column offset 相對於 chunk 起點）
                        col_idx = actual_idx - t_start
                        if col_idx < 0 or col_idx >= dAqk.shape[-1]:
                            continue
                        dAqk_j = dAqk[b, h, i_start:i_end, col_idx]  # [BC]
                        dAkk_j = dAkk[b, h, i_start:i_end, col_idx]  # [BC]
                        
                        # Load k[j], g[j]
                        k_j = k[b, actual_idx, h, :]  # [K]
                        g_j = g[b, actual_idx, h, :]  # [K]
                        
                        # Causal mask: only i >= j contributes
                        # Create mask [BC, K]
                        mask = torch.arange(i_end - i_start, device=q.device).unsqueeze(1) >= j
                        mask = mask.float()
                        
                        # Compute gated contributions
                        gate_factor = torch.exp(g_block_i - g_j.unsqueeze(0))  # [BC, K]
                        
                        # Accumulate gradients with causal mask
                        b_dq2[:i_end - i_start, :] += mask * dAqk_j.unsqueeze(1) * k_j.unsqueeze(0) * gate_factor
                        b_dk2[:i_end - i_start, :] += mask * dAkk_j.unsqueeze(1) * k_j.unsqueeze(0) * gate_factor
                    
                    # Compute dbeta = sum(dk2 * k, dim=1)  [BC]
                    b_db = torch.sum(b_dk2[:i_end - i_start, :] * k_block_i, dim=1)
                    
                    # Apply beta: dk2 *= beta
                    b_dk2 *= beta_block.unsqueeze(1)
                    
                    # Compute dg_q = q * dq2  [BC, K]
                    b_dg = q_block * b_dq2[:i_end - i_start, :]
                    
                    # Accumulate dq2 with existing dq
                    b_dq2 += dq[b, i_start:i_end, h, :]
                    
                    # Store dq2, db
                    dq2[b, i_start:i_end, h, :] = b_dq2[:i_end - i_start, :]
                    db[b, i_start:i_end, h] = b_db
                    
                    # ============================================================
                    # Part 3: dk backward from later blocks (lines 318-384)
                    # ============================================================
                    b_dkt = torch.zeros(i_end - i_start, K, device=k.device, dtype=torch.float32)
                    
                    # Only process if not the last sub-chunk
                    if i_i < NC - 1 and (i_start + BC) < seq_end:
                        # gn = g[last_token_of_block_i]  [K]
                        last_idx = min(i_start + BC, seq_end) - 1
                        gn = g[b, last_idx, h, :]
                        
                        # Loop over later sub-chunks (j > i)
                        for i_j in range(i_i + 1, NC):
                            j_start = t_start + i_j * BC
                            if j_start >= seq_end:
                                break
                            j_end = min(j_start + BC, seq_end)
                            
                            # Load blocks for j
                            q_block_j = q[b, j_start:j_end, h, :]  # [BC, K]
                            k_block_j = k[b, j_start:j_end, h, :]  # [BC, K]
                            g_block_j = g[b, j_start:j_end, h, :]  # [BC, K]
                            beta_block_j = beta[b, j_start:j_end, h]  # [BC]
                            
                            # Load gradients dAqk[j,i], dAkk[j,i] (transposed!)，列索引用 chunk 偏移
                            col_slice_i = slice(i_start - t_start, i_end - t_start)
                            dAqk_block_T = dAqk[b, h, j_start:j_end, col_slice_i].T  # [BC_i, BC_j]
                            dAkk_block_T = dAkk[b, h, j_start:j_end, col_slice_i].T  # [BC_i, BC_j]
                            
                            # Compute gated q, k
                            gate_factor = torch.exp(g_block_j - gn.unsqueeze(0))  # [BC_j, K]
                            q_gated = q_block_j * gate_factor
                            k_gated = k_block_j * beta_block_j.unsqueeze(1) * gate_factor
                            
                            # Accumulate: dkt += dAqk.T @ q_gated + dAkk.T @ k_gated
                            actual_i_len = i_end - i_start
                            actual_j_len = j_end - j_start
                            b_dkt[:actual_i_len, :] += torch.matmul(
                                dAqk_block_T[:actual_i_len, :actual_j_len], q_gated[:actual_j_len, :]
                            )
                            b_dkt[:actual_i_len, :] += torch.matmul(
                                dAkk_block_T[:actual_i_len, :actual_j_len], k_gated[:actual_j_len, :]
                            )
                        
                        # Apply gate: dkt *= exp(gn - g)
                        b_dkt *= torch.exp(gn.unsqueeze(0) - g_block_i)
                    
                    # Process diagonal contributions to dkt (lines 358-374)
                    for j in range(min(BC, seq_end - i_start)):
                        actual_idx = i_start + j
                        if actual_idx >= seq_end:
                            break
                        
                        # Load single token gradients (transposed indices!)，列索引用 chunk 偏移
                        col_slice_i = slice(i_start - t_start, i_end - t_start)
                        dAqk_j = dAqk[b, h, actual_idx, col_slice_i]  # [BC]
                        dAkk_j = dAkk[b, h, actual_idx, col_slice_i]  # [BC]
                        
                        # Load q[j], k[j], g[j], beta[j]
                        q_j = q[b, actual_idx, h, :]  # [K]
                        k_j = k[b, actual_idx, h, :]  # [K]
                        g_j = g[b, actual_idx, h, :]  # [K]
                        beta_j = beta[b, actual_idx, h]
                        
                        # Causal mask: only i <= j contributes (reverse of Part 2!)
                        mask = torch.arange(i_end - i_start, device=k.device).unsqueeze(1) <= j
                        mask = mask.float()
                        
                        # Compute gated contributions
                        gate_factor = torch.exp(g_j.unsqueeze(0) - g_block_i)  # [BC, K]
                        
                        # Accumulate dkt with causal mask
                        b_dkt[:i_end - i_start, :] += mask * dAqk_j.unsqueeze(1) * q_j.unsqueeze(0) * gate_factor
                        b_dkt[:i_end - i_start, :] += mask * dAkk_j.unsqueeze(1) * (k_j * beta_j).unsqueeze(0) * gate_factor
                    
                    # Compute dg_k = (dk2 - dkt) * k  [BC, K]
                    b_dg += (b_dk2[:i_end - i_start, :] - b_dkt) * k_block_i
                    
                    # Accumulate dk2 with existing dk + dkt
                    b_dk2 += dk[b, i_start:i_end, h, :] + b_dkt
                    
                    # Store dk2, dg
                    dk2[b, i_start:i_end, h, :] = b_dk2[:i_end - i_start, :]
                    dg[b, i_start:i_end, h, :] = b_dg
    
    # Return updated gradients (dq2/dk2 are the final dq/dk)
    return dq2, dk2, db, dg


def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    db: Optional[torch.Tensor] = None,
    dg: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute backward gradients for intra-chunk attention.
    
    Source: libs/fla/ops/kda/chunk_intra.py:L480-L574
    
    Args:
        q, k, g, beta: Forward inputs
        dAqk, dAkk: Gradients of attention matrices
        dq, dk, db, dg: Accumulated gradients (optional, will be created if None)
        cu_seqlens: Cumulative sequence lengths for varlen
        chunk_size: Chunk size (default: 64)
    
    Returns:
        dq, dk, db, dg: Updated gradients
    """
    # Initialize gradient tensors if not provided
    B, T, H, K = q.shape
    if dq is None:
        dq = torch.zeros_like(q)
    if dk is None:
        dk = torch.zeros_like(k)
    if db is None:
        db = torch.zeros_like(beta)
    if dg is None:
        dg = torch.zeros_like(g)
    
    if cu_seqlens is None:
        dq, dk, db, dg = _chunk_kda_bwd_kernel_intra_pytorch(
            q, k, g, beta, dAqk, dAkk, dq, dk, dg, db, chunk_size
        )
    else:
        if q.shape[0] != 1:
            raise ValueError("Varlen 模式需先 flatten batch (q.shape[0] 應為 1)")
        dq, dk, db, dg = _chunk_kda_bwd_kernel_intra_pytorch(
            q,
            k,
            g,
            beta,
            dAqk,
            dAkk,
            dq,
            dk,
            dg,
            db,
            chunk_size,
            cu_seqlens=cu_seqlens,
        )

    return dq, dk, db, dg


# ============================================================================
# Internal PyTorch kernel implementations (Stage 2.1.1-2.1.4)
# Converted from Triton kernels in libs/fla/ops/kda/chunk_intra.py
# ============================================================================


def _chunk_kda_fwd_kernel_intra_sub_inter_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of chunk_kda_fwd_kernel_intra_sub_inter.
    
    Source: libs/fla/ops/kda/chunk_intra.py:L27-L102
    
    Computes inter-block attention (i > j) for intra-chunk local attention.
    
    Args:
        q: queries [B, T, H, K]
        k: keys [B, T, H, K]
        g: cumulative gate (log space) [B, T, H, K]
        beta: beta weights [B, T, H]
        scale: attention scale factor
        chunk_size: chunk size (BC in Triton, default 64)
    
    Returns:
        Aqk: q-k attention matrix [B, H, T, chunk_size]
        Akk: k-k similarity matrix [B, H, T, chunk_size]
    
    Key logic (from Triton kernel):
    - Only compute for i_i > i_j (inter-block)
    - gn = g[first_token_of_block_i] (normalization base)
    - b_k = k * exp(g - gn)
    - b_ktg = k.T * exp(gn - gk)
    - Akk = dot(b_k, b_ktg) * beta
    - Aqk = dot(q * exp(g - gn) * scale, b_ktg)
    """
    B, T, H, K = q.shape
    BC = chunk_size
    BT = 64
    NC = BT // BC
    Aqk = torch.zeros(B, H, T, BT, dtype=torch.float32, device=q.device)
    Akk = torch.zeros(B, H, T, BT, dtype=torch.float32, device=q.device)
    seq_infos = _build_sequence_infos(B, T, cu_seqlens)
    
    for b, base, length in seq_infos:
        if length <= 0:
            continue
        NT_local = (length + BT - 1) // BT
        for i_t in range(NT_local):
            rel_chunk_start = i_t * BT
            if rel_chunk_start >= length:
                break
            t_start = base + rel_chunk_start
            t_end = min(base + length, t_start + BT)
            for h in range(H):
                for i_c in range(NC * NC):
                    i_i = i_c // NC
                    i_j = i_c % NC
                    if i_i <= i_j:
                        continue
                    i_start = t_start + i_i * BC
                    if i_start >= base + length:
                        continue
                    i_end = min(i_start + BC, base + length)
                    j_start = t_start + i_j * BC
                    if j_start >= base + length:
                        continue
                    j_end = min(j_start + BC, base + length)
                    
                    # Extract blocks
                    # q_block, k_block_i, g_block_i: [BC, K]
                    # k_block_j, g_block_j: [BC, K]
                    q_block = q[b, i_start:i_end, h, :]  # [BC, K]
                    k_block_i = k[b, i_start:i_end, h, :]  # [BC, K]
                    g_block_i = g[b, i_start:i_end, h, :]  # [BC, K]
                    k_block_j = k[b, j_start:j_end, h, :]  # [BC, K]
                    g_block_j = g[b, j_start:j_end, h, :]  # [BC, K]
                    beta_block = beta[b, i_start:i_end, h]  # [BC]
                    
                    # Gate normalization: gn = g[first_token] [K]
                    gn = g[b, i_start, h, :]  # [K]
                    
                    # Apply gate: k_i * exp(g_i - gn) [BC, K]
                    b_k = k_block_i * torch.exp(g_block_i - gn.unsqueeze(0))
                    
                    # Apply gate to k_j: k_j.T * exp(gn - g_j) [K, BC]
                    b_ktg = k_block_j.T * torch.exp(gn.unsqueeze(1) - g_block_j.T)
                    
                    # Compute Akk: [BC, K] @ [K, BC] = [BC, BC]
                    b_Akk = torch.matmul(b_k, b_ktg) * beta_block.unsqueeze(1)
                    
                    # Compute Aqk: q * exp(g - gn) * scale
                    b_qg = q_block * torch.exp(g_block_i - gn.unsqueeze(0)) * scale
                    b_Aqk = torch.matmul(b_qg, b_ktg)  # [BC, BC]
                    
                    # Store results (handle boundary)
                    actual_i_len = i_end - i_start
                    actual_j_len = j_end - j_start
                    col_offset = j_start - t_start
                    col_slice = slice(col_offset, col_offset + actual_j_len)
                    Akk[b, h, i_start:i_end, col_slice] = b_Akk[:actual_i_len, :actual_j_len]
                    Aqk[b, h, i_start:i_end, col_slice] = b_Aqk[:actual_i_len, :actual_j_len]
    
    return Aqk, Akk


def _chunk_kda_fwd_kernel_intra_sub_intra_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of chunk_kda_fwd_kernel_intra_sub_intra.
    
    Source: libs/fla/ops/kda/chunk_intra.py:L117-L191
    
    Computes intra-block attention (i == j, diagonal) for intra-chunk local attention.
    
    Args:
        q: queries [B, T, H, K]
        k: keys [B, T, H, K]
        g: cumulative gate (log space) [B, T, H, K]
        beta: beta weights [B, T, H]
        scale: attention scale factor
        chunk_size: chunk size (BC in Triton, default 64)
    
    Returns:
        Aqk: q-k attention matrix [B, H, T, chunk_size] (diagonal blocks)
        Akk: k-k similarity matrix [B, H, T, chunk_size] (diagonal blocks)
    
    Key logic (from Triton kernel):
    - Only compute for i_i == i_j (diagonal blocks)
    - Apply causal mask within block
    - Similar gate mechanism as sub_inter
    """
    B, T, H, K = q.shape
    BC = chunk_size
    BT = 64
    NC = BT // BC
    Aqk = torch.zeros(B, H, T, BT, dtype=torch.float32, device=q.device)
    Akk = torch.zeros(B, H, T, BT, dtype=torch.float32, device=q.device)
    seq_infos = _build_sequence_infos(B, T, cu_seqlens)
    
    for b, base, length in seq_infos:
        if length <= 0:
            continue
        NT_local = (length + BT - 1) // BT
        for i_t in range(NT_local):
            rel_chunk_start = i_t * BT
            if rel_chunk_start >= length:
                break
            t_start = base + rel_chunk_start
            for h in range(H):
                for i_i in range(NC):
                    i_start = t_start + i_i * BC
                    if i_start >= base + length:
                        continue
                    i_end = min(i_start + BC, base + length)
                    
                    # Extract blocks (same block for i and j)
                    q_block = q[b, i_start:i_end, h, :]  # [BC, K]
                    k_block = k[b, i_start:i_end, h, :]  # [BC, K]
                    g_block = g[b, i_start:i_end, h, :]  # [BC, K]
                    beta_block = beta[b, i_start:i_end, h]  # [BC]
                    
                    # Gate normalization
                    gn = g[b, i_start, h, :]  # [K]
                    
                    # Apply gate
                    b_k = k_block * torch.exp(g_block - gn.unsqueeze(0))
                    b_ktg = k_block.T * torch.exp(gn.unsqueeze(1) - g_block.T)
                    
                    # Compute attention with causal mask
                    b_Akk = torch.matmul(b_k, b_ktg) * beta_block.unsqueeze(1)
                    b_qg = q_block * torch.exp(g_block - gn.unsqueeze(0)) * scale
                    b_Aqk = torch.matmul(b_qg, b_ktg)
                    
                    # Apply causal mask (lower triangular)
                    actual_len = i_end - i_start
                    causal_mask = torch.tril(torch.ones(actual_len, actual_len, device=q.device))
                    b_Akk = b_Akk[:actual_len, :actual_len] * causal_mask
                    b_Aqk = b_Aqk[:actual_len, :actual_len] * causal_mask
                    
                    # Store results
                    col_offset = i_start - t_start
                    col_slice = slice(col_offset, col_offset + actual_len)
                    Akk[b, h, i_start:i_end, col_slice] = b_Akk
                    Aqk[b, h, i_start:i_end, col_slice] = b_Aqk
    
    return Aqk, Akk
