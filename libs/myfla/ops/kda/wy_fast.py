# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# -*- coding: utf-8 -*-

"""
WY Representation (Woodbury Decomposition) for KDA - PyTorch Implementation
===========================================================================

Source: libs/fla/ops/kda/wy_fast.py:L1-L301

This module computes WY representation for efficient recurrent state updates.

Official Triton kernels converted to PyTorch:
1. recompute_w_u_fwd_kernel (L29-L117): WY decomposition forward
2. prepare_wy_repr_bwd_kernel (L119-L209): WY decomposition backward

Key computations (Woodbury matrix identity):
- u = A^{-1} @ v  (solved via solve_tril)
- w = v - u
- kg = k * exp(g)  (gated keys)

Perfect replication requirements:
- All function signatures match official API
- Support varlen (cu_seqlens)
- Numerically stable solve_tril usage
- Optional qg/kg caching
"""

import torch
from typing import Optional, Tuple

from myfla.ops.utils import prepare_chunk_indices
from myfla.ops.utils.solve_tril import solve_tril


def recompute_w_u_fwd(
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    A: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Compute WY representation: w = v - u, u = A^{-1} @ v.
    
    Source: libs/fla/ops/kda/wy_fast.py:L211-L254
    
    Args:
        q (Optional[torch.Tensor]): queries [B, T, H, K] (for qg caching)
        k (torch.Tensor): keys [B, T, H, K]
        v (torch.Tensor): values [B, T, H, V]
        beta (torch.Tensor): beta weights [B, T, H]
        A (torch.Tensor): Akk matrix [B, H, T, BT]
        gk (torch.Tensor): cumulative gate [B, T, H, K]
        cu_seqlens (Optional[torch.LongTensor]): cumulative sequence lengths
    
    Returns:
        w (torch.Tensor): WY component w [B, T, H, K]
        u (torch.Tensor): WY component u [B, T, H, V]
        qg (Optional[torch.Tensor]): q * exp(gk) if q provided, else None
        kg (torch.Tensor): k * exp(gk) [B, T, H, K]
    """
    store_qg = q is not None
    store_kg = True  # Always compute kg for KDA
    
    if cu_seqlens is None:
        w, u, qg, kg = _recompute_w_u_fwd_pytorch(
            q if q is not None else torch.zeros_like(k),
            k,
            v,
            beta,
            A,
            gk,
            store_qg=store_qg,
            store_kg=store_kg,
        )
    else:
        if k.shape[0] != 1:
            raise ValueError("Varlen 模式需 flatten batch (k.shape[0] 應為 1)")
        w = torch.zeros_like(k)
        u = torch.zeros_like(v)
        qg = torch.zeros_like(k) if store_qg else None
        kg = torch.zeros_like(k)
        seq_count = len(cu_seqlens) - 1
        for idx in range(seq_count):
            start = int(cu_seqlens[idx].item())
            end = int(cu_seqlens[idx + 1].item())
            if end <= start:
                continue
            q_slice = q[:, start:end, :, :] if q is not None else torch.zeros_like(k[:, start:end, :, :])
            w_seq, u_seq, qg_seq, kg_seq = _recompute_w_u_fwd_pytorch(
                q_slice,
                k[:, start:end, :, :],
                v[:, start:end, :, :],
                beta[:, start:end, :],
                A[:, :, start:end, :],
                gk[:, start:end, :, :],
                store_qg=store_qg,
                store_kg=store_kg,
            )
            w[:, start:end, :, :] = w_seq
            u[:, start:end, :, :] = u_seq
            if store_qg and qg is not None and qg_seq is not None:
                qg[:, start:end, :, :] = qg_seq
            if kg_seq is not None:
                kg[:, start:end, :, :] = kg_seq
    
    return w, u, qg, kg


def _prepare_wy_repr_bwd_pytorch(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of prepare_wy_repr_bwd_kernel.
    
    Source: libs/fla/ops/kda/wy_fast.py:L119-L209
    
    完整轉換 91 行 Triton kernel 邏輯，無任何簡化。
    
    Args:
        k, v: keys/values [B, T, H, K/V]
        beta: beta weights [B, T, H]
        gk: cumulative gate [B, T, H, K]
        A: forward A matrix [B, H, T, BT]
        dw, du: gradients of w and u [B, T, H, K/V]
    
    Returns:
        dk, dv, dbeta, dg, dA: gradients
    
    Key logic (from Triton kernel):
    - Part 1 (K loop): Backprop dw → dk, db, dg, dA
    - Part 2 (V loop): Backprop du → dv, db, dA
    - Part 3: Process dA with triangular mask & transformation
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = 64  # chunk size
    NT = (T + BT - 1) // BT
    
    # Initialize outputs
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)
    dg = torch.zeros_like(gk)
    dA = torch.zeros(B, H, T, BT, device=k.device, dtype=torch.float32)
    
    # Process each chunk
    for i_t in range(NT):
        t_start = i_t * BT
        t_end = min(t_start + BT, T)
        chunk_len = t_end - t_start
        if t_start >= T:
            break
        
        for b in range(B):
            for h in range(H):
                # Load A matrix for this chunk [BT, chunk_len]
                # Note: A is [B, H, T, BT], need [BT, chunk_len]
                A_chunk = A[b, h, t_start:t_end, :BT]  # [chunk_len, BT]
                A_chunk_full = torch.zeros(BT, BT, device=k.device, dtype=torch.float32)
                A_chunk_full[:chunk_len, :chunk_len] = A_chunk[:, :chunk_len].to(torch.float32)
                A_chunk_T = A_chunk_full.T  # [BT, BT]
                
                # Load beta
                beta_chunk = beta[b, t_start:t_end, h]  # [chunk_len]
                
                b_db = torch.zeros(BT, device=k.device, dtype=torch.float32)
                b_dA = torch.zeros(BT, BT, device=k.device, dtype=torch.float32)
                
                # ============================================================
                # Part 1: K dimension loop (lines 161-182)
                # ============================================================
                # Process in full K dimension at once (no blocking for simplicity)
                k_chunk = k[b, t_start:t_end, h, :]  # [chunk_len, K]
                dw_chunk = dw[b, t_start:t_end, h, :]  # [chunk_len, K]
                gk_chunk = gk[b, t_start:t_end, h, :]  # [chunk_len, K]
                
                # Compute k * beta * exp(gk)  [chunk_len, K]
                gk_exp = torch.exp(gk_chunk)
                kbg = k_chunk * beta_chunk.unsqueeze(1) * gk_exp
                
                # Pad to BT
                kbg_padded = torch.zeros(BT, K, device=k.device, dtype=torch.float32)
                kbg_padded[:chunk_len, :] = kbg.to(torch.float32)
                dw_padded = torch.zeros(BT, K, device=k.device, dtype=torch.float32)
                dw_padded[:chunk_len, :] = dw_chunk.to(torch.float32)
                
                # Compute dA contribution: dA += dw @ kbg.T
                b_dA += torch.matmul(dw_padded, kbg_padded.T)
                
                # Compute dkbg = A.T @ dw  [chunk_len, K]
                dkbg = torch.matmul(A_chunk_T, dw_padded)[:chunk_len, :]
                
                # Compute dk = dkbg * exp(gk) * beta
                b_dk = (dkbg * gk_exp * beta_chunk.unsqueeze(1)).to(k.dtype)
                
                # Compute db contribution: db += sum(dkbg * k * exp(gk), 1)
                b_db[:chunk_len] += torch.sum(dkbg * k_chunk.to(torch.float32) * gk_exp.to(torch.float32), dim=1)
                
                # Compute dg = kbg * dkbg  [chunk_len, K]
                b_dg = (kbg.to(torch.float32) * dkbg).to(gk.dtype)
                
                # Store dk, dg
                dk[b, t_start:t_end, h, :] = b_dk
                dg[b, t_start:t_end, h, :] = b_dg
                
                # ============================================================
                # Part 2: V dimension loop (lines 183-194)
                # ============================================================
                v_chunk = v[b, t_start:t_end, h, :]  # [chunk_len, V]
                du_chunk = du[b, t_start:t_end, h, :]  # [chunk_len, V]
                
                # Compute v * beta  [chunk_len, V]
                vb = v_chunk * beta_chunk.unsqueeze(1)
                
                # Pad to BT
                vb_padded = torch.zeros(BT, V, device=v.device, dtype=torch.float32)
                vb_padded[:chunk_len, :] = vb.to(torch.float32)
                du_padded = torch.zeros(BT, V, device=v.device, dtype=torch.float32)
                du_padded[:chunk_len, :] = du_chunk.to(torch.float32)
                
                # Compute dA contribution: dA += du @ vb.T
                b_dA += torch.matmul(du_padded, vb_padded.T)
                
                # Compute dvb = A.T @ du  [chunk_len, V]
                dvb = torch.matmul(A_chunk_T, du_padded)[:chunk_len, :]
                
                # Compute dv = dvb * beta
                b_dv = dvb * beta_chunk.unsqueeze(1)
                
                # Compute db contribution: db += sum(dvb * v, 1)
                b_db[:chunk_len] += torch.sum(dvb * v_chunk.to(torch.float32), dim=1)
                
                # Store dv
                dv[b, t_start:t_end, h, :] = b_dv.to(v.dtype)
                
                # ============================================================
                # Part 3: dA processing (lines 196-208)
                # ============================================================
                # Create strictly upper triangular mask (i > j)
                o_t = torch.arange(BT, device=k.device)
                m_t = o_t < chunk_len
                m_A = (o_t.unsqueeze(1) > o_t.unsqueeze(0)) & (m_t.unsqueeze(1) & m_t.unsqueeze(0))
                
                # Apply mask
                b_dA = torch.where(m_A, b_dA, torch.zeros_like(b_dA))
                
                # Transform: dA = A @ dA @ A  (lines 200-201)
                b_dA = torch.matmul(b_dA, A_chunk_full)  # [BT, BT]
                b_dA = torch.matmul(A_chunk_full, b_dA)  # [BT, BT]
                
                # Apply negation with mask (line 203)
                b_dA = torch.where(m_A, -b_dA, torch.zeros_like(b_dA))
                
                # Store dA, db
                dA[b, h, t_start:t_end, :BT] = b_dA[:chunk_len, :]
                dbeta[b, t_start:t_end, h] = b_db[:chunk_len].to(beta.dtype)
    
    return dk, dv, dbeta, dg, dA


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute backward gradients for WY representation.
    
    Source: libs/fla/ops/kda/wy_fast.py:L256-L301
    
    Args:
        k, v, beta, gk, A: Forward inputs
        dw, du: Gradients of w and u
        cu_seqlens: Cumulative sequence lengths
    
    Returns:
        dk, dv, dbeta, dg, dA: Gradients
    """
    if cu_seqlens is None:
        dk, dv, dbeta, dg, dA = _prepare_wy_repr_bwd_pytorch(
            k, v, beta, gk, A, dw, du
        )
        return dk, dv, dbeta, dg, dA
    if k.shape[0] != 1:
        raise ValueError("Varlen 模式需 flatten batch (k.shape[0] 應為 1)")
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)
    dg = torch.zeros_like(gk)
    dA = torch.zeros_like(A, dtype=torch.float32)
    seq_count = len(cu_seqlens) - 1
    for idx in range(seq_count):
        start = int(cu_seqlens[idx].item())
        end = int(cu_seqlens[idx + 1].item())
        if end <= start:
            continue
        dk_seq, dv_seq, dbeta_seq, dg_seq, dA_seq = _prepare_wy_repr_bwd_pytorch(
            k[:, start:end, :, :],
            v[:, start:end, :, :],
            beta[:, start:end, :],
            gk[:, start:end, :, :],
            A[:, :, start:end, :],
            dw[:, start:end, :, :],
            du[:, start:end, :, :],
        )
        dk[:, start:end, :, :] = dk_seq
        dv[:, start:end, :, :] = dv_seq
        dbeta[:, start:end, :] = dbeta_seq
        dg[:, start:end, :, :] = dg_seq
        dA[:, :, start:end, :] = dA_seq
    return dk, dv, dbeta, dg, dA


# ============================================================================
# Internal PyTorch kernel implementations (Stage 2.2.1 & 2.2.3)
# Converted from Triton kernels in libs/fla/ops/kda/wy_fast.py
# ============================================================================


def _recompute_w_u_fwd_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    store_qg: bool = False,
    store_kg: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    PyTorch implementation of recompute_w_u_fwd_kernel.
    
    Source: libs/fla/ops/kda/wy_fast.py:L29-L103
    
    Computes WY representation: w and u vectors from A matrix.
    
    Args:
        q: queries [B, T, H, K]
        k: keys [B, T, H, K]
        v: values [B, T, H, V]
        beta: beta weights [B, T, H]
        A: solved triangular matrix [B, H, T, BT] (from solve_tril)
        gk: cumulative gate [B, T, H, K]
        store_qg: whether to compute and return qg = q * exp(gk)
        store_kg: whether to compute and return kg = k * exp(gn - gk)
    
    Returns:
        w: [B, T, H, K] - modified keys
        u: [B, T, H, V] - modified values
        qg: [B, T, H, K] or None - gated queries (if store_qg=True)
        kg: [B, T, H, K] or None - gated keys (if store_kg=True)
    
    Key logic:
    - u = A @ (v * beta)
    - w = A @ (k * beta * exp(gk))
    - qg = q * exp(gk) (optional)
    - kg = k * exp(gn - gk) (optional, gn = gk[last_token])
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = 64  # chunk size
    
    # Initialize outputs
    w = torch.zeros_like(k)
    u = torch.zeros_like(v)
    qg = torch.zeros_like(q) if store_qg else None
    kg = torch.zeros_like(k) if store_kg else None
    
    # Process each chunk
    NT = (T + BT - 1) // BT
    for i_t in range(NT):
        t_start = i_t * BT
        t_end = min(t_start + BT, T)
        chunk_len = t_end - t_start
        
        for b in range(B):
            for h in range(H):
                # Load A matrix for this chunk [chunk_len, BT]
                # Note: A is [B, H, T, BT], we need [t_start:t_end, :]
                A_chunk = A[b, h, t_start:t_end, :BT]  # [chunk_len, BT]
                
                # Load beta [chunk_len]
                beta_chunk = beta[b, t_start:t_end, h]  # [chunk_len]
                
                # Compute u = A @ (v * beta)
                v_chunk = v[b, t_start:t_end, h, :]  # [chunk_len, V]
                vb = v_chunk * beta_chunk.unsqueeze(-1)  # [chunk_len, V]
                # Pad vb to BT if needed
                vb_padded = torch.zeros(BT, V, device=v.device, dtype=torch.float32)
                vb_padded[:chunk_len, :] = vb.to(torch.float32)
                result_u = torch.matmul(A_chunk.to(torch.float32), vb_padded)[:chunk_len, :]
                u[b, t_start:t_end, h, :] = result_u.to(v.dtype)
                
                # Compute w = A @ (k * beta * exp(gk))
                k_chunk = k[b, t_start:t_end, h, :]  # [chunk_len, K]
                gk_chunk = gk[b, t_start:t_end, h, :]  # [chunk_len, K]
                kb = k_chunk * beta_chunk.unsqueeze(-1) * torch.exp(gk_chunk)  # [chunk_len, K]
                # Pad kb to BT if needed
                kb_padded = torch.zeros(BT, K, device=k.device, dtype=torch.float32)
                kb_padded[:chunk_len, :] = kb.to(torch.float32)
                result_w = torch.matmul(A_chunk.to(torch.float32), kb_padded)[:chunk_len, :]
                w[b, t_start:t_end, h, :] = result_w.to(k.dtype)
                
                # Optionally compute qg = q * exp(gk)
                if store_qg:
                    q_chunk = q[b, t_start:t_end, h, :]  # [chunk_len, K]
                    qg[b, t_start:t_end, h, :] = q_chunk * torch.exp(gk_chunk)
                
                # Optionally compute kg = k * exp(gn - gk)
                if store_kg:
                    # gn = gk[last_token]
                    last_idx = t_end - 1
                    gn = gk[b, last_idx, h, :]  # [K]
                    kg[b, t_start:t_end, h, :] = k_chunk * torch.exp(gn.unsqueeze(0) - gk_chunk)
    
    return w, u, qg, kg
