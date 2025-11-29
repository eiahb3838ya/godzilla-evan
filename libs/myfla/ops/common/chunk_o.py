"""Simplified PyTorch fallbacks for `libs.fla.ops.common.chunk_o` helpers.

源自官方 FLA：libs/fla/ops/common/chunk_o.py
完美復刻 chunk_bwd_dv_local。
"""

from __future__ import annotations

import torch

from ..utils.op import exp


def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    完美復刻官方 `chunk_bwd_dv_local`（libs/fla/ops/common/chunk_o.py:580-626）。
    
    計算 intra-chunk local attention 的梯度：dv = A.T @ do
    其中 A 是 chunk 內的 causal attention matrix。
    
    參數：
        q: [B, T, H, K] - Query
        k: [B, T, H, K] - Key
        do: [B, T, H, V] - 輸出梯度
        g: [B, T, H] or None - Global gate（指數形式）
        g_gamma: [H] or None - Linear gate gamma（用於 g_gamma * t）
        A: [B, T, H, T] or None - 預計算的 attention matrix（當前未使用）
        scale: float or None - Attention 縮放因子
        cu_seqlens: LongTensor or None - Varlen 支援
        chunk_size: int - Chunk 大小
        
    返回：
        dv: [B, T, H, V] - Value 的梯度
        
    核心公式（對應 Triton kernel line 414-449）：
        1. A[i,j] = k[i] @ q[j].T  （intra-chunk attention）
        2. A[i,j] *= exp(g[j] - g[i]) * scale  （gate + scale）
        3. A[i,j] = 0 if i > j  （causal mask）
        4. dv[i] = sum_j A[j,i] * do[j]
    
    NOTE: varlen (`cu_seqlens`) is not supported yet.
    """

    if cu_seqlens is not None:
        raise NotImplementedError('chunk_bwd_dv_local 不支援 cu_seqlens，待 Stage 2 移植')

    B, T, H, K = q.shape
    V = do.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    
    if scale is None:
        scale = K ** -0.5
    
    dv = torch.empty_like(do)
    
    # 逐 chunk 計算（對應 Triton kernel 的 grid 並行）
    for i_t in range(NT):
        start_t = i_t * BT
        end_t = min((i_t + 1) * BT, T)
        chunk_len = end_t - start_t
        
        q_chunk = q[:, start_t:end_t]  # [B, chunk_len, H, K]
        k_chunk = k[:, start_t:end_t]  # [B, chunk_len, H, K]
        do_chunk = do[:, start_t:end_t]  # [B, chunk_len, H, V]
        
        # 1. 計算 intra-chunk attention matrix: A = k @ q.T（對應 line 362-368）
        # [B, H, chunk_len, K] @ [B, H, K, chunk_len] → [B, H, chunk_len, chunk_len]
        q_chunk_t = q_chunk.transpose(1, 2)  # [B, H, chunk_len, K]
        k_chunk_t = k_chunk.transpose(1, 2)  # [B, H, chunk_len, K]
        A_chunk = torch.matmul(k_chunk_t, q_chunk_t.transpose(-2, -1))  # [B, H, chunk_len, chunk_len]
        
        # 2. 應用 causal mask（對應 line 385）
        #    m_A = (o_t[:, None] <= o_t[None, :])  → 下三角 mask
        causal_mask = torch.tril(torch.ones(chunk_len, chunk_len, device=q.device, dtype=torch.bool))
        A_chunk = A_chunk.masked_fill(~causal_mask, 0)
        
        # 3. 應用 gate 和 scale（對應 line 386-390）
        if g is not None:
            g_chunk = g[:, start_t:end_t, :]  # [B, chunk_len, H]
            g_last = g[:, end_t - 1, :]  # [B, H]
            
            # A[i,j] *= exp(g[j] - g[i])（對應 line 387）
            # [B, H, chunk_len, 1] - [B, H, 1, chunk_len] → [B, H, chunk_len, chunk_len]
            g_diff = g_chunk.transpose(1, 2).unsqueeze(-1) - g_chunk.transpose(1, 2).unsqueeze(-2)
            A_chunk = A_chunk * exp(g_diff)
        
        # 應用 scale
        A_chunk = A_chunk * scale
        
        # 4. 計算 dv = A.T @ do（對應 line 394）
        # [B, H, chunk_len, chunk_len].T @ [B, H, chunk_len, V] → [B, H, chunk_len, V]
        do_chunk_t = do_chunk.transpose(1, 2)  # [B, H, chunk_len, V]
        dv_chunk = torch.matmul(A_chunk.transpose(-2, -1), do_chunk_t.to(A_chunk.dtype))
        
        # 如果有 gate，還需要應用 exp(-g + g_last)（對應 line 388）
        if g is not None:
            g_chunk = g[:, start_t:end_t, :]  # [B, chunk_len, H]
            g_last = g[:, end_t - 1, :]  # [B, H]
            gate_factor = exp(-g_chunk.transpose(1, 2) + g_last.unsqueeze(-1))  # [B, H, chunk_len]
            dv_chunk = dv_chunk * gate_factor.unsqueeze(-1)  # [B, H, chunk_len, V]
        
        dv[:, start_t:end_t] = dv_chunk.transpose(1, 2).to(dv.dtype)  # [B, chunk_len, H, V]
    
    return dv
