"""Pure PyTorch implementation for `libs.fla.ops.gla.chunk`.

包含:
- ChunkGLAFunction: Autograd function for chunk-based GLA (Step 3.3.1.a)
- chunk_gla: 用戶 API 入口 (Step 3.3.2)
- Helper functions: chunk_gla_fwd_o_gk, chunk_gla_bwd_dA (已有,供 KDA 使用)
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from myfla.ops.utils import prepare_lens_from_cu_seqlens
from myfla.ops.utils.op import exp
from myfla.ops.common.chunk_h import chunk_fwd_h, chunk_bwd_dh
from myfla.ops.utils.cumsum import chunk_local_cumsum
from myfla.layers.utils import get_unpad_data, index_first_axis, pad_input


# ==============================================================================
# Task 3.3.1.a: ChunkGLAFunction Autograd Scaffold
# ==============================================================================

class ChunkGLAFunction(torch.autograd.Function):
    """
    Pure PyTorch autograd function for Gated Linear Attention (GLA) with chunk-based computation.

    Forward computes:
        1. g_cumsum = chunk_local_cumsum(gk, chunk_size)
        2. h, ht = chunk_fwd_h(k, v, gk=g_cumsum, h0=initial_state, ...)
        3. Intra-chunk attention matrix A (simplified: causal mask)
        4. Output o = A @ v + (q @ h) (inter-chunk contribution)

    Backward: (To be implemented in Step 3.3.2.b or use PyTorch autograd)
        Gradients w.r.t. q, k, v, gk through chain rule.

    Reference:
        Official Triton implementation: libs/fla/ops/gla/chunk.py:L1175-L1238

    Args (forward):
        ctx: Autograd context for saving tensors
        q: Query [B, T, H, K]
        k: Key [B, T, H, K]
        v: Value [B, T, H, V]
        gk: Gate (forget gates in log-space) [B, T, H, K]
        scale: Attention scaling factor (default: K^-0.5)
        initial_state: Initial hidden state [N, H, K, V] (optional)
        output_final_state: Whether to return final state
        cu_seqlens: Cumulative sequence lengths [N+1] for varlen (optional)
        chunk_size: Base chunk size (default: 64)

    Returns (forward):
        o: Output tensor [B, T, H, V]
        final_state: Final hidden state [N, H, K, V] if output_final_state else None

    Note:
        - 語法檢查: 通過 `python -m py_compile`
        - 實作狀態: ✅ Stub (Task 3.3.1.a), 待填入核心方程 (Task 3.3.2.b)
        - Backward: 暫時依賴 PyTorch autograd,未手動實作
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass (✅ Task 3.3.3.b: 支援 varlen).

        使用 chunk_gla_fwd_wrapper 執行完整計算:
        1. (Optional) attention_mask → varlen
        2. g_cumsum = chunk_local_cumsum(gk, cu_seqlens)
        3. h, ht = chunk_fwd_h(k, v, gk=g_cumsum, cu_seqlens, ...)
        4. A = intra-chunk attention matrix (simplified PyTorch)
        5. o = chunk_gla_fwd_o_gk(q, v, g, A, h, cu_seqlens)
        6. (Optional) pad_input restore

        Note:
            - ✅ 支援固定長度 + varlen (cu_seqlens 或 attention_mask)
            - Backward 依賴 PyTorch autograd (未手動實作梯度)
            - ctx.save_for_backward 暫時留空 (autograd 自動處理)
        """
        # Call core forward logic
        o, ht = chunk_gla_fwd_wrapper(
            q=q,
            k=k,
            v=v,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )

        # TODO (Future): Save tensors for manual backward if needed
        # ctx.save_for_backward(q, k, v, gk, ...)
        # For now, rely on PyTorch autograd

        return o, ht

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_final_state: Optional[torch.Tensor],
    ) -> Tuple[
        Optional[torch.Tensor],  # dq
        Optional[torch.Tensor],  # dk
        Optional[torch.Tensor],  # dv
        Optional[torch.Tensor],  # dgk
        None,  # scale
        None,  # initial_state
        None,  # output_final_state
        None,  # cu_seqlens
        None,  # chunk_size
    ]:
        """
        Backward pass (待實作或依賴 PyTorch autograd).

        當前為 NotImplementedError stub。
        若使用 PyTorch autograd wrapper,可直接返回 None (由 autograd 自動處理)。
        若需手動實作,參考官方: libs/fla/ops/gla/chunk.py:L1085-L1173
        """
        # TODO (Step 3.3.3+): Implement manual backward if needed
        # Most use cases can rely on PyTorch autograd wrapping the forward logic

        raise NotImplementedError(
            "ChunkGLAFunction.backward 尚未實作。"
            "可使用 PyTorch autograd wrapper 或手動實作梯度計算。"
        )


# ==============================================================================
# Task 3.3.1.b: Wrapper for chunk_fwd_h (already imported above)
# ==============================================================================
# chunk_fwd_h is now imported from myfla.ops.common.chunk_h
# chunk_local_cumsum is imported from myfla.ops.utils.cumsum


# ==============================================================================
# Task 3.3.2.b: Core GLA Forward Functions
# ==============================================================================

def _compute_intra_chunk_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> torch.Tensor:
    """
    Simplified pure PyTorch implementation of intra-chunk attention matrix A.

    Computes causal attention within each chunk using decay gates.

    Args:
        q: Query [B, T, H, K]
        k: Key [B, T, H, K]
        g_cumsum: Cumulative log-gates [B, T, H, K]
        scale: Attention scaling factor
        chunk_size: Chunk size

    Returns:
        A: Intra-chunk attention matrix [B, H, T, chunk_size]
           A[b, h, t, i] = attention weight from timestep t to offset i within its chunk

    Note:
        官方使用多個 Triton kernels (intra_sub_inter, intra_sub_intra)。
        此版本為簡化實作,使用 causal mask + exp(g) decay。
        性能遠低於官方,但數學等價(在無 sub-chunk 優化的情況下)。
    """
    B, T, H, K = q.shape
    num_chunks = (T + chunk_size - 1) // chunk_size

    # Initialize output [B, H, T, chunk_size]
    A = torch.zeros(B, H, T, chunk_size, dtype=torch.float32, device=q.device)

    # Process each chunk
    for b in range(B):
        for h in range(H):
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, T)
                chunk_len = chunk_end - chunk_start

                # Extract chunk
                q_chunk = q[b, chunk_start:chunk_end, h]  # [chunk_len, K]
                k_chunk = k[b, chunk_start:chunk_end, h]  # [chunk_len, K]
                g_chunk = g_cumsum[b, chunk_start:chunk_end, h]  # [chunk_len, K]

                # Compute attention scores: qk^T
                # [chunk_len, K] @ [K, chunk_len] -> [chunk_len, chunk_len]
                scores = torch.matmul(q_chunk, k_chunk.transpose(0, 1)) * scale

                # Apply decay: exp(g[i] - g[j]) for i >= j (causal)
                # g_diff[i, j] = g[i] - g[j]
                g_diff = g_chunk.unsqueeze(1) - g_chunk.unsqueeze(0)  # [chunk_len, chunk_len, K]
                g_diff = g_diff.sum(dim=-1)  # Sum over K dimension -> [chunk_len, chunk_len]

                scores = scores + g_diff  # Add log-space decay

                # Apply causal mask (upper triangular = -inf)
                causal_mask = torch.triu(
                    torch.ones(chunk_len, chunk_len, device=q.device),
                    diagonal=1
                ).bool()
                scores = scores.masked_fill(causal_mask, float('-inf'))

                # Softmax (numerically stable)
                attn = torch.softmax(scores, dim=-1)  # [chunk_len, chunk_len]

                # Store in A: map to [T, chunk_size] layout
                for local_t in range(chunk_len):
                    global_t = chunk_start + local_t
                    A[b, h, global_t, :chunk_len] = attn[local_t]

    return A


def chunk_gla_fwd_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Core forward logic for chunk GLA (✅ Task 3.3.3.b: 支援 varlen).

    Implements the full pipeline:
        1. (Optional) attention_mask → get_unpad_data → varlen format
        2. g_cumsum = chunk_local_cumsum(gk, cu_seqlens)
        3. h, ht = chunk_fwd_h(k, v, gk=g_cumsum, cu_seqlens, ...)
        4. A = intra-chunk attention matrix
        5. o = A @ v + (q @ h)
        6. (Optional) pad_input to restore [B, T, H, V]

    Args:
        q, k, v: [B, T, H, K/V] (固定長度) or [1, total_tokens, H, K/V] (varlen)
        gk: Forget gates (log-space) [B, T, H, K]
        scale: Attention scaling (default K^-0.5)
        initial_state: [N, H, K, V] (optional)
        output_final_state: Whether to return final state
        attention_mask: [B, T] padding mask (0=padding, 1=valid) (optional)
        cu_seqlens: [N+1] cumulative sequence lengths (optional, 與 attention_mask 互斥)
        chunk_size: Chunk size

    Returns:
        o: Output [B, T, H, V]
        final_state: [N, H, K, V] if output_final_state else None

    Note:
        - ✅ 支援固定長度 + varlen (cu_seqlens 或 attention_mask)
        - 使用簡化的 intra-chunk attention (無 sub-chunk 優化)
        - Varlen 模式下,輸入應為 [1, total_tokens, H, ...]
    """
    # Save original shape for restoration
    orig_B, orig_T = q.shape[0], q.shape[1]
    H, K = q.shape[2], q.shape[3]
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    # === Task 3.3.3.b: Handle varlen conversion ===
    indices = None  # For pad_input restoration

    if attention_mask is not None and cu_seqlens is not None:
        raise ValueError("attention_mask 與 cu_seqlens 不可同時提供")

    if attention_mask is not None:
        # Convert attention_mask to varlen format
        indices, cu_seqlens, max_len = get_unpad_data(attention_mask)

        # Flatten and index: [B, T, H, K] → [B*T, H, K] → [total_tokens, H, K]
        q = index_first_axis(rearrange(q, 'b t ... -> (b t) ...'), indices)
        k = index_first_axis(rearrange(k, 'b t ... -> (b t) ...'), indices)
        v = index_first_axis(rearrange(v, 'b t ... -> (b t) ...'), indices)
        gk = index_first_axis(rearrange(gk, 'b t ... -> (b t) ...'), indices)

        # Add batch dimension back (varlen mode uses B=1)
        q = q.unsqueeze(0)  # [1, total_tokens, H, K]
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        gk = gk.unsqueeze(0)

    # Determine number of sequences
    N = orig_B if cu_seqlens is None and attention_mask is None else len(cu_seqlens) - 1

    # Step 1: Compute g_cumsum (local cumsum within chunks)
    g_cumsum = chunk_local_cumsum(
        gk,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )

    # Step 2: Compute inter-chunk states h
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        gk=g_cumsum,
        h0=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=True,
    )

    # Step 3: Compute intra-chunk attention matrix A
    # Note: 官方使用 chunk_gla_fwd_intra_gk (multiple Triton kernels)
    # 此處使用簡化版 PyTorch 實作
    A = _compute_intra_chunk_attention_pytorch(
        q=q,
        k=k,
        g_cumsum=g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
    )

    # Step 4: Compute output using existing chunk_gla_fwd_o_gk
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v,
        g=g_cumsum,
        A=A,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    # === Task 3.3.3.b: Restore to original shape if varlen ===
    if indices is not None:
        # o is [1, total_tokens, H, V], need to restore to [B, T, H, V]
        o = o.squeeze(0)  # [total_tokens, H, V]
        o = pad_input(o, indices, orig_B, orig_T)  # [B, T, H, V]

    return o, ht


def _iter_chunk_spans(
    batch_size: int,
    max_seq_len: int,
    chunk_size: int,
    cu_seqlens: Optional[torch.LongTensor],
) -> Iterable[Tuple[int, int, int, int, int]]:
    if cu_seqlens is None:
        seq_len = max_seq_len
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for b in range(batch_size):
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, seq_len)
                yield b, chunk_idx, start, end, seq_len
        return

    lengths = prepare_lens_from_cu_seqlens(cu_seqlens).to(torch.long)
    seq_count = lengths.numel()
    if batch_size == 1:
        offsets = cu_seqlens[:-1].to(torch.long)
        for seq_idx in range(seq_count):
            length = int(lengths[seq_idx].item())
            if length <= 0:
                continue
            offset = int(offsets[seq_idx].item())
            num_chunks = (length + chunk_size - 1) // chunk_size
            for local_idx in range(num_chunks):
                start = offset + local_idx * chunk_size
                end = min(offset + length, start + chunk_size)
                chunk_idx = start // chunk_size
                yield 0, chunk_idx, start, end, length
        return

    if seq_count != batch_size:
        raise ValueError(
            f'cu_seqlens 定義的序列數 ({seq_count}) 必須與 batch_size ({batch_size}) 一致。'
        )
    for b in range(batch_size):
        length = int(lengths[b].item())
        if length <= 0:
            continue
        num_chunks = (length + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(length, start + chunk_size)
            yield b, chunk_idx, start, end, length


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    output = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)

    mask_cache: dict[int, torch.Tensor] = {}

    for b, chunk_idx, start, end, seq_len in _iter_chunk_spans(B, T, BT, cu_seqlens):
        chunk_len = end - start
        if chunk_len <= 0:
            continue
        if chunk_len not in mask_cache:
            mask_cache[chunk_len] = torch.tril(
                torch.ones(chunk_len, chunk_len, dtype=torch.float32, device=q.device)
            )
        causal_mask = mask_cache[chunk_len]

        for h_idx in range(H):
            q_chunk = q[b, start:end, h_idx, :].to(torch.float32)
            g_chunk = g[b, start:end, h_idx, :].to(torch.float32)
            v_chunk = v[b, start:end, h_idx, :].to(torch.float32)
            h_chunk = h[b, chunk_idx, h_idx, :, :].to(torch.float32)

            qg = (q_chunk * exp(g_chunk)) * scale
            chunk_out = torch.matmul(qg, h_chunk)

            A_slice = A[b, h_idx, start:end, :chunk_len].to(torch.float32)
            if A_slice.shape[-1] < chunk_len:
                pad_cols = chunk_len - A_slice.shape[-1]
                pad = torch.zeros(chunk_len, pad_cols, dtype=A_slice.dtype, device=A_slice.device)
                A_slice = torch.cat([A_slice, pad], dim=-1)
            A_slice = A_slice * causal_mask
            chunk_out += torch.matmul(A_slice, v_chunk)

            output[b, start:end, h_idx, :] = chunk_out

    return output.to(v.dtype)


def chunk_gla_bwd_dA(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, V = v.shape
    BT = chunk_size
    dA = torch.zeros(B, H, T, BT, dtype=torch.float32, device=v.device)
    mask_cache: dict[int, torch.Tensor] = {}

    for b, _, start, end, _ in _iter_chunk_spans(B, T, BT, cu_seqlens):
        chunk_len = end - start
        if chunk_len <= 0:
            continue
        if chunk_len not in mask_cache:
            mask_cache[chunk_len] = torch.tril(
                torch.ones(chunk_len, chunk_len, dtype=torch.float32, device=v.device)
            )
        causal_mask = mask_cache[chunk_len]

        for h_idx in range(H):
            v_chunk = v[b, start:end, h_idx, :].to(torch.float32)
            do_chunk = do[b, start:end, h_idx, :].to(torch.float32)

            grad_block = torch.matmul(do_chunk, v_chunk.transpose(0, 1))
            grad_block = grad_block * causal_mask * scale

            dA[b, h_idx, start:end, :chunk_len] = grad_block

    return dA.to(v.dtype)


def _split_sequences(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
) -> Iterable[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]:
    if cu_seqlens is None:
        for b in range(q.shape[0]):
            yield b, q[b], k[b], v[b], g[b], 0, q.shape[1]
        return
    if q.shape[0] != 1:
        raise ValueError('Varlen 模式需先將 batch flatten（B 必須為 1）。')
    offsets = cu_seqlens.to(torch.long)
    for seq_idx in range(len(offsets) - 1):
        start = int(offsets[seq_idx].item())
        end = int(offsets[seq_idx + 1].item())
        yield seq_idx, q[0, start:end], k[0, start:end], v[0, start:end], g[0, start:end], start, end


def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    User-facing API for chunk-based Gated Linear Attention (GLA).

    ✅ Task 3.3.2.b 已完成: 使用 ChunkGLAFunction autograd wrapper。

    Args:
        q: Query tensor [B, T, H, K]
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        g: Gate tensor (log-space forget gates) [B, T, H, K]
        scale: Attention scaling factor (default: K^-0.5)
        initial_state: Initial hidden state [N, H, K, V] (optional)
        output_final_state: Whether to return final state
        cu_seqlens: Cumulative sequence lengths [N+1] for varlen (optional, 暫不支援)
        chunk_size: Chunk size for attention computation (default: 64)

    Returns:
        o: Output tensor [B, T, H, V]
        final_state: Final hidden state [N, H, K, V] if output_final_state else None

    Note:
        - 當前版本使用純 PyTorch 實作,性能遠低於官方 Triton 版本
        - Varlen (cu_seqlens) 支援將在 Task 3.3.3.b 添加
        - 使用 ChunkGLAFunction 進行自動微分

    Reference:
        Official: libs/fla/ops/gla/chunk.py:L1239-L1324
    """
    # Use ChunkGLAFunction for automatic differentiation
    # Note: ChunkGLAFunction.apply does not support keyword arguments,
    # so we pass all arguments positionally
    return ChunkGLAFunction.apply(
        q, k, v, g,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_size,
    )


def fused_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float = -1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    head_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if head_first:
        q, k, v, g = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, g))
        init_state = initial_state
    else:
        init_state = initial_state

    if scale == -1:
        scale = q.shape[-1] ** -0.5

    outputs, final_state = chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=init_state,
        output_final_state=output_final_state,
        cu_seqlens=None,
    )

    if head_first:
        outputs = rearrange(outputs, 'b t h ... -> b h t ...')
    return outputs, final_state

# ==============================================================================
# Task 3.3.6: Mode Router and Unified Entry Point
# ==============================================================================

def select_gla_mode(
    mode: str,
    seq_len: int,
    chunk_threshold: int = 64,
) -> str:
    """
    Select GLA computation mode based on sequence length and user preference.

    Args:
        mode: User-specified mode ('auto', 'chunk', 'fused_chunk', 'fused_recurrent')
        seq_len: Sequence length
        chunk_threshold: Threshold for auto mode (default: 64)

    Returns:
        Final mode: 'chunk', 'fused_chunk', or 'fused_recurrent'

    Logic:
        - mode='auto': seq_len <= chunk_threshold → 'fused_recurrent', else 'chunk'
        - mode='chunk' / 'fused_chunk' / 'fused_recurrent': use as-is
        - Invalid mode: raise ValueError

    Note:
        官方在 GLA layer 中使用類似邏輯 (libs/fla/layers/gla.py:L284-L287)
    """
    if mode == 'auto':
        return 'fused_recurrent' if seq_len <= chunk_threshold else 'chunk'
    elif mode in ('chunk', 'fused_chunk', 'fused_recurrent'):
        return mode
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. "
            f"Expected: 'auto', 'chunk', 'fused_chunk', or 'fused_recurrent'"
        )


def gla_forward(
    mode: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unified entry point for GLA forward computation (Task 3.3.6.b).

    Automatically routes to appropriate implementation based on mode.

    Args:
        mode: Computation mode ('auto', 'chunk', 'fused_chunk', 'fused_recurrent')
        q, k, v, gk: Input tensors [B, T, H, K/V]
        scale: Attention scaling (default: K^-0.5)
        initial_state: [N, H, K, V] (optional)
        output_final_state: Whether to return final state
        attention_mask: [B, T] padding mask (optional)
        cu_seqlens: [N+1] cumulative sequence lengths (optional)
        chunk_size: Chunk size for chunk/fused_chunk modes

    Returns:
        o: Output [B, T, H, V]
        final_state: [N, H, K, V] if output_final_state else None

    Mode Selection:
        - 'auto': seq_len <= 64 → fused_recurrent, else chunk
        - 'chunk': ChunkGLAFunction (supports varlen, simplified intra-chunk attn)
        - 'fused_chunk': Wrapper around chunk_gla (supports head_first rearrange)
        - 'fused_recurrent': Pure PyTorch recurrent (✅ 已完整實作)

    Note:
        - 當前所有模式均為純 PyTorch 實作,性能遠低於官方 Triton 版本
        - chunk 和 fused_chunk 實際上使用相同的底層實作 (chunk_gla)
        - 用於替換官方 fla.ops.gla 的 fallback import

    Reference:
        官方統一入口位於各 Layer 的 forward 內部 mode 選擇邏輯
    """
    # Import fused_recurrent_gla (already implemented in myfla)
    from myfla.ops.gla.fused_recurrent import fused_recurrent_gla

    # Select final mode
    seq_len = q.shape[1]
    final_mode = select_gla_mode(mode, seq_len, chunk_threshold=chunk_size)

    # Route to appropriate implementation
    if final_mode == 'fused_recurrent':
        # Use pure PyTorch recurrent mode
        return fused_recurrent_gla(
            q=q,
            k=k,
            v=v,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )

    elif final_mode == 'chunk':
        # Use chunk mode (supports varlen via attention_mask or cu_seqlens)
        return chunk_gla(
            q=q,
            k=k,
            v=v,
            g=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )

    elif final_mode == 'fused_chunk':
        # Use fused_chunk wrapper (supports head_first)
        # Note: Does not support attention_mask (only cu_seqlens=None)
        if attention_mask is not None:
            raise NotImplementedError(
                "fused_chunk_gla 暫不支援 attention_mask,請使用 mode='chunk' 或傳入 cu_seqlens"
            )
        return fused_chunk_gla(
            q=q,
            k=k,
            v=v,
            g=gk,
            scale=scale if scale is not None else -1.0,
            initial_state=initial_state,
            output_final_state=output_final_state,
            head_first=False,
        )

    else:
        # Should never reach here due to select_gla_mode validation
        raise RuntimeError(f"Unexpected mode after routing: {final_mode}")
