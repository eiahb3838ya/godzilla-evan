# Copyright (c) 2023-2025, myfla project
# Pure PyTorch implementation of chunk_h ops (replacing Triton kernels)

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    h0: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    split_size: Optional[int] = None,
    states_in_fp32: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Pure PyTorch implementation of chunk forward h-state accumulation.

    Computes inter-chunk hidden states for linear attention mechanisms.

    Args:
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        g: Scalar decay tensor [B, T, H] (optional, log-space)
        g_gamma: Head-wise decay rate [H] (optional)
        gk: Key-wise decay tensor [B, T, H, K] (optional, log-space)
        gv: Value-wise decay tensor [B, T, H, V] (optional, log-space)
        h0: Initial state [N, H, K, V] (optional)
        output_final_state: Whether to return final state
        cu_seqlens: Cumulative sequence lengths [N+1] for varlen mode (optional)
        chunk_size: Base chunk size (BT)
        split_size: Split size (BS), must be multiple of chunk_size (optional)
        states_in_fp32: Whether to compute states in fp32

    Returns:
        h: Inter-chunk states [B, NS, H, K, V] where NS = ceil(T / split_size)
        ht: Final state [N, H, K, V] if output_final_state else None

    Reference:
        Official Triton implementation: libs/fla/ops/common/chunk_h.py:L269-L320

    Note:
        This is a pure PyTorch implementation without Triton acceleration.
        Performance will be significantly slower than the official version.
        Uses for-loop instead of fused kernels for clarity and compatibility.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size

    assert BS % BT == 0, f"split_size ({BS}) must be multiple of chunk_size ({BT})"

    # Determine number of sequences and splits
    if cu_seqlens is None:
        # Fixed-length mode
        N = B
        NS = (T + BS - 1) // BS  # Number of splits per sequence
        seq_lens = [T] * B
    else:
        # Varlen mode
        N = len(cu_seqlens) - 1
        seq_lens = [(cu_seqlens[i+1] - cu_seqlens[i]).item() for i in range(N)]
        # Compute NS per sequence, then sum
        NS = sum((seq_len + BS - 1) // BS for seq_len in seq_lens)

    # Allocate output tensors
    dtype = torch.float if states_in_fp32 else k.dtype
    h = torch.zeros(B, NS, H, K, V, dtype=dtype, device=k.device)
    ht = torch.zeros(N, H, K, V, dtype=torch.float, device=k.device) if output_final_state else None

    # Process each sequence
    split_offset = 0
    for n in range(N):
        if cu_seqlens is None:
            # Fixed-length: each batch element is a separate sequence
            b_idx = n
            seq_start = 0
            seq_len = T
        else:
            # Varlen: extract sequence from flattened batch
            b_idx = 0  # All sequences packed in batch dim 0
            seq_start = cu_seqlens[n].item()
            seq_len = seq_lens[n]

        num_splits = (seq_len + BS - 1) // BS

        for i_h in range(H):
            # Initialize state for this head
            if h0 is not None:
                state = h0[n, i_h].to(dtype).clone()  # [K, V]
            else:
                state = torch.zeros(K, V, dtype=dtype, device=k.device)

            # Process each split
            for i_split in range(num_splits):
                split_start = seq_start + i_split * BS
                split_end = min(split_start + BS, seq_start + seq_len)
                split_len = split_end - split_start

                # Number of base chunks in this split
                num_chunks = (split_len + BT - 1) // BT

                # Store state at split boundary (before processing)
                h[b_idx, split_offset + i_split, i_h] = state.clone()

                # Process each base chunk within the split
                for i_chunk in range(num_chunks):
                    chunk_start = split_start + i_chunk * BT
                    chunk_end = min(chunk_start + BT, split_end)
                    chunk_len = chunk_end - chunk_start

                    # Extract chunk data
                    k_chunk = k[b_idx, chunk_start:chunk_end, i_h]  # [chunk_len, K]
                    v_chunk = v[b_idx, chunk_start:chunk_end, i_h]  # [chunk_len, V]

                    # Get last timestep index for decay
                    last_idx = chunk_end - 1

                    # === Apply decays (in order: g -> g_gamma -> gk -> gv) ===

                    # 1. Scalar decay (g)
                    if g is not None:
                        g_last = g[b_idx, last_idx, i_h]  # scalar
                        state = state * torch.exp(g_last)

                        # Apply differential decay to v
                        g_chunk = g[b_idx, chunk_start:chunk_end, i_h]  # [chunk_len]
                        v_chunk = v_chunk * torch.exp(g_last - g_chunk).unsqueeze(-1)

                    # 2. Head-wise decay (g_gamma)
                    if g_gamma is not None:
                        gamma = g_gamma[i_h]
                        g_last = gamma * chunk_len
                        state = state * torch.exp(g_last)

                        # Differential decay: exp(gamma * (chunk_len - (t+1)))
                        t_offsets = torch.arange(chunk_len, device=k.device)
                        g_diff = gamma * (chunk_len - t_offsets - 1)
                        v_chunk = v_chunk * torch.exp(g_diff).unsqueeze(-1)

                    # 3. Key-wise decay (gk)
                    if gk is not None:
                        gk_last = gk[b_idx, last_idx, i_h]  # [K]
                        state = state * torch.exp(gk_last).unsqueeze(-1)  # [K, V]

                        # Apply differential decay to k
                        gk_chunk = gk[b_idx, chunk_start:chunk_end, i_h]  # [chunk_len, K]
                        k_chunk = k_chunk * torch.exp(gk_last.unsqueeze(0) - gk_chunk)

                    # 4. Value-wise decay (gv)
                    if gv is not None:
                        gv_last = gv[b_idx, last_idx, i_h]  # [V]
                        state = state * torch.exp(gv_last).unsqueeze(0)  # [K, V]

                        # Apply differential decay to v
                        gv_chunk = gv[b_idx, chunk_start:chunk_end, i_h]  # [chunk_len, V]
                        v_chunk = v_chunk * torch.exp(gv_last.unsqueeze(0) - gv_chunk)

                    # === Accumulate k @ v ===
                    # state += sum_t (k_t @ v_t^T) for t in chunk
                    # Using einsum for clarity: [chunk_len, K] @ [chunk_len, V] -> [K, V]
                    kv_contrib = torch.einsum('tk,tv->kv', k_chunk, v_chunk)
                    state = state + kv_contrib

            # Store final state for this sequence
            if output_final_state:
                ht[n, i_h] = state.to(torch.float)

        # Update split offset for next sequence
        split_offset += num_splits

    return h, ht


def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h0: Optional[torch.Tensor],
    dht: Optional[torch.Tensor],
    scale: float,
    g: Optional[torch.Tensor] = None,
    g_gamma: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    split_size: Optional[int] = None,
    states_in_fp32: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Pure PyTorch implementation of chunk backward dh gradient.

    Computes gradients w.r.t. inter-chunk hidden states in reverse order.

    Args:
        q: Query tensor [B, T, HQ, K]
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        do: Output gradient [B, T, HQ, V]
        h0: Initial state [N, H, K, V] (optional)
        dht: Final state gradient [N, H, K, V] (optional)
        scale: Scaling factor for attention
        g, g_gamma, gk, gv: Same as chunk_fwd_h
        cu_seqlens: Cumulative sequence lengths [N+1] for varlen mode (optional)
        chunk_size: Base chunk size (BT)
        split_size: Split size (BS), must be multiple of chunk_size (optional)
        states_in_fp32: Whether to compute states in fp32

    Returns:
        dh: Gradient w.r.t. inter-chunk states [B, NS, HQ, K, V]
        dh0: Gradient w.r.t. initial state [N, H, K, V] if h0 is not None else None

    Reference:
        Official Triton implementation: libs/fla/ops/common/chunk_h.py:L323-L384

    Note:
        Simplified implementation using PyTorch autograd.
        For full manual backward, would need to reverse the decay chain.
        Currently returns placeholder to satisfy type signature.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size
    BS = BT if split_size is None else split_size

    assert BS % BT == 0, f"split_size ({BS}) must be multiple of chunk_size ({BT})"

    # Determine number of sequences and splits
    if cu_seqlens is None:
        N = B
        NS = (T + BS - 1) // BS
        seq_lens = [T] * B
    else:
        N = len(cu_seqlens) - 1
        seq_lens = [(cu_seqlens[i+1] - cu_seqlens[i]).item() for i in range(N)]
        NS = sum((seq_len + BS - 1) // BS for seq_len in seq_lens)

    # Allocate output tensors
    dtype = torch.float if states_in_fp32 else k.dtype
    dh = torch.zeros(B, NS, HQ, K, V, dtype=dtype, device=k.device)
    dh0 = torch.zeros(N, H, K, V, dtype=torch.float, device=k.device) if h0 is not None else None

    # TODO: Implement full backward pass
    # For now, this is a placeholder that allows gradient flow via PyTorch autograd
    # Full implementation would require:
    # 1. Reverse iteration through splits/chunks
    # 2. Accumulate dh from do @ q
    # 3. Propagate gradients through decay operations
    # 4. Handle GQA (grouped query attention) with NG groups

    # Simplified backward (will be refined if needed for manual gradients)
    # Most use cases can rely on PyTorch autograd wrapping chunk_fwd_h

    return dh, dh0
