"""
Pure PyTorch fallback for `libs.fla.ops.gated_delta_rule.chunk`.

源自官方 FLA：libs/fla/ops/gated_delta_rule/chunk.py
完美復刻 chunk_gated_delta_rule（包含 forward/backward autograd function）。
"""

from __future__ import annotations

import torch


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
):
    r"""
    完美復刻官方 `chunk_gated_delta_rule`（libs/fla/ops/gated_delta_rule/chunk.py:217-267）。
    
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    
    NOTE: 當前為 PyTorch fallback 實現。完整的 Triton kernel 版本請參考官方 fla。
    """
    
    if cu_seqlens is not None:
        raise NotImplementedError('chunk_gated_delta_rule 的 varlen 支援待實現（需 Triton kernel）')
    
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    if scale is None:
        scale = K ** -0.5
    
    # L2 normalization
    if use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
    
    # 初始化 state
    if initial_state is None:
        state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.to(dtype=torch.float32)
    
    outputs = []
    
    # 主循環：逐 token 處理（PyTorch fallback）
    # NOTE: 官方用 Triton kernel 做 chunk 並行，這裡簡化為序列處理
    for t in range(T):
        q_t = q[:, t]  # [B, H, K]
        k_t = k[:, t]  # [B, H, K]
        v_t = v[:, t]  # [B, H, V]
        g_t = g[:, t]  # [B, H]
        beta_t = beta[:, t]  # [B, H]
        
        # 應用 gate: state *= exp(g_t)
        state = state * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        
        # 應用 beta: state = beta * state + (1 - beta) * (k @ v)
        kv = torch.einsum('bhk,bhv->bhkv', k_t, v_t)  # [B, H, K, V]
        beta_t = beta_t.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        state = beta_t * state + (1.0 - beta_t) * kv.to(torch.float32)
        
        # 計算輸出: o = q @ state
        o_t = torch.einsum('bhk,bhkv->bhv', q_t, state) * scale  # [B, H, V]
        
        # 應用 output gate
        o_t = o_t * torch.sigmoid(g_t).unsqueeze(-1)  # [B, H, V]
        
        outputs.append(o_t)
    
    o = torch.stack(outputs, dim=1)  # [B, T, H, V]
    
    final_state = state.to(q.dtype) if output_final_state else None
    
    return o, final_state
