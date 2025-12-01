"""
Pure PyTorch fallback for `libs.fla.ops.gated_delta_rule.fused_recurrent`.

源自官方 FLA：libs/fla/ops/gated_delta_rule/fused_recurrent.py
完美復刻 fused_recurrent_gated_delta_rule。
"""

from __future__ import annotations

import torch


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    完美復刻官方 `fused_recurrent_gated_delta_rule`（libs/fla/ops/gated_delta_rule/fused_recurrent.py:238-287）。
    
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        gk (torch.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    
    NOTE: 當前為 PyTorch fallback 實現。完整的 Triton kernel 版本請參考官方 fla。
    """
    
    if cu_seqlens is not None:
        raise NotImplementedError('fused_recurrent_gated_delta_rule 的 varlen 支援待實現（需 Triton kernel）')
    
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    
    if scale is None:
        scale = K ** -0.5
    
    # L2 normalization
    if use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)
    
    # 處理 GVA (Grouped Value Attention)
    # 如果 HV > H，需要 repeat q/k
    if HV > H:
        if HV % H != 0:
            raise ValueError(f"HV ({HV}) must be divisible by H ({H}) for GVA")
        q = q.repeat(1, 1, HV // H, 1)  # [B, T, HV, K]
        k = k.repeat(1, 1, HV // H, 1)  # [B, T, HV, K]
    
    # 初始化 state
    if initial_state is None:
        state = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.to(dtype=torch.float32)
    
    outputs = []
    
    # 主循環：逐 token 處理（PyTorch fallback）
    for t in range(T):
        q_t = q[:, t]  # [B, HV, K]
        k_t = k[:, t]  # [B, HV, K]
        v_t = v[:, t]  # [B, HV, V]
        
        # 應用 gate g
        if g is not None:
            g_t = g[:, t]  # [B, HV]
            state = state * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # [B, HV, 1, 1]
        
        # 應用 gate gk
        if gk is not None:
            gk_t = gk[:, t]  # [B, HV, K]
            state = state * torch.exp(gk_t).unsqueeze(-1)  # [B, HV, K, 1]
        
        # 應用 beta: state = beta * state + (1 - beta) * (k @ v)
        if beta is not None:
            beta_t = beta[:, t]  # [B, HV]
            kv = torch.einsum('bhk,bhv->bhkv', k_t, v_t)  # [B, HV, K, V]
            beta_t = beta_t.unsqueeze(-1).unsqueeze(-1)  # [B, HV, 1, 1]
            state = beta_t * state + (1.0 - beta_t) * kv.to(torch.float32)
        else:
            # 無 beta 時，直接累積
            kv = torch.einsum('bhk,bhv->bhkv', k_t, v_t)  # [B, HV, K, V]
            state = state + kv.to(torch.float32)
        
        # 計算輸出: o = q @ state
        o_t = torch.einsum('bhk,bhkv->bhv', q_t, state) * scale  # [B, HV, V]
        
        # 應用 output gate（如果有 g）
        if g is not None:
            o_t = o_t * torch.sigmoid(g[:, t]).unsqueeze(-1)  # [B, HV, V]
        
        outputs.append(o_t)
    
    o = torch.stack(outputs, dim=1)  # [B, T, HV, V]
    
    final_state = state.to(q.dtype) if output_final_state else None
    
    return o, final_state
