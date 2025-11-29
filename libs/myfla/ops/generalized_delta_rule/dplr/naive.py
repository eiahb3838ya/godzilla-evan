# Source: Simplified PyTorch delta-rule recurrence (based on libs/fla/ops/generalized_delta_rule/dplr/naive.py)

from __future__ import annotations

from einops import rearrange
import torch


def dplr_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    scale: float | None = None,
    reverse: bool = False,
):
    """Reference recurrence: S_t = exp(gk_t) * S_{t-1} + k_t v_t^T + (S alpha) beta^T."""
    orig_dtype = q.dtype
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]

    q_float, k_float, v_float = q.float(), k.float(), v.float()
    alpha_float, beta_float, gk_float = alpha.float(), beta.float(), gk.float()
    if scale is None:
        scale = d_k ** -0.5
    q_float = q_float * scale
    state = torch.zeros(b, h, d_k, d_v, dtype=q_float.dtype, device=q_float.device)
    if initial_state is not None:
        state = state + initial_state.float()

    outputs = torch.zeros_like(v_float)
    time_indices = range(l - 1, -1, -1) if reverse else range(l)

    for t in time_indices:
        decay = gk_float[:, :, t].exp().unsqueeze(-1)
        kv_term = torch.einsum('bhd,bhv->bhdv', k_float[:, :, t], v_float[:, :, t])
        alpha_term = torch.einsum('bhd,bhdv->bhv', alpha_float[:, :, t], state)
        beta_term = torch.einsum('bhv,bhd->bhdv', alpha_term, beta_float[:, :, t])
        state = state * decay + kv_term + beta_term
        outputs[:, :, t] = torch.einsum('bhd,bhdv->bhv', q_float[:, :, t], state)

    final_state = state if output_final_state else None
    return outputs.to(orig_dtype), final_state


def dplr_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    scale: float | None = None,
    reverse: bool = False,
):
    """變長版本：x shape [1, T, ...]; cu_seqlens=[0,len1,len1+len2,...]."""
    if q.shape[0] != 1:
        raise ValueError("cu_seqlens 變長模式僅支援 batch=1，請先攤平成單序列。")
    num_seq = cu_seqlens.numel() - 1
    d_k = q.shape[-1]
    d_v = v.shape[-1]
    heads = q.shape[2]
    outputs = torch.zeros_like(v)
    final_states = torch.zeros(num_seq, heads, d_k, d_v, device=q.device, dtype=q.dtype) if output_final_state else None
    if initial_state is None:
        init_states = torch.zeros(num_seq, heads, d_k, d_v, device=q.device, dtype=q.dtype)
    else:
        init_states = initial_state

    for idx in range(num_seq):
        start = cu_seqlens[idx].item()
        end = cu_seqlens[idx + 1].item()
        if end <= start:
            continue
        sub_q = rearrange(q[:, start:end], 'b t h d -> b h t d')
        sub_k = rearrange(k[:, start:end], 'b t h d -> b h t d')
        sub_v = rearrange(v[:, start:end], 'b t h d -> b h t d')
        sub_alpha = rearrange(alpha[:, start:end], 'b t h d -> b h t d')
        sub_beta = rearrange(beta[:, start:end], 'b t h d -> b h t d')
        sub_gk = rearrange(gk[:, start:end], 'b t h d -> b h t d')
        init_state_seq = init_states[idx].unsqueeze(0)
        o, st = dplr_recurrence(
            sub_q,
            sub_k,
            sub_v,
            sub_alpha,
            sub_beta,
            sub_gk,
            initial_state=init_state_seq,
            output_final_state=output_final_state,
            scale=scale,
            reverse=reverse,
        )
        outputs[:, start:end] = rearrange(o, 'b h t d -> b t h d')
        if output_final_state and st is not None:
            final_states[idx] = st.squeeze(0)
    outputs = rearrange(outputs, 'b t h d -> b h t d')
    return outputs, final_states
