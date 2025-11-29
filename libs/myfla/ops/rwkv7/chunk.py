from __future__ import annotations

import warnings

from einops import rearrange
import torch

from myfla.ops.generalized_delta_rule import dplr_recurrence, dplr_varlen


def _maybe_time_major(tensor, head_first: bool):
    if head_first:
        return rearrange(tensor, 'b h t d -> b t h d')
    return tensor


def _to_head_major(tensor):
    return rearrange(tensor, 'b t h d -> b h t d')


def _ensure_matching_shapes(reference, tensors):
    ref_shape = reference.shape[:-1]
    for name, tensor in tensors:
        if tensor.shape[:-1] != ref_shape:
            raise ValueError(f"{name} 與參考張量形狀不匹配：{tensor.shape[:-1]} vs {ref_shape}")


def _validate_cu_inputs(batch_size, cu_seqlens, initial_state):
    if cu_seqlens is None:
        return
    if batch_size != 1:
        raise ValueError("cu_seqlens 僅支援 batch=1，請先將序列攤平成單批次。")
    expected = cu_seqlens.numel() - 1
    if initial_state is not None and initial_state.shape[0] != expected:
        raise ValueError(
            f"initial_state.shape[0]={initial_state.shape[0]} 與 cu_seqlens 序列數 {expected} 不符。",
        )


def chunk_dplr_delta_rule(
    q,
    k,
    v,
    a,
    b,
    gk,
    scale: float | None = 1.0,
    initial_state=None,
    output_final_state=True,
    cu_seqlens=None,
    head_first: bool = False,
):
    q_time = _maybe_time_major(q, head_first)
    k_time = _maybe_time_major(k, head_first)
    v_time = _maybe_time_major(v, head_first)
    a_time = _maybe_time_major(a, head_first)
    b_time = _maybe_time_major(b, head_first)
    gk_time = _maybe_time_major(gk, head_first)
    _ensure_matching_shapes(q_time, [('w', gk_time), ('k', k_time), ('v', v_time), ('a', a_time), ('b', b_time)])
    _validate_cu_inputs(q_time.shape[0], cu_seqlens, initial_state)
    eff_scale = (q_time.shape[-1] ** -0.5) if scale is None else scale
    if cu_seqlens is None:
        outputs, final_state = dplr_recurrence(
            _to_head_major(q_time),
            _to_head_major(k_time),
            _to_head_major(v_time),
            _to_head_major(a_time),
            _to_head_major(b_time),
            _to_head_major(gk_time),
            initial_state=initial_state,
            output_final_state=output_final_state,
            scale=eff_scale,
        )
    else:
        outputs, final_state = dplr_varlen(
            q_time,
            k_time,
            v_time,
            a_time,
            b_time,
            gk_time,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
            scale=eff_scale,
        )
    return outputs, final_state


def fused_recurrent_dplr_delta_rule(
    q,
    k,
    v,
    a,
    b,
    gk,
    scale: float | None = 1.0,
    initial_state=None,
    output_final_state=True,
    cu_seqlens=None,
    reverse: bool = False,
    head_first: bool = False,
):
    q_time = _maybe_time_major(q, head_first)
    k_time = _maybe_time_major(k, head_first)
    v_time = _maybe_time_major(v, head_first)
    a_time = _maybe_time_major(a, head_first)
    b_time = _maybe_time_major(b, head_first)
    gk_time = _maybe_time_major(gk, head_first)
    _ensure_matching_shapes(q_time, [('w', gk_time), ('k', k_time), ('v', v_time), ('a', a_time), ('b', b_time)])
    _validate_cu_inputs(q_time.shape[0], cu_seqlens, initial_state)
    eff_scale = (q_time.shape[-1] ** -0.5) if scale is None else scale
    if cu_seqlens is None:
        outputs, final_state = dplr_recurrence(
            _to_head_major(q_time),
            _to_head_major(k_time),
            _to_head_major(v_time),
            _to_head_major(a_time),
            _to_head_major(b_time),
            _to_head_major(gk_time),
            initial_state=initial_state,
            output_final_state=output_final_state,
            scale=eff_scale,
            reverse=reverse,
        )
    else:
        outputs, final_state = dplr_varlen(
            q_time,
            k_time,
            v_time,
            a_time,
            b_time,
            gk_time,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
            scale=eff_scale,
            reverse=reverse,
        )
    return outputs, final_state


def chunk_rwkv7(
    r,
    w,
    k,
    v,
    a,
    b,
    scale=1.0,
    initial_state=None,
    output_final_state=True,
    cu_seqlens=None,
    head_first=False,
):
    if head_first:
        warnings.warn("head_first 介面已棄用，未來版本將移除，請優先傳入 [B, T, H, ...] 格式。", stacklevel=2)
    if not head_first and r.shape[1] < r.shape[2]:
        warnings.warn(
            f"偵測到 seq_len({r.shape[1]}) < num_heads({r.shape[2]})，可能使用了 head-first 輸入格式。",
            stacklevel=2,
        )
    outputs, state = chunk_dplr_delta_rule(
        r,
        k,
        v,
        a,
        b,
        w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
    )
    if head_first:
        return outputs, state
    outputs = rearrange(outputs, 'b h t d -> b t h d')
    return outputs, state


def fused_mul_recurrent_rwkv7(
    r,
    w,
    k,
    v,
    kk,
    a,
    scale=1.0,
    initial_state=None,
    output_final_state=False,
    reverse=False,
    cu_seqlens=None,
    head_first=False,
):
    legacy_mode = torch.allclose(k, kk)
    if legacy_mode:
        a_term = a
        b_term = kk * a
    else:
        a_term = -kk
        b_term = kk * a
    outputs, state = fused_recurrent_dplr_delta_rule(
        r,
        k,
        v,
        a_term,
        b_term,
        w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        head_first=head_first,
    )
    if head_first:
        return outputs, state
    outputs = rearrange(outputs, 'b h t d -> b t h d')
    return outputs, state
