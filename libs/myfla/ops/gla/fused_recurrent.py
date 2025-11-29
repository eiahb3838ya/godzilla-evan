"""Pure PyTorch fallback for `libs.fla.ops.gla.fused_recurrent`.

此版本僅依賴 PyTorch Tensor 運算，並保留與官方函式相同的簽名，以供 myfla.layers.gla 呼叫。
"""

from __future__ import annotations

from typing import Tuple

import torch


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
) -> Tuple[int, int, int, int]:
    if not (q.shape == k.shape):
        raise ValueError("`q` 與 `k` 形狀必須一致")
    if q.shape[:3] != v.shape[:3]:
        raise ValueError("`q/k/v` 的 batch、seq、head 維度需一致")
    if gk is not None and gk.shape != q.shape:
        raise ValueError("`gk` 形狀需與 `q` 相同")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError("使用 varlen (`cu_seqlens`) 時，`q` 的 batch 維度必須為 1")
        seq_count = len(cu_seqlens) - 1
        if initial_state is not None and initial_state.shape[0] != seq_count:
            raise ValueError("`initial_state` 序列數需與 `cu_seqlens` 一致")
    else:
        if initial_state is not None and initial_state.shape[0] not in (q.shape[0],):
            raise ValueError("固定長度模式下 `initial_state` 的 batch 維度需等於 `q` 的 batch")
    return q.shape


def _fused_recurrent_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None,
    state: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """處理單一序列（shape: [T, H, D]）。"""
    T, H, K = q.shape
    V = v.shape[-1]
    outputs = []
    current_state = state.clone()
    for t in range(T):
        q_t = q[t] * scale  # [H, K]
        k_t = k[t]  # [H, K]
        v_t = v[t]  # [H, V]

        if gk is not None:
            forget = torch.exp(gk[t])  # [H, K]
            current_state = current_state * forget.unsqueeze(-1)

        # outer product: [H, K, V]
        delta = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        current_state = current_state + delta.to(current_state.dtype)

        o_t = torch.einsum('hkv,hk->hv', current_state, q_t.to(current_state.dtype))
        outputs.append(o_t.to(v.dtype))
    return torch.stack(outputs, dim=0), current_state


def fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    純 PyTorch 版 fused_recurrent GLA。

    Args mirror `libs.fla.ops.gla.fused_recurrent_gla`。
    """
    if gv is not None:
        raise NotImplementedError("PyTorch 版本暫不支援 `gv`（GLA 目前未使用該分支）")
    B, T, H, K = _validate_inputs(q, k, v, gk, initial_state, cu_seqlens)
    if scale is None:
        scale = K ** -0.5

    def _init_state(num_seq: int) -> torch.Tensor:
        if initial_state is None:
            return torch.zeros(num_seq, H, K, v.shape[-1], dtype=torch.float32, device=q.device)
        return initial_state.to(torch.float32)

    if cu_seqlens is None:
        state = _init_state(B)
        outputs = []
        final_states = []
        itr = range(B - 1, -1, -1) if reverse else range(B)
        for b in itr:
            seq_q = q[b]
            seq_k = k[b]
            seq_v = v[b]
            seq_gk = gk[b] if gk is not None else None
            out, new_state = _fused_recurrent_single(seq_q, seq_k, seq_v, seq_gk, state[b], scale)
            outputs.append(out)
            final_states.append(new_state)
        outputs = outputs[::-1] if reverse else outputs
        final_states = final_states[::-1] if reverse else final_states
        stacked_out = torch.stack(outputs, dim=0)
        stacked_state = torch.stack(final_states, dim=0)
    else:
        seq_count = len(cu_seqlens) - 1
        state = _init_state(seq_count)
        outputs_list = []
        final_state_list = []
        q_flat = q[0]
        k_flat = k[0]
        v_flat = v[0]
        gk_flat = gk[0] if gk is not None else None
        order = range(seq_count - 1, -1, -1) if reverse else range(seq_count)
        for idx in order:
            start = int(cu_seqlens[idx].item())
            end = int(cu_seqlens[idx + 1].item())
            if end <= start:
                outputs_list.append(torch.empty(0, H, v.shape[-1], device=q.device, dtype=v.dtype))
                final_state_list.append(state[idx])
                continue
            out, new_state = _fused_recurrent_single(
                q_flat[start:end],
                k_flat[start:end],
                v_flat[start:end],
                None if gk_flat is None else gk_flat[start:end],
                state[idx],
                scale,
            )
            outputs_list.append(out)
            final_state_list.append(new_state)
        outputs_list = outputs_list[::-1] if reverse else outputs_list
        final_state_list = final_state_list[::-1] if reverse else final_state_list
        stacked_out = torch.cat(outputs_list, dim=0).unsqueeze(0)
        stacked_state = torch.stack(final_state_list, dim=0)

    final_state = stacked_state.to(v.dtype) if output_final_state else None
    return stacked_out, final_state


__all__ = ['fused_recurrent_gla']
