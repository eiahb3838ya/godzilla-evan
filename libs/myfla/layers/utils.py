"""Utility helpers for `myfla.layers`.

源碼鉤子：`libs/fla/layers/utils.py`（commit 與官方同步）。
"""

from __future__ import annotations

from typing import Tuple

import torch
from einops import rearrange, repeat

from myfla.ops.utils import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
from myfla.utils import tensor_cache


class IndexFirstAxis(torch.autograd.Function):
    """Autograd-friendly gather along第一維（對應 fla.layers.utils.IndexFirstAxis）。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(indices)
        if x.ndim < 2:
            raise ValueError('IndexFirstAxis 僅支援 ndim >= 2')
        ctx.first_axis_dim, other_shape = x.shape[0], x.shape[1:]
        second_dim = other_shape.numel()
        flat = rearrange(x, 'b ... -> b (...)')
        gathered = torch.gather(flat, 0, repeat(indices, 'z -> z d', d=second_dim))
        return gathered.reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        (indices,) = ctx.saved_tensors
        if grad_output.ndim < 2:
            raise ValueError('梯度輸入維度必須 >= 2')
        other_shape = grad_output.shape[1:]
        grad_flat = rearrange(grad_output, 'b ... -> b (...)')
        dx = grad_flat.new_zeros((ctx.first_axis_dim, grad_flat.shape[1]))
        dx.scatter_(0, repeat(indices, 'z -> z d', d=grad_flat.shape[1]), grad_flat)
        return dx.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    """拋光版 scatter，用於 `pad_input`（對應 fla.layers.utils.IndexPutFirstAxis）。"""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        indices: torch.Tensor,
        first_axis_dim: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(indices)
        if indices.ndim != 1:
            raise ValueError('indices 必須為 1 維')
        if x.ndim < 2:
            raise ValueError('x ndim 必須 >= 2')
        y = x.new_zeros((first_axis_dim, *x.shape[1:]))
        y[indices] = x
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:  # type: ignore[override]
        (indices,) = ctx.saved_tensors
        return grad_output[indices], None, None


index_put_first_axis = IndexPutFirstAxis.apply


@tensor_cache
def get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """複刻 `fla.layers.utils.get_unpad_data`。

    Args:
        attention_mask: `[batch, seq_len]` 之 0/1（或 bool）Tensor，1 代表有效 token。
    """

    if attention_mask.ndim != 2:
        raise ValueError('attention_mask 應為 [batch, seq_len]')
    mask = attention_mask.to(dtype=torch.bool)
    lens = prepare_lens_from_mask(mask)
    cu_seqlens = prepare_cu_seqlens_from_mask(mask, dtype=torch.int32)
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    max_len = int(lens.max().item()) if lens.numel() > 0 else 0
    return indices.to(torch.long), cu_seqlens, max_len


def unpad_input(
    q: torch.Tensor,
    states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = False,
):
    """配合 delta/kda layer 的 padding→varlen 轉換（移植自 fla.layers.utils.unpad_input）。"""

    indices_k, cu_seqlens_k, max_k = get_unpad_data(attention_mask)
    batch_size, seq_len = states[0].shape[:2]

    def _flatten_and_index(tensor: torch.Tensor) -> torch.Tensor:
        return index_first_axis(rearrange(tensor, 'b s ... -> (b s) ...'), indices_k)

    state = tuple(_flatten_and_index(s) for s in states)

    if q_len == seq_len:
        q = _flatten_and_index(q)
        indices_q = indices_k
        cu_seqlens_q = cu_seqlens_k
        max_q = max_k
    elif q_len == 1:
        max_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        raise NotImplementedError('僅支援 q_len == seq_len 或 q_len == 1')

    if keepdim:
        q = q.unsqueeze(0)
        state = tuple(s.unsqueeze(0) for s in state)

    return q, state, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_q, max_k)


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """將 varlen 張量重新填回 `[batch, seq_len, ...]`（移植自 fla.layers.utils.pad_input）。"""

    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, '(b s) ... -> b s ...', b=batch_size)


__all__ = [
    'get_unpad_data',
    'index_first_axis',
    'index_put_first_axis',
    'pad_input',
    'unpad_input',
]
