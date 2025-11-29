"""Pack/unpack helpers（源碼鉤子：`libs/fla/ops/utils/pack.py`)."""

from __future__ import annotations

from typing import Optional

import torch

from myfla.ops.utils.index import prepare_lens


def _validate_padding_side(padding_side: str) -> None:
    if padding_side not in {'left', 'right'}:
        raise ValueError("padding_side 需為 'left' 或 'right'")


def pack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str = 'left',
) -> torch.Tensor:
    """Pure PyTorch fallback for sequence packing.

    完整對齊 `fla.ops.utils.pack.pack_sequence` 的介面，後續若需 Triton 版本可在此替換。
    """

    _validate_padding_side(padding_side)
    if x.ndim < 2:
        raise ValueError('pack_sequence 僅支援 ndim >= 2')
    lens = prepare_lens(cu_seqlens)
    total_tokens = int(lens.sum().item())
    if total_tokens == 0:
        return x.new_zeros((0,) + x.shape[2:])

    outputs = []
    seq_len = x.shape[1]
    for batch_idx, length in enumerate(lens.tolist()):
        if length == 0:
            continue
        if padding_side == 'left':
            start = seq_len - length
        else:
            start = 0
        end = start + length
        outputs.append(x[batch_idx, start:end])
    return torch.cat(outputs, dim=0)


def unpack_sequence(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str = 'left',
    desired_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    """Inverse operation of :func:`pack_sequence`."""

    _validate_padding_side(padding_side)
    if x.ndim < 1:
        raise ValueError('x 至少為 1 維')
    lens = prepare_lens(cu_seqlens)
    batch_size = lens.shape[0]
    if desired_shape is None:
        max_len = int(lens.max().item()) if lens.numel() > 0 else 0
        desired_shape = (batch_size, max_len, *x.shape[1:])
    y = x.new_zeros(desired_shape)

    offset = 0
    seq_len = desired_shape[1]
    for batch_idx, length in enumerate(lens.tolist()):
        if length == 0:
            continue
        slice_x = x[offset: offset + length]
        offset += length
        if padding_side == 'left':
            start = seq_len - length
        else:
            start = 0
        end = start + length
        y[batch_idx, start:end] = slice_x
    return y


__all__ = ['pack_sequence', 'unpack_sequence']
