"""Index helpers for ops（移植自 `libs/fla/ops/utils/index.py`)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from myfla.utils import tensor_cache


def _ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError('divisor 必須為正整數')
    return (value + divisor - 1) // divisor


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Return lengths per batch element."""

    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert 0/1 mask to per batch lengths."""

    boolean_mask = mask.to(dtype=torch.bool)
    return boolean_mask.sum(dim=-1, dtype=torch.int32)


@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.int32,
) -> torch.Tensor:
    if dtype is None:
        dtype = lens.dtype
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


@tensor_cache
def prepare_cu_seqlens_from_mask(
    mask: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.int32,
) -> torch.Tensor:
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


@tensor_cache
def prepare_lens_from_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_split_cu_seqlens(
    batch_size: int,
    seq_len: int,
    split_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.int32,
    device: Optional[torch.device] = torch.device('cpu'),
) -> torch.Tensor:
    if split_size <= 0:
        raise ValueError('split_size 必須為正整數')
    if cu_seqlens is None:
        total_tokens = batch_size * seq_len
        sequence_offsets = list(range(0, total_tokens, seq_len)) + [total_tokens]
    else:
        sequence_offsets = cu_seqlens.tolist()

    values = [
        i
        for bos, eos in zip(sequence_offsets[:-1], sequence_offsets[1:], strict=False)
        for i in range(bos, eos, split_size)
    ]
    values.append(sequence_offsets[-1])
    return torch.tensor(values, dtype=dtype, device=device)


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.Tensor) -> torch.Tensor:
    lens = prepare_lens(cu_seqlens)
    if lens.numel() == 0:
        return cu_seqlens.new_zeros(0)
    positions = [
        torch.arange(int(length.item()), dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for length in lens
    ]
    if not positions:
        return cu_seqlens.new_zeros(0)
    return torch.cat(positions, dim=0)


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.Tensor) -> torch.Tensor:
    position_ids = prepare_position_ids(cu_seqlens)
    if position_ids.numel() == 0:
        return cu_seqlens.new_zeros(0)
    return position_ids.eq(0).cumsum(0, dtype=cu_seqlens.dtype) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.Tensor) -> torch.Tensor:
    position_ids = prepare_position_ids(cu_seqlens)
    sequence_ids = prepare_sequence_ids(cu_seqlens)
    if position_ids.numel() == 0:
        return cu_seqlens.new_zeros((0, 2))
    stacked = torch.stack([sequence_ids, position_ids], dim=1)
    return stacked.to(dtype=cu_seqlens.dtype)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError('chunk_size 必須為正整數')
    lens = prepare_lens(cu_seqlens)
    if lens.numel() == 0:
        return cu_seqlens.new_zeros((0, 2))
    chunk_counts = [
        _ceil_div(int(length.item()), chunk_size)
        for length in lens
    ]
    indices = [
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in chunk_counts if n > 0
    ]
    if not indices:
        return cu_seqlens.new_zeros((0, 2))
    indices_tensor = torch.cat(indices, dim=0)
    chunk_ids = indices_tensor.eq(0).cumsum(0, dtype=cu_seqlens.dtype) - 1
    return torch.stack([chunk_ids, indices_tensor], dim=1)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError('chunk_size 必須為正整數')
    lens = prepare_lens(cu_seqlens)
    if lens.numel() == 0:
        return cu_seqlens.new_zeros(1)
    chunk_counts = torch.tensor(
        [_ceil_div(int(length.item()), chunk_size) for length in lens],
        dtype=cu_seqlens.dtype,
        device=cu_seqlens.device,
    )
    return torch.cat([cu_seqlens.new_zeros(1), chunk_counts]).cumsum(0, dtype=cu_seqlens.dtype)


@tensor_cache
def get_max_num_splits(cu_seqlens: torch.Tensor, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError('chunk_size 必須為正整數')
    lens = prepare_lens(cu_seqlens)
    if lens.numel() == 0:
        return 0
    max_len = int(lens.max().item())
    return _ceil_div(max_len, chunk_size)
