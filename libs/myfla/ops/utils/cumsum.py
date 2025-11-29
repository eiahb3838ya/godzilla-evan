"""Chunked cumsum helpers（源碼鉤子：`libs/fla/ops/utils/cumsum.py`)."""

from __future__ import annotations

from typing import Optional

import torch

from myfla.ops.utils.index import prepare_lens
from myfla.utils import input_guard


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _maybe_scale(tensor: torch.Tensor, scale: Optional[torch.Tensor | float]) -> torch.Tensor:
    if scale is None:
        return tensor
    if isinstance(scale, torch.Tensor):
        return tensor * scale.to(dtype=tensor.dtype, device=tensor.device)
    return tensor * tensor.new_tensor(scale)


def _chunked_sequence_cumsum(
    seq: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: Optional[torch.Tensor | float],
    result_dtype: torch.dtype,
) -> torch.Tensor:
    length = seq.shape[0]
    out = seq.new_empty((length, *seq.shape[1:]), dtype=result_dtype)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        chunk = seq[start:end].to(torch.float32)
        if reverse:
            processed = torch.flip(torch.flip(chunk, dims=(0,)).cumsum(0), dims=(0,))
        else:
            processed = chunk.cumsum(0)
        processed = _maybe_scale(processed, scale)
        out[start:end] = processed.to(result_dtype)
    return out


def _global_sequence_cumsum(
    seq: torch.Tensor,
    reverse: bool,
    scale: Optional[torch.Tensor | float],
    result_dtype: torch.dtype,
) -> torch.Tensor:
    chunk = seq.to(torch.float32)
    if reverse:
        processed = torch.flip(torch.flip(chunk, dims=(0,)).cumsum(0), dims=(0,))
    else:
        processed = chunk.cumsum(0)
    processed = _maybe_scale(processed, scale)
    return processed.to(result_dtype)


def _canonicalize_scalar(tensor: torch.Tensor, head_first: bool) -> torch.Tensor:
    return tensor if head_first else tensor.permute(0, 2, 1).contiguous()


def _canonicalize_vector(tensor: torch.Tensor, head_first: bool) -> torch.Tensor:
    return tensor if head_first else tensor.permute(0, 2, 1, 3).contiguous()


def _restore_scalar(tensor: torch.Tensor, head_first: bool) -> torch.Tensor:
    return tensor if head_first else tensor.permute(0, 2, 1)


def _restore_vector(tensor: torch.Tensor, head_first: bool) -> torch.Tensor:
    return tensor if head_first else tensor.permute(0, 2, 1, 3)


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: Optional[torch.Tensor | float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    if not _is_power_of_two(chunk_size):
        raise ValueError('chunk_size 必須為 2 的冪次')
    canonical = _canonicalize_scalar(g, head_first=head_first)
    result_dtype = output_dtype or g.dtype
    out = torch.empty_like(canonical, dtype=result_dtype)
    if cu_seqlens is None:
        for b in range(canonical.shape[0]):
            for h in range(canonical.shape[1]):
                out[b, h] = _chunked_sequence_cumsum(canonical[b, h], chunk_size, reverse, scale, result_dtype)
    else:
        if canonical.shape[0] != 1:
            raise ValueError('變長模式僅支援 batch size 1')
        lens = prepare_lens(cu_seqlens)
        offsets = cu_seqlens.to(torch.long)
        for seq_idx in range(len(lens)):
            bos = int(offsets[seq_idx].item())
            eos = int(offsets[seq_idx + 1].item())
            if bos == eos:
                continue
            for h in range(canonical.shape[1]):
                out[0, h, bos:eos] = _chunked_sequence_cumsum(
                    canonical[0, h, bos:eos],
                    chunk_size,
                    reverse,
                    scale,
                    result_dtype,
                )
    return _restore_scalar(out, head_first=head_first)


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: Optional[torch.Tensor | float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    if not _is_power_of_two(chunk_size):
        raise ValueError('chunk_size 必須為 2 的冪次')
    canonical = _canonicalize_vector(g, head_first=head_first)
    result_dtype = output_dtype or g.dtype
    out = torch.empty_like(canonical, dtype=result_dtype)
    if cu_seqlens is None:
        for b in range(canonical.shape[0]):
            for h in range(canonical.shape[1]):
                out[b, h] = _chunked_sequence_cumsum(canonical[b, h], chunk_size, reverse, scale, result_dtype)
    else:
        if canonical.shape[0] != 1:
            raise ValueError('變長模式僅支援 batch size 1')
        lens = prepare_lens(cu_seqlens)
        offsets = cu_seqlens.to(torch.long)
        for seq_idx in range(len(lens)):
            bos = int(offsets[seq_idx].item())
            eos = int(offsets[seq_idx + 1].item())
            if bos == eos:
                continue
            for h in range(canonical.shape[1]):
                out[0, h, bos:eos] = _chunked_sequence_cumsum(
                    canonical[0, h, bos:eos],
                    chunk_size,
                    reverse,
                    scale,
                    result_dtype,
                )
    return _restore_vector(out, head_first=head_first)


@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor | float] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    canonical = _canonicalize_scalar(s, head_first=head_first)
    result_dtype = output_dtype or s.dtype
    out = torch.empty_like(canonical, dtype=result_dtype)
    if cu_seqlens is None:
        for b in range(canonical.shape[0]):
            for h in range(canonical.shape[1]):
                out[b, h] = _global_sequence_cumsum(canonical[b, h], reverse, scale, result_dtype)
    else:
        if canonical.shape[0] != 1:
            raise ValueError('變長模式僅支援 batch size 1')
        lens = prepare_lens(cu_seqlens)
        offsets = cu_seqlens.to(torch.long)
        for seq_idx in range(len(lens)):
            bos = int(offsets[seq_idx].item())
            eos = int(offsets[seq_idx + 1].item())
            if bos == eos:
                continue
            for h in range(canonical.shape[1]):
                out[0, h, bos:eos] = _global_sequence_cumsum(
                    canonical[0, h, bos:eos],
                    reverse,
                    scale,
                    result_dtype,
                )
    return _restore_scalar(out, head_first=head_first)


@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor | float] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    canonical = _canonicalize_vector(s, head_first=head_first)
    result_dtype = output_dtype or s.dtype
    out = torch.empty_like(canonical, dtype=result_dtype)
    if cu_seqlens is None:
        for b in range(canonical.shape[0]):
            for h in range(canonical.shape[1]):
                out[b, h] = _global_sequence_cumsum(canonical[b, h], reverse, scale, result_dtype)
    else:
        if canonical.shape[0] != 1:
            raise ValueError('變長模式僅支援 batch size 1')
        lens = prepare_lens(cu_seqlens)
        offsets = cu_seqlens.to(torch.long)
        for seq_idx in range(len(lens)):
            bos = int(offsets[seq_idx].item())
            eos = int(offsets[seq_idx + 1].item())
            if bos == eos:
                continue
            for h in range(canonical.shape[1]):
                out[0, h, bos:eos] = _global_sequence_cumsum(
                    canonical[0, h, bos:eos],
                    reverse,
                    scale,
                    result_dtype,
                )
    return _restore_vector(out, head_first=head_first)


@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor | float] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    if s.ndim == 3:
        return chunk_global_cumsum_scalar(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    if s.ndim == 4:
        return chunk_global_cumsum_vector(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    raise ValueError(
        f"Unsupported input shape {s.shape}; 預期 [B, T, H] 或 [B, T, H, D]（head_first=False）或 [B, H, T]/[B, H, T, D]",
    )


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: Optional[torch.Tensor | float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    if g.ndim == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    if g.ndim == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    raise ValueError(
        f"Unsupported input shape {g.shape}; 預期 [B, T, H]/[B, T, H, D] 或 head_first variant",
    )


__all__ = [
    'chunk_global_cumsum',
    'chunk_global_cumsum_scalar',
    'chunk_global_cumsum_vector',
    'chunk_local_cumsum',
    'chunk_local_cumsum_scalar',
    'chunk_local_cumsum_vector',
]
