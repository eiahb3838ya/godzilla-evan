"""Pure PyTorch token_shift 實作，對齊 libs.fla.modules.token_shift 行為。"""

from __future__ import annotations

import torch
import torch.nn as nn


def _normalize_batch_cache(cache: torch.Tensor, batch: int, dim: int) -> torch.Tensor:
    if cache.dim() == 2:
        if cache.shape != (batch, dim):
            raise ValueError(f"cache shape 必須為 [{batch}, {dim}]，取得 {tuple(cache.shape)}")
        return cache
    if cache.dim() == 3 and cache.shape[1] == 1 and cache.shape[0] == batch and cache.shape[2] == dim:
        return cache[:, 0, :]
    raise ValueError("cache 形狀應為 [B, D] 或 [B, 1, D]")


def token_shift(
    x: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    cache: torch.Tensor | None = None,
    output_cache: bool = False,
):
    """
    Args:
        x: [B, T, D]。
        cu_seqlens: 變長序列的 cumulative 長度（與 FlashAttention 相同約定）。
        cache: 初始 state（無 cu 時 shape=[B,D]；有 cu 時 shape=[N,D]）。
        output_cache: True 時回傳 (delta, conv_state)，否則只回傳 delta。
    """
    if cu_seqlens is None:
        delta = _token_shift_batch(x, cache)
        if output_cache:
            cache_out = x[:, -1, :].contiguous()
            return delta, cache_out
        return delta
    delta, cache_out = _token_shift_varlen(x, cu_seqlens, cache, output_cache)
    if output_cache:
        return delta, cache_out
    return delta


def _token_shift_batch(x: torch.Tensor, cache: torch.Tensor | None) -> torch.Tensor:
    batch, _, dim = x.shape
    time_shift = nn.ZeroPad2d((0, 0, 1, -1))
    shifted = time_shift(x)
    if cache is not None:
        cache_norm = _normalize_batch_cache(cache, batch, dim)
        shifted[:, 0, :] = cache_norm
    delta = shifted - x
    return delta


def _token_shift_varlen(
    x: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    cache: torch.Tensor | None,
    output_cache: bool,
):
    if x.dim() != 3:
        raise ValueError("變長模式輸入必須為 [1, T, D]")
    batch, total_len, dim = x.shape
    if batch != 1:
        raise ValueError("使用 cu_seqlens 時 batch 大小需為 1")
    seq_count = cu_seqlens.numel() - 1
    if cache is not None:
        if cache.dim() != 2 or cache.shape != (seq_count, dim):
            raise ValueError(f"變長模式 cache 需為 [{seq_count}, {dim}]，取得 {tuple(cache.shape)}")
        cache_tensor = cache
    else:
        cache_tensor = None

    delta = torch.zeros_like(x)
    cache_out = x.new_empty((seq_count, dim)) if output_cache else None

    for idx in range(seq_count):
        start = cu_seqlens[idx].item()
        end = cu_seqlens[idx + 1].item()
        if end <= start:
            if cache_out is not None:
                cache_out[idx] = 0
            continue
        segment = x[0, start:end, :]
        prev = cache_tensor[idx] if cache_tensor is not None else None
        shifted = torch.zeros_like(segment)
        if segment.size(0) > 1:
            shifted[1:, :] = segment[:-1, :]
        if prev is not None:
            shifted[0, :] = prev
        delta_segment = shifted - segment
        delta[0, start:end, :] = delta_segment
        if cache_out is not None:
            cache_out[idx] = segment[-1, :]

    return delta, cache_out
