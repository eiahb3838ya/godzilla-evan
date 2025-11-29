"""PyTorch fallback for `libs/fla/ops/utils/solve_tril.py`."""

from __future__ import annotations

from typing import Optional

import torch

from myfla.ops.utils.index import prepare_lens
from myfla.utils import input_guard

_SUPPORTED_BT = {16, 32, 64}


def _solve_chunk(
    data: torch.Tensor,
    out: torch.Tensor,
    chunk_size: int,
    start: int,
    end: int,
) -> None:
    device = data.device
    dtype = data.dtype
    pos = start
    while pos < end:
        chunk_len = min(chunk_size, end - pos)
        block = data[pos:pos + chunk_len, :, :chunk_len].permute(1, 0, 2).contiguous()
        work_dtype = torch.float32 if block.dtype in (torch.float16, torch.bfloat16) else block.dtype
        block = block.to(work_dtype)
        strict = torch.tril(block, diagonal=-1)
        eye = torch.eye(chunk_len, dtype=work_dtype, device=device).expand(block.shape[0], -1, -1)
        matrix = eye + strict
        inv = torch.linalg.inv(matrix)
        out_chunk = inv.permute(1, 0, 2).to(out.dtype)
        out[pos:pos + chunk_len, :, :chunk_len] = out_chunk
        pos += chunk_len


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.float,
) -> torch.Tensor:
    """
    Compute `(I + A)^-1` where `A` is strictly lower triangular within每個 chunk。

    Args:
        A: `[B, T, H, BT]` tensor，`BT` 僅支援 16/32/64。
        cu_seqlens: packed sequence offsets（與 fla 版一致，batch varlen 時僅支援 `B==1`）。
        output_dtype: 輸出 dtype，預設 `torch.float`。
    """

    if A.ndim != 4:
        raise ValueError('solve_tril 僅支援 4 維輸入 [B, T, H, BT]')
    B, T, H, BT = A.shape
    if BT not in _SUPPORTED_BT:
        raise ValueError(f'solve_tril 僅支援 chunk size {sorted(_SUPPORTED_BT)}，收到 {BT}')
    if output_dtype is None:
        output_dtype = A.dtype

    Ai = torch.zeros_like(A, dtype=output_dtype)
    data = A.to(A.dtype)

    if cu_seqlens is None:
        for b in range(B):
            _solve_chunk(data[b], Ai[b], BT, 0, T)
    else:
        if B != 1:
            raise ValueError('變長模式僅支援 batch size = 1')
        offsets = cu_seqlens.to(torch.long).tolist()
        if offsets[0] != 0:
            raise ValueError('cu_seqlens 應以 0 起始')
        for idx in range(len(offsets) - 1):
            start, end = offsets[idx], offsets[idx + 1]
            if end <= start:
                continue
            _solve_chunk(data[0], Ai[0], BT, start, end)

    return Ai


__all__ = ['solve_tril']
