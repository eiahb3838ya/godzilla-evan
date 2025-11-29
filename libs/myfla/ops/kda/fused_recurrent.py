"""Fused recurrent KDA kernel（待移植自 libs.fla.ops.kda.fused_recurrent_kda）。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def fused_recurrent_kda(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    raise NotImplementedError(
        "Port of libs.fla.ops.kda.fused_recurrent_kda 尚未完成",
    )


__all__ = ['fused_recurrent_kda']
