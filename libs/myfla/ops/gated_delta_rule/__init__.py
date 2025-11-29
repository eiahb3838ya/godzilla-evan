"""
完美復刻官方 `libs.fla.ops.gated_delta_rule`。

源自：libs/fla/ops/gated_delta_rule/__init__.py
"""

from .chunk import chunk_gated_delta_rule
from .fused_recurrent import fused_recurrent_gated_delta_rule

__all__ = [
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
