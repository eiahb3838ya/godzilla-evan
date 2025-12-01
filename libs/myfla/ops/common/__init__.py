"""Common helpers shared by delta-rule / KDA（ported from `libs.fla.ops.common`)."""

from .chunk_delta_rule import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_bwd_dv_local

__all__ = [
    'chunk_gated_delta_rule_fwd_h',
    'chunk_gated_delta_rule_bwd_dhu',
    'chunk_bwd_dv_local',
]
