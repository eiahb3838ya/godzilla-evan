# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# -*- coding: utf-8 -*-

"""
KDA (Kimi Delta Attention) Operations - PyTorch Implementation
===============================================================

Source: libs/fla/ops/kda/__init__.py

Perfect replication of official FLA KDA ops in pure PyTorch.
All Triton kernels are converted to PyTorch implementations.

Exported APIs:
- chunk_kda: Main entry for chunk-based KDA
- ChunkKDAFunction: torch.autograd.Function wrapper
- chunk_kda_fwd_intra / chunk_kda_bwd_intra: Intra-chunk attention
- recompute_w_u_fwd / prepare_wy_repr_bwd: WY representation
- chunk_kda_bwd_dqkwg: Inter-chunk backward gradients
- fused_recurrent_kda: Recurrent mode (Stage 3)
- fused_kda_gate: Gate function (Stage 4)
"""

# Stage 2.1: chunk_intra.py - Intra-chunk local attention
from .chunk_intra import chunk_kda_fwd_intra, chunk_kda_bwd_intra

# Stage 2.2: wy_fast.py - WY representation (Woodbury decomposition)
from .wy_fast import recompute_w_u_fwd, prepare_wy_repr_bwd

# Stage 2.3: chunk_inter.py - Inter-chunk backward
from .chunk_inter import chunk_kda_bwd_dqkwg

# Stage 2.5: chunk.py - Main entry
from .chunk import chunk_kda, ChunkKDAFunction

# Stage 3: fused_recurrent.py - Recurrent mode
from .fused_recurrent import fused_recurrent_kda

# Stage 4: gate.py - Gate function
from .gate import fused_kda_gate

# Stage 2.6: naive.py - Reference implementation (for testing)
from .naive import naive_chunk_kda, naive_recurrent_kda

__all__ = [
    'chunk_kda',
    'ChunkKDAFunction',
    'chunk_kda_fwd_intra',
    'chunk_kda_bwd_intra',
    'recompute_w_u_fwd',
    'prepare_wy_repr_bwd',
    'chunk_kda_bwd_dqkwg',
    'fused_recurrent_kda',
    'fused_kda_gate',
    'naive_chunk_kda',
    'naive_recurrent_kda',
]
