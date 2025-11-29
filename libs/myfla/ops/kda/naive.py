# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# -*- coding: utf-8 -*-

"""
Naive Reference Implementations for KDA - For Testing
=====================================================

Source: libs/fla/ops/kda/naive.py:L1-L155

Provides slow but readable reference implementations for testing.

Perfect replication requirements:
- API matches official naive implementations
- Used for numerical validation in tests
"""

import torch
from typing import Optional, Tuple


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Naive recurrent KDA implementation (token-by-token).
    
    Source: libs/fla/ops/kda/naive.py:L7-L58
    
    TODO: Stage 2.6 - Implement for testing
    """
    raise NotImplementedError(
        "naive_recurrent_kda: Stage 2.6 - Reference implementation for testing."
    )


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Naive chunk-based KDA implementation.
    
    Source: libs/fla/ops/kda/naive.py:L39-L155
    
    TODO: Stage 2.6 - Implement for testing
    """
    raise NotImplementedError(
        "naive_chunk_kda: Stage 2.6 - Reference implementation for testing."
    )

