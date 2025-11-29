# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# -*- coding: utf-8 -*-

"""
KDA Chunk Mode - Main Entry - PyTorch Implementation
====================================================

Source: libs/fla/ops/kda/chunk.py:L1-L357

This is the main entry point for KDA chunk-based attention.

Components:
- chunk_kda_fwd: Forward logic orchestration
- chunk_kda_bwd: Backward logic orchestration
- ChunkKDAFunction: torch.autograd.Function wrapper
- chunk_kda: User-facing API

Perfect replication requirements:
- API matches official chunk_kda exactly
- Support all features: varlen, initial_state, L2norm, etc.
- torch.compiler.disable decorator for Python 3.8 compatibility
"""

import torch
from typing import Optional, Tuple

# Stage 1 dependencies (already implemented)
from myfla.modules.l2norm import l2norm_fwd, l2norm_bwd
from myfla.ops.common.chunk_delta_rule import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_bwd_dhu,
)
from myfla.ops.common.chunk_o import chunk_bwd_dv_local
from myfla.ops.utils import chunk_local_cumsum
from myfla.ops.kda.chunk_intra import chunk_kda_fwd_intra, chunk_kda_bwd_intra
from myfla.ops.kda.chunk_inter import chunk_kda_bwd_dqkwg
from myfla.ops.kda.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd

# Stage 2.4: GLA dependencies（PyTorch 版，待 varlen 支援）
from myfla.ops.gla import chunk_gla_bwd_dA, chunk_gla_fwd_o_gk
from myfla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    KDA forward pass (chunk mode).
    
    Source: libs/fla/ops/kda/chunk.py:L17-L69
    """
    chunk_size = 64
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    Aqk, Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    w, u, _, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        gk=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g, o, Aqk, Akk, final_state


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    KDA backward pass (chunk mode).
    
    Source: libs/fla/ops/kda/chunk.py:L72-L176
    
    Returns:
        dq, dk, dv, db, dg, dh0
    """
    chunk_size = 64
    w, u, qg, kg = recompute_w_u_fwd(
        q=q,
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        gk=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    dAqk = chunk_gla_bwd_dA(
        v=v_new,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk, dw, dg = chunk_kda_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dk2, dv, db, dg2, dAkk = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        gk=g,
        A=Akk,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
    )
    dq, dk2, db, dg2 = chunk_kda_bwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk2,
        db=db,
        dg=dg2,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    return dq, dk, dv, db, dg, dh0


class ChunkKDAFunction(torch.autograd.Function):
    """
    torch.autograd.Function wrapper for KDA chunk mode.
    
    Source: libs/fla/ops/kda/chunk.py:L179-L244
    """
    
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g, o, Aqk, Akk, final_state = chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, g, beta, Aqk, Akk)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: Optional[torch.Tensor],
    ) -> Tuple:
        q, k, v, g, beta, Aqk, Akk = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_kda_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens=ctx.cu_seqlens,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg.to(g.dtype),
            db.to(beta.dtype),
            None,
            dh0,
            None,
            None,
            None,
        )


# Python 3.8 compatibility: conditional decorator for torch.compiler.disable
try:
    _compiler_disable = torch.compiler.disable
except AttributeError:
    def _compiler_disable(fn):
        return fn


@_compiler_disable
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    KDA chunk-based attention (main entry).
    
    Source: libs/fla/ops/kda/chunk.py:L247-L356
    
    Args:
        q (torch.Tensor): queries of shape [B, T, H, K]
        k (torch.Tensor): keys of shape [B, T, H, K]
        v (torch.Tensor): values of shape [B, T, H, V]
        g (torch.Tensor): (forget) gating tensor (in log space!) of shape [B, T, H, K]
        beta (torch.Tensor): betas of shape [B, T, H]
        scale (Optional[float]): Scale factor. Default: 1 / sqrt(K)
        initial_state (Optional[torch.Tensor]): Initial state [N, H, K, V]
        output_final_state (Optional[bool]): Whether to output final state
        use_qk_l2norm_in_kernel (bool): Whether to apply L2norm to q,k
        cu_seqlens (torch.LongTensor): Cumulative sequence lengths for varlen
    
    Returns:
        o (torch.Tensor): Outputs of shape [B, T, H, V]
        final_state (torch.Tensor): Final state [N, H, K, V] if output_final_state=True
    
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                "Batch size must be 1 when using `cu_seqlens`. Please flatten sequences manually."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "Number of initial states must match number of sequences described by cu_seqlens."
            )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise AssertionError("initial_state 必須為 float32，以符合官方實作。")
    if q.shape != k.shape or q.shape != g.shape:
        raise AssertionError("q, k, g shape mismatch.")
    if beta.shape != q.shape[:3]:
        raise AssertionError("beta shape 必須為 [B, T, H].")
    if v.shape[:3] != q.shape[:3]:
        raise AssertionError("v shape 必須符合 [B, T, H, V].")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state
