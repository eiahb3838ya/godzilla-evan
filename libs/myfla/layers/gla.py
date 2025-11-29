"""Pure PyTorch 版 Gated Linear Attention（GLA）層。

此檔案以 `libs/fla/layers/gla.py` 為基準進行移植，僅保留 PyTorch 張量運算。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from myfla.layers.utils import get_unpad_data, index_first_axis, pad_input
from myfla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution

# ✅ 使用 myfla 純 PyTorch 實作 (已完成移植, 2025-11-28)
from myfla.ops.gla import (
    chunk_gla,
    fused_chunk_gla,
    fused_recurrent_gla,
    gla_forward,
    select_gla_mode,
)

CacheType = Any


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


ACT2FN: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'relu': F.relu,
    'silu': F.silu,
    'swish': _swish,
    'gelu': F.gelu,
    'sigmoid': torch.sigmoid,
    'identity': _identity,
}


class GatedLinearAttention(nn.Module):
    """
    純 PyTorch 版 Gated Linear Attention。

    Args mirror `libs.fla.layers.gla.GatedLinearAttention`.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: Optional[int] = None,
        **_: Any,
    ) -> None:
        super().__init__()

        if gate_fn not in ACT2FN:
            raise ValueError(f'Unsupported gate_fn `{gate_fn}`')
        if feature_map is not None and feature_map not in ACT2FN:
            raise ValueError(f'Unsupported feature_map `{feature_map}`')
        if elementwise_affine is False:
            raise NotImplementedError('myfla RMSNorm 目前僅支援 elementwise_affine=True')

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.clamp_min = clamp_min
        self.layer_idx = 0 if layer_idx is None else layer_idx

        if mode not in {'chunk', 'fused_recurrent', 'fused_chunk'}:
            raise ValueError(f'Not supported mode `{mode}`.')
        if self.key_dim % num_heads != 0:
            raise ValueError(f'key dim must be divisible by num_heads ({num_heads})')
        if self.value_dim % num_heads != 0:
            raise ValueError(f'value dim must be divisible by num_heads ({num_heads})')

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        else:
            self.g_proj = None

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )

        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True),
        )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormGated(
                hidden_size=self.head_v_dim,
                elementwise_affine=True,
                eps=norm_eps,
            )
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, eps=norm_eps)
            self.gate_fn = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[CacheType] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, None, Optional[CacheType]]:
        _ = output_attentions
        use_cache = bool(use_cache)

        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(
                'attention_mask 必須為 [batch, seq_len]，且僅支援 padding mask (0/1)。'
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, 'b s ... -> (b s) ...'),
                indices,
            ).unsqueeze(0)

        conv_state_q = conv_state_k = conv_state_v = None
        if self.use_short_conv:
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k, gk = (
                repeat(x, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_k_dim)
                for x in (k, gk)
            )
            v = repeat(v, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_v_dim)
        else:
            k, gk = (rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim) for x in (k, gk))
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:  # pragma: no cover - mode 檢查已在 __init__
            raise NotImplementedError(f'Not supported mode `{mode}`.')

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, '... h d -> ... (h d)')
            else:
                o = rearrange(self.g_norm(o), '... h d -> ... (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), '... h d -> ... (h d)')
        o = self.o_proj(o)

        if attention_mask is not None and indices is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


__all__ = ['GatedLinearAttention']
