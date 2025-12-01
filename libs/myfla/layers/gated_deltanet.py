from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from myfla.layers.utils import get_unpad_data, index_first_axis, pad_input
from myfla.modules import RMSNorm, FusedRMSNormGated, ShortConvolution
from myfla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

# Conditional torch.compile for Python 3.8 / PyTorch < 2.0 compatibility
try:
    compile_fn = torch.compile
except AttributeError:
    def compile_fn(fn):
        return fn


@compile_fn
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


@compile_fn
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class GatedDeltaNet(nn.Module):
    """
    純 PyTorch 版 GatedDeltaNet（對齊 fla/layers/gated_deltanet.py 的介面）。
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2.0,
        head_dim: int = 256,
        num_heads: int = 4,
        num_v_heads: Optional[int] = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.mode = mode
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.allow_neg_eigval = allow_neg_eigval
        self.layer_idx = 0 if layer_idx is None else layer_idx

        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))

        if use_short_conv:
            self.q_conv = ShortConvolution(self.key_dim, conv_size, bias=conv_bias)
            self.k_conv = ShortConvolution(self.key_dim, conv_size, bias=conv_bias)
            self.v_conv = ShortConvolution(self.value_dim, conv_size, bias=conv_bias)
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. Do not disable it unless必要。",
            )
            self.q_conv = self.k_conv = self.v_conv = None

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.g_proj = None
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def _proj_with_optional_conv(
        self,
        proj: nn.Linear,
        conv: Optional[ShortConvolution],
        hidden_states: torch.Tensor,
        cache: Optional[torch.Tensor],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        projected = proj(hidden_states)
        if conv is None:
            return F.silu(projected), None
        return conv(projected, cache=cache, output_final_state=use_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional[Any]]:
        _ = output_attentions
        use_cache = bool(use_cache)
        batch_size, seq_len, _ = hidden_states.shape

        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, -seq_len:].unsqueeze(-1)
            hidden_states = hidden_states * mask

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if last_state is None:
            conv_cache_q = conv_cache_k = conv_cache_v = None
            recurrent_state = None
        else:
            conv_cache_q, conv_cache_k, conv_cache_v = last_state['conv_state']
            recurrent_state = last_state['recurrent_state']

        q, new_conv_q = self._proj_with_optional_conv(self.q_proj, self.q_conv, hidden_states, conv_cache_q, use_cache)
        k, new_conv_k = self._proj_with_optional_conv(self.k_proj, self.k_conv, hidden_states, conv_cache_k, use_cache)
        v, new_conv_v = self._proj_with_optional_conv(self.v_proj, self.v_conv, hidden_states, conv_cache_v, use_cache)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = torch.sigmoid(self.b_proj(hidden_states)).view(batch_size, seq_len, self.num_v_heads)
        if self.allow_neg_eigval:
            beta = beta * 2.0
        g = -torch.exp(self.A_log).view(1, 1, self.num_v_heads)
        g = g + F.softplus(self.a_proj(hidden_states) + self.dt_bias.view(1, 1, -1))

        if self.num_v_heads != self.num_heads:
            raise NotImplementedError("myfla GatedDeltaNet currently requires num_v_heads == num_heads")

        ops_fn = chunk_gated_delta_rule if (self.training or self.mode == 'chunk') else fused_recurrent_gated_delta_rule
        out, final_state = ops_fn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=None,
            initial_state=recurrent_state,
            output_final_state=use_cache,
        )

        if self.use_gate and self.g_proj is not None:
            gate = self.g_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_v_dim)
            normed = self.o_norm(out, gate)
        else:
            normed = self.o_norm(out)

        normed = normed.view(batch_size, seq_len, self.value_dim)
        result = self.o_proj(normed)
        if mask is not None:
            result = result * mask

        if past_key_values is not None or use_cache:
            conv_state = None
            if self.use_short_conv:
                conv_state = (new_conv_q, new_conv_k, new_conv_v)
            past_key_values.update(
                recurrent_state=final_state,
                conv_state=conv_state,
                layer_idx=self.layer_idx,
                offset=seq_len,
            )

        return result, None, past_key_values
