from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .rwkv6 import LoRA
from myfla.modules import GroupNorm
from myfla.modules.l2norm import l2_norm
from myfla.modules.token_shift import token_shift
from myfla.ops.rwkv7 import (
    chunk_rwkv7,
    fused_addcmul_rwkv7,
    fused_k_rwkv7,
    fused_mul_recurrent_rwkv7,
    gate_output_correction,
)

if TYPE_CHECKING:  # pragma: no cover
    from fla.models.utils import Cache
else:
    Cache = Any


class RWKV7Attention(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = None,
        decay_low_rank_dim: Optional[int] = None,
        gate_low_rank_dim: Optional[int] = None,
        a_low_rank_dim: Optional[int] = None,
        v_low_rank_dim: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: Optional[int] = None,
        fuse_norm: bool = False,
        value_dim: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mode = mode
        if mode not in {'chunk', 'fused_recurrent'}:
            raise ValueError(f"Not supported mode `{mode}`.")
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        if head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        else:
            assert num_heads is not None
            self.num_heads = num_heads
            self.head_dim = int(hidden_size // num_heads)
        if self.value_dim % self.num_heads != 0:
            raise ValueError("value_dim must be divisible by num_heads.")
        self.head_v_dim = int(self.value_dim // self.num_heads)

        factor = self.head_dim / 64.0

        def _auto_rank(multiplier: float, use_factor: bool = True) -> int:
            scale = multiplier * (hidden_size ** 0.5)
            if use_factor:
                scale *= factor
            rounded = int(round(scale / 32.0) * 32.0)
            return max(32, rounded)

        self.decay_low_rank_dim = (
            decay_low_rank_dim if decay_low_rank_dim is not None else _auto_rank(2.5)
        )
        self.gate_low_rank_dim = (
            gate_low_rank_dim if gate_low_rank_dim is not None else _auto_rank(5.0, use_factor=False)
        )
        self.a_low_rank_dim = (
            a_low_rank_dim if a_low_rank_dim is not None else _auto_rank(2.5)
        )
        self.v_low_rank_dim = (
            v_low_rank_dim if v_low_rank_dim is not None else _auto_rank(1.7)
        )

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.fuse_norm = fuse_norm

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=self.decay_low_rank_dim, activation='tanh')
        self.v_lora: Optional[LoRA]
        if self.layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=self.v_low_rank_dim, activation=None)
        else:
            self.v_lora = None
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=self.a_low_rank_dim, activation=None)
        self.g_lora = LoRA(
            hidden_size,
            self.value_dim,
            low_rank_dim=self.gate_low_rank_dim,
            activation='sigmoid',
            bias=False,
        )

        if self.fuse_norm:
            self.g_norm = GroupNorm(
                num_groups=self.num_heads,
                hidden_size=self.value_dim,
                elementwise_affine=bool(elementwise_affine),
                eps=self.head_dim * norm_eps,
                bias=True,
            )
        else:
            self.g_norm = nn.GroupNorm(
                num_groups=self.num_heads,
                num_channels=self.value_dim,
                eps=self.head_dim * norm_eps,
                affine=bool(elementwise_affine),
            )

        try:
            from transformers.modeling_utils import _init_weights  # type: ignore
        except ImportError:
            _init_weights = True
        if _init_weights:
            self.apply(self._initialize_weights)

        for _, module in self.named_modules():
            module._in_rwkv_module = True  # type: ignore[attr-defined]

        warnings.warn(
            "According to Bo, you are using a potentially buggy FLA implementation of RWKV. "
            "If you plan to report any numbers based on this implementation, we strongly recommend "
            "cross-checking with the official repo: https://github.com/BlinkDL/RWKV-LM. "
            "Bo may disagree with results reported from this version.",
        )

    @staticmethod
    def _orthogonal_init(weight: torch.Tensor, gain: float = 1.0) -> None:
        original_dtype = weight.dtype
        tmp = weight.float()
        nn.init.orthogonal_(tmp, gain=gain)
        weight.data.copy_(tmp.to(original_dtype))

    @torch.no_grad()
    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return

        if isinstance(module, RWKV7Attention) and self.layer_idx is not None and self.num_hidden_layers:
            denom = max(self.num_hidden_layers - 1, 1)
            ratio_0_to_1 = self.layer_idx / denom
            ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)

            ddd = torch.ones(1, 1, self.hidden_size, device=self.x_r.device)
            www = torch.zeros(self.hidden_size, device=self.x_r.device)
            zigzag = torch.zeros(self.hidden_size, device=self.x_r.device)
            linear = torch.zeros(self.hidden_size, device=self.x_r.device)

            for n in range(self.hidden_size):
                linear[n] = n / (self.hidden_size - 1) - 0.5
                zigzag[n] = ((n % self.head_dim) - ((self.head_dim - 1) / 2)) / ((self.head_dim - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (self.hidden_size - 1)) ** (1 + ratio_0_to_1 ** 0.3)
                ddd[0, 0, n] = n / self.hidden_size

            self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_r.dtype)
            self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_w.dtype)
            self.x_k.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_k.dtype)
            self.x_v.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_v.dtype)
            self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_a.dtype)
            self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_g.dtype)

            nn.init.constant_(self.k_a, 1.02)
            nn.init.constant_(self.r_k, -0.04)
            self.k_k.data.copy_((torch.zeros(self.hidden_size, device=self.k_k.device) + 0.71 - linear * 0.1).to(self.k_k.dtype))

            self.w_lora.set_bias_value(www + 0.5 + zigzag * 2.5)
            self.a_lora.set_bias_value(-0.19 + zigzag * 0.3 + linear * 0.4)

            if self.layer_idx != 0 and self.v_lora is not None:
                self.v_lora._initialize_weights(self.v_lora)
                self.v_lora.set_bias_value(0.73 - linear * 0.4)

            if hasattr(self.g_norm, 'weight') and self.g_norm.weight is not None:
                self.g_norm.weight.data[:] = ((self.layer_idx + 1) / self.num_hidden_layers) ** 0.7

            self._orthogonal_init(self.r_proj.weight)
            self._orthogonal_init(self.k_proj.weight, gain=0.1)
            self._orthogonal_init(self.v_proj.weight)
            self.o_proj.weight.data.zero_()

            del ddd, www, zigzag, linear

        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional[Any], Optional[torch.Tensor]]:
        _ = output_attentions  # unused
        use_cache = bool(use_cache)
        batch_size, seq_len, _ = hidden_states.shape

        am = None
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] for padding purposes."
                )
            if attention_mask.size(1) < seq_len:
                raise ValueError("attention_mask length must be >= seq_len.")
            am = attention_mask.narrow(1, attention_mask.size(1) - seq_len, seq_len).unsqueeze(-1)
            hidden_states = hidden_states * am

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if last_state is None:
            conv_cache = None
            recurrent_state = None
        else:
            conv_cache = last_state['conv_state']
            recurrent_state = last_state['recurrent_state']

        delta, conv_state = token_shift(hidden_states, cu_seqlens, output_cache=True, cache=conv_cache)
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(
            hidden_states, delta, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g,
        )

        r = self.r_proj(xr)
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()

        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0 or self.layer_idx is None:
            v_first = v
        elif self.v_lora is not None and v_first is not None:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())

        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        kk = rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim)
        if self.fuse_norm:
            kk = l2_norm(kk)
        else:
            kk = F.normalize(kk, dim=-1, p=2.0)

        k = fused_k_rwkv7(k, a, self.k_a)

        if am is not None:
            v = v * am

        r = rearrange(r, 'b t (h d) -> b t h d', d=self.head_dim)
        w = rearrange(w, 'b t (h d) -> b t h d', d=self.head_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.head_dim)
        a = rearrange(a, 'b t (h d) -> b t h d', d=self.head_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        if self.training or seq_len >= 64:
            o, recurrent_state = chunk_rwkv7(
                r=r,
                w=w,
                k=k,
                v=v,
                a=-kk,
                b=kk * a,
                scale=1.0,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_mul_recurrent_rwkv7(
                r=r,
                w=w,
                k=k,
                v=v,
                kk=kk,
                a=a,
                scale=1.0,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                layer_idx=self.layer_idx,
                offset=r.shape[1],
            )

        if self.fuse_norm:
            o = self.g_norm(rearrange(o, 'b t h d -> b t (h d)'))
        else:
            o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)

        o = gate_output_correction(o, r, k, self.r_k, v, g)
        o = self.o_proj(o)

        return o, None, past_key_values, v_first
