# Source: libs/fla/layers/rwkv6.py (LoRA subset, version 0.4.0)
"""供 RWKV7Attention 使用的 LoRA 子模組（純 PyTorch 版）。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRA(nn.Module):
    """低秩適配層，逐行對齊 libs.fla.layers.rwkv6.LoRA。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool | None = True,
        activation: str | None = 'tanh',
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        if activation is None:
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Not supported activation `{activation}`.")

        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=False),
            self.activation,
            nn.Linear(low_rank_dim, output_dim, bias=bias),
        )
        try:
            from transformers.modeling_utils import _init_weights
        except ImportError:
            _init_weights = True
        if _init_weights:
            self.apply(self._initialize_weights)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"input_dim={self.input_dim}, low_rank_dim={self.low_rank_dim}, output_dim={self.output_dim}"
        if not self.bias:
            s += f", bias={self.bias}"
        s += ")"
        return s

    def _initialize_weights(self, module: nn.Module) -> None:
        if getattr(module, "_is_hf_initialized", False):
            return

        nn.init.zeros_(self.lora[0].weight)
        original_dtype = self.lora[2].weight.dtype
        shape = self.lora[2].weight.shape
        weight_fp32 = self.lora[2].weight.float()
        gain = math.sqrt(shape[1] / shape[0]) if shape[1] > shape[0] else 1
        nn.init.orthogonal_(weight_fp32, gain=gain * 0.1)
        self.lora[2].weight.data.copy_(weight_fp32.to(original_dtype))
        if self.lora[2].bias is not None:
            nn.init.zeros_(self.lora[2].bias)

        module._is_hf_initialized = True

    def set_bias_value(self, value):
        if self.bias and self.lora[2].bias is not None:
            if isinstance(value, torch.Tensor):
                self.lora[2].bias.data.copy_(value.to(self.lora[2].bias.dtype))
            else:
                nn.init.constant_(self.lora[2].bias, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


__all__ = ['LoRA']
