from __future__ import annotations

import torch


def fused_addcmul_rwkv7(hidden_states: torch.Tensor, delta: torch.Tensor,
                        x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor,
                        x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor | None):
    """PyTorch fallback for libs.fla.ops.rwkv7.fused_addcmul."""
    def _fma(param: torch.Tensor) -> torch.Tensor:
        return hidden_states + delta * param

    xr = _fma(x_r)
    xw = _fma(x_w)
    xk = _fma(x_k)
    xv = _fma(x_v)
    xa = _fma(x_a)
    xg = _fma(x_g) if x_g is not None else xr
    return xr, xw, xk, xv, xa, xg
