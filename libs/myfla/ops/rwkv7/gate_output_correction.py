from __future__ import annotations

import torch


def gate_output_correction(o, r, k, r_k, v, g):
    correction_term = ((r * k * r_k.unsqueeze(0).unsqueeze(0)).sum(-1, keepdim=True) * v).view(o.shape)
    return (o + correction_term) * g


class GateOutputCorrectionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, r, k, r_k, v, g):
        ctx.save_for_backward(o, r, k, r_k, v, g)
        return gate_output_correction(o, r, k, r_k, v, g)

    @staticmethod
    def backward(ctx, grad_output):
        o, r, k, r_k, v, g = ctx.saved_tensors
        B, T, HD = o.shape
        H, D = r.shape[-2], r.shape[-1]

        r_k_b = r_k.unsqueeze(0).unsqueeze(0)
        correction_scalar = (r * k * r_k_b).sum(-1, keepdim=True)
        gated_input = o + (correction_scalar * v).view(B, T, HD)

        grad_g = grad_output * gated_input
        grad_gate_input = grad_output * g

        grad_o = grad_gate_input
        grad_corr = grad_gate_input.view(B, T, H, D)
        grad_v = grad_corr * correction_scalar
        grad_corr_scalar = (grad_corr * v).sum(-1, keepdim=True)
        grad_r = grad_corr_scalar * k * r_k_b
        grad_k = grad_corr_scalar * r * r_k_b
        grad_r_k = (grad_corr_scalar * r * k).sum(dim=(0, 1))

        return grad_o, grad_r, grad_k, grad_r_k, grad_v, grad_g
