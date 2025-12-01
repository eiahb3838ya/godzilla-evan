"""
Unit tests for KDA gate operations.

Tests: libs/myfla/ops/kda/gate.py
Source: libs/fla/ops/kda/gate.py

Test coverage:
- kda_gate_ref: Reference implementation
- fused_kda_gate: User API with autograd
- Forward pass: shape, dtype, numerical correctness
- Backward pass: gradcheck, gradient flow
- Edge cases: with/without g_bias, with/without b
"""

import unittest

import torch
import torch.nn.functional as F
from einops import rearrange

from myfla.ops.kda.gate import (
    KDAGateFunction,
    fused_kda_gate,
    kda_gate_bwd,
    kda_gate_fwd,
    kda_gate_ref,
)


class TestKDAGateRef(unittest.TestCase):
    """Test kda_gate_ref reference implementation."""

    def setUp(self):
        torch.manual_seed(42)
        self.B, self.T, self.H, self.D = 2, 4, 2, 8
        self.HD = self.H * self.D

    def test_forward_shape_basic(self):
        """Test basic forward pass without optional parameters."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)

        g_out, b_out = kda_gate_ref(g, A, self.D)

        # Check output shapes
        self.assertEqual(g_out.shape, (self.B, self.T, self.H, self.D))
        self.assertIsNone(b_out)
        self.assertEqual(g_out.dtype, torch.float32)

    def test_forward_with_bias(self):
        """Test forward pass with g_bias."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)
        g_bias = torch.randn(self.HD, requires_grad=True)

        g_out, b_out = kda_gate_ref(g, A, self.D, g_bias=g_bias)

        self.assertEqual(g_out.shape, (self.B, self.T, self.H, self.D))
        self.assertIsNone(b_out)

    def test_forward_with_b(self):
        """Test forward pass with b parameter."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)
        b = torch.randn(self.B, self.T, self.H, requires_grad=True)

        g_out, b_out = kda_gate_ref(g, A, self.D, b=b)

        self.assertEqual(g_out.shape, (self.B, self.T, self.H, self.D))
        self.assertIsNotNone(b_out)
        self.assertEqual(b_out.shape, (self.B, self.T, self.H))
        self.assertEqual(b_out.dtype, torch.float32)

        # Check b_out is sigmoid(b)
        expected_b = torch.sigmoid(b.float())
        torch.testing.assert_close(b_out, expected_b, rtol=1e-5, atol=1e-5)

    def test_forward_all_parameters(self):
        """Test forward pass with all optional parameters."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)
        g_bias = torch.randn(self.HD, requires_grad=True)
        b = torch.randn(self.B, self.T, self.H, requires_grad=True)

        g_out, b_out = kda_gate_ref(g, A, self.D, g_bias=g_bias, b=b)

        self.assertEqual(g_out.shape, (self.B, self.T, self.H, self.D))
        self.assertEqual(b_out.shape, (self.B, self.T, self.H))

    def test_formula_correctness(self):
        """Test mathematical formula: g_out = -exp(A) * softplus(g)."""
        g = torch.randn(self.B, self.T, self.HD)
        A = torch.randn(self.H)

        g_out, _ = kda_gate_ref(g, A, self.D)

        # Manual computation
        A_flat = A.view(-1)
        g_reshaped = rearrange(g, '... (h d) -> ... h d', d=self.D)
        A_exp = -A_flat.float().exp().unsqueeze(-1)  # [H, 1]
        g_softplus = F.softplus(g_reshaped.float(), beta=1.0, threshold=20.0)
        expected = A_exp * g_softplus

        torch.testing.assert_close(g_out, expected, rtol=1e-5, atol=1e-5)

    def test_vllm_format(self):
        """Test vLLM format: [num_tokens, H*D]."""
        num_tokens = 10
        g = torch.randn(num_tokens, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)

        g_out, _ = kda_gate_ref(g, A, self.D)

        self.assertEqual(g_out.shape, (num_tokens, self.H, self.D))


class TestKDAGateForwardBackward(unittest.TestCase):
    """Test kda_gate_fwd and kda_gate_bwd."""

    def setUp(self):
        torch.manual_seed(42)
        self.B, self.T, self.H, self.D = 2, 4, 2, 8
        self.HD = self.H * self.D

    def test_forward_matches_ref(self):
        """Test kda_gate_fwd matches kda_gate_ref."""
        g = torch.randn(self.B, self.T, self.HD)
        A = torch.randn(self.H)
        g_bias = torch.randn(self.HD)
        b = torch.randn(self.B, self.T, self.H)

        # Forward with kda_gate_fwd
        g_out_fwd, b_out_fwd = kda_gate_fwd(g, A, self.D, g_bias, b)

        # Forward with reference
        g_out_ref, b_out_ref = kda_gate_ref(g, A, self.D, g_bias, b)

        # Should be identical (both use same implementation)
        torch.testing.assert_close(g_out_fwd, g_out_ref, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(b_out_fwd, b_out_ref, rtol=1e-6, atol=1e-6)

    def test_backward_basic(self):
        """Test backward pass basic functionality."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)

        # Forward
        g_out, _ = kda_gate_fwd(g, A, self.D)

        # Backward
        grad_output = torch.randn_like(g_out)
        dg, dA, dgbias, db = kda_gate_bwd(grad_output, g, A, self.D)

        # Check shapes
        self.assertEqual(dg.shape, g.shape)
        self.assertEqual(dA.shape, A.shape)
        self.assertIsNone(dgbias)
        self.assertIsNone(db)

    def test_backward_with_bias(self):
        """Test backward pass with g_bias."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)
        g_bias = torch.randn(self.HD, requires_grad=True)

        # Forward
        g_out, _ = kda_gate_fwd(g, A, self.D, g_bias=g_bias)

        # Backward
        grad_output = torch.randn_like(g_out)
        dg, dA, dgbias, db = kda_gate_bwd(grad_output, g, A, self.D, g_bias=g_bias)

        # Check shapes
        self.assertEqual(dg.shape, g.shape)
        self.assertEqual(dA.shape, A.shape)
        self.assertIsNotNone(dgbias)
        self.assertEqual(dgbias.shape, g_bias.shape)
        self.assertIsNone(db)

    def test_backward_with_b(self):
        """Test backward pass with b parameter."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)
        b = torch.randn(self.B, self.T, self.H, requires_grad=True)

        # Forward
        g_out, b_out = kda_gate_fwd(g, A, self.D, b=b)

        # Backward
        grad_output = torch.randn_like(g_out)
        gb = torch.randn_like(b_out)
        dg, dA, dgbias, db = kda_gate_bwd(grad_output, g, A, self.D, b=b, gb=gb)

        # Check shapes
        self.assertEqual(dg.shape, g.shape)
        self.assertEqual(dA.shape, A.shape)
        self.assertIsNone(dgbias)
        self.assertIsNotNone(db)
        self.assertEqual(db.shape, b.shape)


class TestKDAGateFunction(unittest.TestCase):
    """Test KDAGateFunction autograd wrapper."""

    def setUp(self):
        torch.manual_seed(42)
        self.B, self.T, self.H, self.D = 2, 4, 2, 8
        self.HD = self.H * self.D

    def test_autograd_basic(self):
        """Test autograd with basic parameters."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True, dtype=torch.float64)
        A = torch.randn(self.H, requires_grad=True, dtype=torch.float64)

        def func(g_in, A_in):
            out, _ = KDAGateFunction.apply(g_in, A_in, self.D, None, None, 1.0, 20.0)
            return out

        # Gradcheck
        self.assertTrue(
            torch.autograd.gradcheck(
                func,
                (g, A),
                eps=1e-3,
                atol=1e-2,
                check_undefined_grad=False,
            )
        )

    def test_autograd_with_bias(self):
        """Test autograd with g_bias."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True, dtype=torch.float64)
        A = torch.randn(self.H, requires_grad=True, dtype=torch.float64)
        g_bias = torch.randn(self.HD, requires_grad=True, dtype=torch.float64)

        def func(g_in, A_in, gb_in):
            out, _ = KDAGateFunction.apply(g_in, A_in, self.D, gb_in, None, 1.0, 20.0)
            return out

        # Gradcheck
        self.assertTrue(
            torch.autograd.gradcheck(
                func,
                (g, A, g_bias),
                eps=1e-3,
                atol=1e-2,
                check_undefined_grad=False,
            )
        )

    def test_backward_flow(self):
        """Test gradient flow through KDAGateFunction."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)
        g_bias = torch.randn(self.HD, requires_grad=True)

        # Forward
        g_out, _ = KDAGateFunction.apply(g, A, self.D, g_bias, None, 1.0, 20.0)

        # Backward
        loss = g_out.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(g.grad)
        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(g_bias.grad)

        # Check gradient shapes
        self.assertEqual(g.grad.shape, g.shape)
        self.assertEqual(A.grad.shape, A.shape)
        self.assertEqual(g_bias.grad.shape, g_bias.shape)


class TestFusedKDAGate(unittest.TestCase):
    """Test fused_kda_gate user API."""

    def setUp(self):
        torch.manual_seed(42)
        self.B, self.T, self.H, self.D = 2, 4, 2, 8
        self.HD = self.H * self.D

    def test_api_basic(self):
        """Test basic API usage."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)

        g_out = fused_kda_gate(g, A, self.D)

        # Check output shape
        self.assertEqual(g_out.shape, (self.B, self.T, self.H, self.D))
        self.assertIsInstance(g_out, torch.Tensor)

    def test_api_with_b(self):
        """Test API with b parameter returns tuple."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)
        b = torch.randn(self.B, self.T, self.H, requires_grad=True)

        result = fused_kda_gate(g, A, self.D, b=b)

        # Check returns tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        g_out, b_out = result
        self.assertEqual(g_out.shape, (self.B, self.T, self.H, self.D))
        self.assertEqual(b_out.shape, (self.B, self.T, self.H))

    def test_api_gradient_flow(self):
        """Test gradient flow through fused_kda_gate."""
        g = torch.randn(self.B, self.T, self.HD, requires_grad=True)
        A = torch.randn(self.H, requires_grad=True)
        g_bias = torch.randn(self.HD, requires_grad=True)
        b = torch.randn(self.B, self.T, self.H, requires_grad=True)

        # Forward
        g_out, b_out = fused_kda_gate(g, A, self.D, g_bias=g_bias, b=b)

        # Backward
        loss = g_out.sum() + b_out.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(g.grad)
        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(g_bias.grad)
        self.assertIsNotNone(b.grad)

    def test_api_matches_ref(self):
        """Test fused_kda_gate matches kda_gate_ref."""
        g = torch.randn(self.B, self.T, self.HD)
        A = torch.randn(self.H)
        g_bias = torch.randn(self.HD)
        b = torch.randn(self.B, self.T, self.H)

        # Using API
        g_out_api, b_out_api = fused_kda_gate(g, A, self.D, g_bias=g_bias, b=b)

        # Using reference
        g_out_ref, b_out_ref = kda_gate_ref(g, A, self.D, g_bias=g_bias, b=b)

        # Should match
        torch.testing.assert_close(g_out_api, g_out_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b_out_api, b_out_ref, rtol=1e-5, atol=1e-5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def setUp(self):
        torch.manual_seed(42)

    def test_single_head(self):
        """Test with single head."""
        B, T, D = 2, 4, 8
        H = 1
        HD = H * D

        g = torch.randn(B, T, HD, requires_grad=True)
        A = torch.randn(H, requires_grad=True)

        g_out = fused_kda_gate(g, A, D)

        self.assertEqual(g_out.shape, (B, T, H, D))

    def test_large_beta(self):
        """Test with large beta parameter."""
        B, T, H, D = 2, 4, 2, 8
        HD = H * D

        g = torch.randn(B, T, HD)
        A = torch.randn(H)

        # Large beta should make softplus steeper
        g_out_small = fused_kda_gate(g, A, D, beta=0.1)
        g_out_large = fused_kda_gate(g, A, D, beta=10.0)

        # Outputs should differ
        self.assertFalse(torch.allclose(g_out_small, g_out_large))

    def test_threshold_effect(self):
        """Test softplus threshold parameter."""
        B, T, H, D = 2, 4, 2, 8
        HD = H * D

        # Large positive values to trigger threshold
        g = torch.ones(B, T, HD) * 100.0
        A = torch.randn(H)

        g_out = fused_kda_gate(g, A, D, threshold=20.0)

        # Should not have NaN or Inf
        self.assertFalse(torch.isnan(g_out).any())
        self.assertFalse(torch.isinf(g_out).any())

    def test_different_A_shapes(self):
        """Test different A parameter shapes."""
        B, T, H, D = 2, 4, 2, 8
        HD = H * D
        g = torch.randn(B, T, HD)

        # Test [H]
        A1 = torch.randn(H)
        g_out1 = fused_kda_gate(g, A1, D)

        # Test [1, 1, H, 1]
        A2 = torch.randn(1, 1, H, 1)
        g_out2 = fused_kda_gate(g, A2, D)

        # Shapes should be same
        self.assertEqual(g_out1.shape, g_out2.shape)


if __name__ == '__main__':
    unittest.main()
