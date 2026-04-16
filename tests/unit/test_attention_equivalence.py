"""Unit tests for LpWindowAttention equivalence and correctness.

Tests cover:
- LpWindowAttention produces valid attention weights (rows sum to 1).
- Output shape matches input shape.
- The _capture dict is populated with all expected keys after forward.
- No NaN outputs for various p values.
- p=2 LpWindowAttention produces the same normalized Q as Henry et al. reference.
"""

from __future__ import annotations

import torch

from lpqknorm.models.attention import LpWindowAttention
from lpqknorm.models.lp_qknorm import LpQKNormConfig


class TestLpWindowAttentionUnit:
    """Unit tests that exercise LpWindowAttention in isolation (no SwinUNETR)."""

    def _make_module(self, p: float = 3.0) -> LpWindowAttention:
        return LpWindowAttention(
            dim=24,
            num_heads=3,
            window_size=(7, 7),
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            lp_cfg=LpQKNormConfig(p=p),
        )

    def _make_input(self, batch_windows: int = 4) -> torch.Tensor:
        """Create input of shape (B*nW, n, C) = (batch_windows, 49, 24)."""
        torch.manual_seed(42)
        return torch.randn(batch_windows, 49, 24)

    def test_output_shape(self) -> None:
        module = self._make_module()
        x = self._make_input()
        out = module(x, mask=None)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_attention_rows_sum_to_one(self) -> None:
        module = self._make_module()
        module.eval()
        x = self._make_input()
        with torch.inference_mode():
            _ = module(x, mask=None)
        attn = module._capture["attention"]
        row_sums = attn.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-5,
            rtol=0.0,
        )

    def test_capture_dict_has_all_keys(self) -> None:
        module = self._make_module()
        x = self._make_input()
        _ = module(x, mask=None)
        expected_keys = {"q", "k", "q_hat", "k_hat", "alpha", "logits", "attention"}
        assert expected_keys.issubset(module._capture.keys()), (
            f"Missing keys: {expected_keys - module._capture.keys()}"
        )

    def test_capture_shapes_consistent(self) -> None:
        module = self._make_module()
        x = self._make_input(batch_windows=2)
        _ = module(x, mask=None)
        cap = module._capture
        # q, k, q_hat, k_hat: (B*nW, num_heads, n, head_dim)
        assert cap["q"].shape == cap["k"].shape
        assert cap["q_hat"].shape == cap["q"].shape
        assert cap["k_hat"].shape == cap["k"].shape
        # logits, attention: (B*nW, num_heads, n, n) — square
        assert cap["logits"].shape[-1] == cap["logits"].shape[-2]
        assert cap["attention"].shape == cap["logits"].shape
        # alpha: scalar
        assert cap["alpha"].shape == torch.Size([])

    def test_no_nan_output(self) -> None:
        for p in [1.5, 2.0, 3.0, 4.0]:
            module = self._make_module(p=p)
            x = self._make_input()
            out = module(x, mask=None)
            assert not torch.isnan(out).any(), f"NaN in output for p={p}"

    def test_p2_normalized_q_matches_henry_reference(self) -> None:
        """At p=2 the normalized Q inside LpWindowAttention must match
        the Henry et al. reference: q / (||q||_2 + eps)."""
        eps = 1e-6
        module = self._make_module(p=2.0)
        module.eval()
        x = self._make_input()
        with torch.inference_mode():
            _ = module(x, mask=None)
        q = module._capture["q"]
        q_hat = module._capture["q_hat"]
        q_hat_ref = q / (q.norm(p=2, dim=-1, keepdim=True) + eps)
        torch.testing.assert_close(q_hat, q_hat_ref, atol=1e-5, rtol=0.0)

    def test_with_mask(self) -> None:
        """Forward pass with a non-None mask must not error and must produce
        valid attention weights."""
        module = self._make_module()
        module.eval()
        n = 49
        # Simulate 2 windows worth of inputs with a mask.
        # nW=2, B=1, so b = B*nW = 2.
        x = torch.randn(2, n, 24)
        # Mask shape: (nW, n, n) — zero for allowed, large negative for blocked.
        mask = torch.zeros(2, n, n)
        mask[1, :, n // 2 :] = -100.0  # block second half in window 1
        with torch.inference_mode():
            out = module(x, mask=mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gradient_flows_through_module(self) -> None:
        """Backward pass must not error and gradients must be non-zero."""
        module = self._make_module(p=3.0)
        x = torch.randn(2, 49, 24, requires_grad=True)
        out = module(x, mask=None)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any(), "Gradients should be non-zero"

    def test_relative_position_bias_table_shape(self) -> None:
        module = self._make_module()
        # For window_size=(7,7): table shape = ((2*7-1)*(2*7-1), num_heads) = (169, 3)
        assert module.relative_position_bias_table.shape == (169, 3)

    def test_relative_position_index_shape(self) -> None:
        module = self._make_module()
        # For window_size=(7,7): index shape = (49, 49)
        assert module.relative_position_index.shape == (49, 49)
