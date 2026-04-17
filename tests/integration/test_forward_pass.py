"""Integration tests for the full SwinUNETR + Lp-QKNorm pipeline.

Tests cover:
- Drop-in shape compatibility: patched model output == stock model output shape.
- Weight transfer integrity: qkv, proj, and relative position bias tables match.
- Hook capture correctness: stage-0 hooks capture all expected fields.
- Vanilla baseline: lp_cfg=None returns stock model with no LpWindowAttention.
- All attentions replaced: no WindowAttention instances remain after patching.

All tests are marked @pytest.mark.integration per project convention.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from lpqknorm.models.attention import LpWindowAttention
from lpqknorm.models.hooks import AttentionHookRegistry
from lpqknorm.models.lp_qknorm import LpQKNormConfig
from lpqknorm.models.swin_unetr_lp import build_swin_unetr_lp


def _build_kwargs() -> dict[str, Any]:
    return {
        "img_size": (224, 224),
        "in_channels": 1,
        "out_channels": 2,
        "feature_size": 24,
    }


def _count_modules(model: nn.Module, cls: type) -> int:
    return sum(1 for _, m in model.named_modules() if isinstance(m, cls))


@pytest.mark.integration
class TestDropInShapeCompatibility:
    def test_attention_shape_preserved(self) -> None:
        """Patched model must produce the same output shape as stock MONAI."""
        from monai.networks.nets import SwinUNETR

        stock = SwinUNETR(
            in_channels=1, out_channels=2, feature_size=24, spatial_dims=2
        )
        lp_model = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=LpQKNormConfig(p=3.0))
        x = torch.randn(2, 1, 224, 224)
        with torch.inference_mode():
            out_stock = stock(x)
            out_lp = lp_model(x)
        assert out_stock.shape == out_lp.shape, (
            f"Shape mismatch: stock={out_stock.shape}, lp={out_lp.shape}"
        )

    def test_single_batch_forward(self) -> None:
        """Single-image batch must work without error."""
        model = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=LpQKNormConfig(p=2.5))
        x = torch.randn(1, 1, 224, 224)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1, 2, 224, 224)
        assert not torch.isnan(out).any()


@pytest.mark.integration
class TestWeightTransferIntegrity:
    def test_copy_weights_transfers_stock_tensors(self) -> None:
        """The ``_copy_weights`` helper used during patching must transfer
        qkv / proj / relative-position-bias tensors byte-identically.

        Historically this test verified that the built patched model had the
        same weights as the stock model it was constructed from.  That
        invariant no longer holds because ``build_swin_unetr_lp`` applies the
        Phase-2 weight-initialization spec after patching, deliberately
        overwriting the copied tensors.  The structural contract of the
        copy step itself is still tested here.
        """
        from monai.networks.nets import SwinUNETR
        from monai.networks.nets.swin_unetr import (
            WindowAttention as MonaiWindowAttention,
        )

        from lpqknorm.models.swin_unetr_lp import _copy_weights

        torch.manual_seed(123)
        stock = SwinUNETR(
            in_channels=1, out_channels=2, feature_size=24, spatial_dims=2
        )
        stock_attns = [
            (n, m)
            for n, m in stock.named_modules()
            if isinstance(m, MonaiWindowAttention)
        ]
        assert stock_attns, "no stock WindowAttention modules found"

        for _, stock_attn in stock_attns:
            lp_attn = LpWindowAttention(
                dim=stock_attn.dim,
                num_heads=stock_attn.num_heads,
                window_size=stock_attn.window_size,
                qkv_bias=stock_attn.qkv.bias is not None,
                attn_drop=stock_attn.attn_drop.p,
                proj_drop=stock_attn.proj_drop.p,
                lp_cfg=LpQKNormConfig(p=2.0),
            )
            _copy_weights(stock_attn, lp_attn)

            assert torch.equal(stock_attn.qkv.weight, lp_attn.qkv.weight)
            assert torch.equal(stock_attn.proj.weight, lp_attn.proj.weight)
            assert torch.equal(
                stock_attn.relative_position_bias_table.data,
                lp_attn.relative_position_bias_table.data,
            )
            if stock_attn.qkv.bias is not None:
                assert torch.equal(stock_attn.qkv.bias, lp_attn.qkv.bias)

    def test_build_swin_unetr_lp_preserves_module_topology(self) -> None:
        """Every stock WindowAttention slot must be occupied by an
        LpWindowAttention with matching dim / num_heads / window_size /
        qkv_bias, even after the post-patching re-initialization."""
        from monai.networks.nets import SwinUNETR
        from monai.networks.nets.swin_unetr import (
            WindowAttention as MonaiWindowAttention,
        )

        torch.manual_seed(123)
        stock = SwinUNETR(
            in_channels=1, out_channels=2, feature_size=24, spatial_dims=2
        )
        stock_shapes = {
            n: (m.dim, m.num_heads, tuple(m.window_size), m.qkv.bias is not None)
            for n, m in stock.named_modules()
            if isinstance(m, MonaiWindowAttention)
        }

        torch.manual_seed(123)
        patched = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=LpQKNormConfig(p=2.0))
        patched_shapes = {
            n: (m.dim, m.num_heads, tuple(m.window_size), m.qkv.bias is not None)
            for n, m in patched.named_modules()
            if isinstance(m, LpWindowAttention)
        }
        assert stock_shapes == patched_shapes, (
            "Module topology drift after patching: "
            f"stock keys={set(stock_shapes)}, patched keys={set(patched_shapes)}"
        )


@pytest.mark.integration
class TestHookCapture:
    def test_hooks_capture_expected_tensors(self) -> None:
        """Hook captures at stage 0 must have all fields and correct norms."""
        model = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=LpQKNormConfig(p=3.0))
        registry = AttentionHookRegistry()
        registry.register(model, stages=[0])
        x = torch.randn(1, 1, 224, 224)
        with torch.inference_mode():
            _ = model(x)
        captures = registry.captures()

        assert len(captures) >= 1, "Expected at least one capture from stage 0"

        for c in captures:
            assert c.q is not None, "q must not be None"
            assert c.k is not None, "k must not be None"
            assert c.q_hat is not None, "q_hat must not be None"
            assert c.k_hat is not None, "k_hat must not be None"
            assert c.logits is not None, "logits must not be None"
            assert c.attention is not None, "attention must not be None"
            assert c.alpha is not None, "alpha must not be None"

            assert c.q.shape == c.k.shape
            assert c.q_hat.shape == c.q.shape
            assert c.attention.shape[-1] == c.attention.shape[-2]  # square

            # Verify Lp-norm of q_hat is approximately 1
            norm = c.q_hat.abs().pow(3.0).sum(-1).pow(1.0 / 3.0)
            torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-3, rtol=0.0)

            assert c.stage_index == 0
            assert c.block_index >= 0

        registry.remove()

    def test_hooks_multiple_stages(self) -> None:
        """Hooks registered on stages [0, 1] must capture from both."""
        model = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=LpQKNormConfig(p=2.5))
        registry = AttentionHookRegistry()
        registry.register(model, stages=[0, 1])
        x = torch.randn(1, 1, 224, 224)
        with torch.inference_mode():
            _ = model(x)
        captures = registry.captures()

        stages_seen = {c.stage_index for c in captures}
        assert 0 in stages_seen, "Expected captures from stage 0"
        assert 1 in stages_seen, "Expected captures from stage 1"

        registry.remove()

    def test_clear_resets_captures(self) -> None:
        """clear() must empty captures without removing hooks."""
        model = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=LpQKNormConfig(p=3.0))
        registry = AttentionHookRegistry()
        registry.register(model, stages=[0])
        x = torch.randn(1, 1, 224, 224)
        with torch.inference_mode():
            _ = model(x)
        assert len(registry.captures()) > 0

        registry.clear()
        assert len(registry.captures()) == 0

        # Hooks still active: running forward again should produce new captures.
        with torch.inference_mode():
            _ = model(x)
        assert len(registry.captures()) > 0

        registry.remove()


@pytest.mark.integration
class TestVanillaBaseline:
    def test_vanilla_baseline_returns_stock(self) -> None:
        """lp_cfg=None must return a model with no LpWindowAttention modules."""
        model = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=None)
        n_lp = _count_modules(model, LpWindowAttention)
        assert n_lp == 0, f"Expected 0 LpWindowAttention, found {n_lp}"

    def test_vanilla_baseline_forward(self) -> None:
        """Vanilla baseline must produce valid output."""
        model = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=None)
        x = torch.randn(1, 1, 224, 224)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1, 2, 224, 224)
        assert not torch.isnan(out).any()


@pytest.mark.integration
class TestAllAttentionsReplaced:
    def test_no_window_attention_remains(self) -> None:
        """After patching, no stock WindowAttention should remain."""
        from monai.networks.nets.swin_unetr import (
            WindowAttention as MonaiWindowAttention,
        )

        model = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=LpQKNormConfig(p=3.0))
        n_stock = _count_modules(model, MonaiWindowAttention)
        n_lp = _count_modules(model, LpWindowAttention)
        assert n_stock == 0, f"Found {n_stock} unpatched WindowAttention module(s)"
        assert n_lp == 8, f"Expected 8 LpWindowAttention modules, found {n_lp}"
