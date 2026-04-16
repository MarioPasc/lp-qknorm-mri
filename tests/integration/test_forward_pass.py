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
    def test_weights_transferred(self) -> None:
        """After patching, qkv, proj, and relative position bias in each
        LpWindowAttention must match the weights from the stock
        WindowAttention that build_swin_unetr_lp constructed internally.

        Strategy: build one stock model, snapshot its attention weights,
        then build a patched model from the same constructor and verify
        the patched model's attention weights are internally consistent
        (i.e., the patching copied weights from the stock model it created).
        """
        from monai.networks.nets import SwinUNETR
        from monai.networks.nets.swin_unetr import (
            WindowAttention as MonaiWindowAttention,
        )

        # Build a stock model and save its attention weights as reference.
        torch.manual_seed(123)
        stock = SwinUNETR(
            in_channels=1, out_channels=2, feature_size=24, spatial_dims=2
        )
        stock_weights: dict[str, dict[str, torch.Tensor]] = {}
        for name, mod in stock.named_modules():
            if isinstance(mod, MonaiWindowAttention):
                stock_weights[name] = {
                    "qkv.weight": mod.qkv.weight.clone(),
                    "proj.weight": mod.proj.weight.clone(),
                    "rpb": mod.relative_position_bias_table.data.clone(),
                }
                if mod.qkv.bias is not None:
                    stock_weights[name]["qkv.bias"] = mod.qkv.bias.clone()

        # Build patched model with the same seed so its internal stock
        # construction produces identical weights.
        torch.manual_seed(123)
        patched = build_swin_unetr_lp(**_build_kwargs(), lp_cfg=LpQKNormConfig(p=2.0))

        patched_attns = [
            (n, m)
            for n, m in patched.named_modules()
            if isinstance(m, LpWindowAttention)
        ]
        assert len(patched_attns) == len(stock_weights), (
            f"Module count mismatch: stock={len(stock_weights)}, patched={len(patched_attns)}"
        )

        for p_name, p_mod in patched_attns:
            s_name = p_name  # same path in the module tree
            assert s_name in stock_weights, f"No matching stock weights for {p_name}"
            ref = stock_weights[s_name]

            assert torch.equal(ref["qkv.weight"], p_mod.qkv.weight), (
                f"qkv.weight mismatch at {s_name}"
            )
            assert torch.equal(ref["proj.weight"], p_mod.proj.weight), (
                f"proj.weight mismatch at {s_name}"
            )
            assert torch.equal(
                ref["rpb"],
                p_mod.relative_position_bias_table.data,
            ), f"relative_position_bias_table mismatch at {s_name}"
            if "qkv.bias" in ref and p_mod.qkv.bias is not None:
                assert torch.equal(ref["qkv.bias"], p_mod.qkv.bias), (
                    f"qkv.bias mismatch at {s_name}"
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
