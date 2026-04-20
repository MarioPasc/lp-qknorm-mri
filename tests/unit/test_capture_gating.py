"""Tests covering the capture-gating memory fix.

Before the fix, ``LpWindowAttention.forward`` unconditionally stored
references to ``q, k, q_hat, k_hat, logits, attention, relative_position_bias``
in ``self._capture``.  Those references pinned intermediate activations
(and their autograd sub-graph) past ``backward()``, effectively doubling
attention memory during training.  The fix:

- ``LpWindowAttention`` gains a ``_capture_enabled`` flag (default True so
  standalone unit tests keep working).
- :func:`lpqknorm.models.attention.set_capture_enabled` flips the flag on
  every ``LpWindowAttention`` inside a model.
- :class:`~lpqknorm.models.hooks.AttentionHookRegistry` arms the flag
  during :meth:`register` and disarms it during :meth:`remove`, so probes
  and callbacks can re-enable captures locally.

This test module verifies each of those invariants and also computes a
rough bound on the activation memory saved by disabling capture on a
realistic windowed shape.
"""

from __future__ import annotations

import gc

import torch

from lpqknorm.models.attention import LpWindowAttention, set_capture_enabled
from lpqknorm.models.hooks import AttentionHookRegistry
from lpqknorm.models.lp_qknorm import LpQKNormConfig


def _make_module(p: float = 3.0) -> LpWindowAttention:
    return LpWindowAttention(
        dim=24,
        num_heads=3,
        window_size=(7, 7),
        qkv_bias=True,
        lp_cfg=LpQKNormConfig(p=p),
    )


def _input(batch_windows: int = 4) -> torch.Tensor:
    return torch.randn(batch_windows, 49, 24)


class TestCaptureEnabledDefault:
    """Default-on behaviour preserves the standalone test semantics."""

    def test_capture_populated_by_default(self) -> None:
        module = _make_module()
        _ = module(_input(), mask=None)
        for key in ("q", "k", "q_hat", "k_hat", "logits", "attention", "alpha"):
            assert key in module._capture, f"missing {key!r} when enabled"


class TestCaptureDisabled:
    """With capture disabled, the forward pass stores no intermediate refs."""

    def test_disabled_capture_keeps_dict_empty(self) -> None:
        module = _make_module()
        set_capture_enabled(module, False)
        _ = module(_input(), mask=None)
        assert module._capture == {}, (
            "When _capture_enabled is False, forward() must leave _capture "
            f"empty, got keys: {list(module._capture)}"
        )

    def test_set_capture_enabled_counts_modules_touched(self) -> None:
        """Helper must return the number of LpWindowAttention instances it saw."""
        import torch.nn as nn

        container = nn.Sequential(
            _make_module(),
            nn.Linear(24, 24),
            _make_module(),
        )
        touched = set_capture_enabled(container, False)
        assert touched == 2
        assert all(
            (not m._capture_enabled)
            for m in container
            if isinstance(m, LpWindowAttention)
        )

    def test_backward_works_with_capture_disabled(self) -> None:
        """Training path: capture off, gradients still flow through the module."""
        module = _make_module(p=2.5)
        set_capture_enabled(module, False)
        x = _input(batch_windows=2).requires_grad_(True)
        y = module(x, mask=None)
        y.square().mean().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestHookRegistryToggle:
    """AttentionHookRegistry arms captures and disarms them on remove()."""

    def test_register_enables_capture_and_remove_restores(self) -> None:
        import torch.nn as nn

        # Build a faux model mirroring the attribute layout of SwinUNETR's
        # encoder: ``swinViT.layers1[0].blocks[*].attn``.
        attn_a = _make_module()
        attn_b = _make_module()

        class _Block(nn.Module):
            def __init__(self, attn: LpWindowAttention) -> None:
                super().__init__()
                self.attn = attn

        class _Layer(nn.Module):
            def __init__(self, blocks: list[_Block]) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(blocks)

        class _SwinViT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers1 = nn.ModuleList([_Layer([_Block(attn_a), _Block(attn_b)])])

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.swinViT = _SwinViT()

        model = _Model()
        set_capture_enabled(model, False)  # simulate training default

        assert not attn_a._capture_enabled
        assert not attn_b._capture_enabled

        registry = AttentionHookRegistry()
        registry.register(model, stages=[0])

        assert attn_a._capture_enabled, "register() must arm captures"
        assert attn_b._capture_enabled

        # Fire a forward so captures populate and the hook fires.
        _ = attn_a(_input(), mask=None)
        _ = attn_b(_input(), mask=None)
        caps = registry.captures()
        assert len(caps) == 2
        assert caps[0].attention is not None

        registry.remove()
        assert not attn_a._capture_enabled, "remove() must disarm captures"
        assert not attn_b._capture_enabled
        assert attn_a._capture == {}, "remove() must clear stale captures"


class TestActivationMemoryBound:
    """Rough sanity bound on how much memory capture=False saves.

    We cannot measure GPU memory from this unit test, but we can count
    tensor elements that would otherwise be pinned on a realistic Swin
    stage-0 windowed batch.  This doubles as documentation for the
    activation-memory delta.
    """

    def test_capture_tensor_elements_zero_when_disabled(self) -> None:
        # Realistic stage-0 shapes for 224^2 input, feature_size=24,
        # patch_size=2 -> 112^2 feature map -> 16x16=256 windows of 49
        # tokens, head_dim=8, num_heads=3.  Reduce to a small multiplier
        # that still exercises the shape; the bound is linear.
        b_nw, heads, n, head_dim = 16, 3, 49, 8
        module = _make_module()
        x = torch.randn(b_nw, n, head_dim * heads)

        # Baseline: capture enabled (default).
        _ = module(x, mask=None)
        baseline_elements = sum(
            v.numel() for v in module._capture.values() if isinstance(v, torch.Tensor)
        )
        assert baseline_elements > 0

        # Disabled: zero pinned elements.
        set_capture_enabled(module, False)
        gc.collect()
        _ = module(x, mask=None)
        disabled_elements = sum(
            v.numel() for v in module._capture.values() if isinstance(v, torch.Tensor)
        )
        assert disabled_elements == 0

        # Document the delta in the assertion message for future maintainers.
        delta_bytes_fp32 = baseline_elements * 4
        assert baseline_elements > 2 * b_nw * heads * n * n, (
            f"Expected at least one (logits) + one (attention) tensor "
            f"of shape ({b_nw},{heads},{n},{n}) pinned. "
            f"Baseline pinned {baseline_elements} elements "
            f"(~{delta_bytes_fp32 / 1024:.1f} KiB in fp32)."
        )
