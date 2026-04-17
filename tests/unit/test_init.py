"""Unit tests for the Phase-2 weight-initialization spec.

Covers:

1. Default init runs end-to-end (tiny model).
2. Empirical std of ``nn.Linear`` weights close to ``linear_init_std``.
3. ``nn.LayerNorm`` defaults (weight=1, bias=0).
4. Per-stage ``alpha_raw`` target matches ``log(d_k)`` within ``1e-6``.
5. Determinism across ``p`` values: shared tensors byte-identical.
6. Forward-pass smoke test at 16x1x224x224.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lpqknorm.models.attention import LpWindowAttention
from lpqknorm.models.init import (
    initialize_model,
    softplus_inverse,
)
from lpqknorm.models.lp_qknorm import LpQKNorm, LpQKNormConfig
from lpqknorm.models.swin_unetr_lp import build_swin_unetr_lp
from lpqknorm.utils.exceptions import LpInitError


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _build(
    p: float | None,
    *,
    feature_size: int = 12,
    alpha_scheme: str = "log_dk",
    alpha_fixed: float | None = None,
    linear_std: float = 0.02,
    seed: int = 20260417,
) -> nn.Module:
    """Seed RNG and build a tiny patched SwinUNETR for unit tests."""
    torch.manual_seed(seed)
    lp_cfg = LpQKNormConfig(p=p) if p is not None else None
    return build_swin_unetr_lp(
        img_size=(224, 224),
        in_channels=1,
        out_channels=1,
        feature_size=feature_size,
        lp_cfg=lp_cfg,
        init_scheme="scratch_trunc_normal",
        linear_init_std=linear_std,
        alpha_init_scheme=alpha_scheme,  # type: ignore[arg-type]
        alpha_init_fixed=alpha_fixed,
    )


# ---------------------------------------------------------------------------
# softplus_inverse
# ---------------------------------------------------------------------------


class TestSoftplusInverse:
    def test_inverts_softplus(self) -> None:
        """softplus(softplus_inverse(x)) ≈ x for x in the main regime."""
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 19.5]:
            y = softplus_inverse(x)
            recovered = float(F.softplus(torch.tensor(y)).item())
            assert abs(recovered - x) < 1e-5, f"x={x}: recovered={recovered}"

    def test_stable_branch_large_x(self) -> None:
        """For x > 20 the stable branch avoids overflow."""
        y = softplus_inverse(25.0)
        recovered = float(F.softplus(torch.tensor(y)).item())
        assert math.isfinite(y)
        assert abs(recovered - 25.0) < 1e-4

    def test_raises_on_nonpositive(self) -> None:
        with pytest.raises(LpInitError):
            softplus_inverse(0.0)
        with pytest.raises(LpInitError):
            softplus_inverse(-1.0)


# ---------------------------------------------------------------------------
# Acceptance test 1: default init runs end-to-end
# ---------------------------------------------------------------------------


def test_default_init_runs_end_to_end() -> None:
    """AT1: tiny model builds with defaults and has no NaN/Inf in any param."""
    model = _build(p=3.0, feature_size=12)
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), (
            f"parameter '{name}' contains NaN/Inf: "
            f"n_nonfinite={(~torch.isfinite(param)).sum().item()}"
        )


# ---------------------------------------------------------------------------
# Acceptance test 2: linear-weight empirical std
# ---------------------------------------------------------------------------


def test_linear_weight_empirical_std() -> None:
    """AT2: each nn.Linear (>=1024 elements) has weight.std() ~ 0.02."""
    model = _build(p=3.0, feature_size=24)
    any_checked = False
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.numel() >= 1024:
            any_checked = True
            s = float(m.weight.detach().std().item())
            # trunc_normal_(std=0.02) has empirical std slightly < 0.02 due
            # to truncation; 0.004 tolerance covers sampling + truncation bias.
            assert abs(s - 0.02) < 0.004, f"Linear std={s}, expected ~0.02"
    assert any_checked, "no nn.Linear with >=1024 elements found"


# ---------------------------------------------------------------------------
# Acceptance test 3: LayerNorm defaults
# ---------------------------------------------------------------------------


def test_layernorm_defaults() -> None:
    """AT3: every nn.LayerNorm has weight==1 and bias==0."""
    model = _build(p=3.0, feature_size=12)
    any_checked = False
    for _, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            any_checked = True
            if m.weight is not None:
                assert torch.allclose(m.weight.detach(), torch.ones_like(m.weight))
            if m.bias is not None:
                assert torch.allclose(m.bias.detach(), torch.zeros_like(m.bias))
    assert any_checked, "no nn.LayerNorm found in model"


# ---------------------------------------------------------------------------
# Acceptance test 4: alpha target scale matches log(d_k)
# ---------------------------------------------------------------------------


def test_alpha_target_log_dk() -> None:
    """AT4: per stage, softplus(alpha_raw) ≈ log(d_k) within 1e-6."""
    model = _build(p=3.0, feature_size=24, alpha_scheme="log_dk")
    seen_attn = 0
    for _, m in model.named_modules():
        if not isinstance(m, LpWindowAttention):
            continue
        seen_attn += 1
        d_k = int(m.dim // m.num_heads)
        expected = math.log(d_k)
        alpha = float(F.softplus(m.lp_qknorm.alpha_raw).detach().item())
        assert abs(alpha - expected) < 1e-6, (
            f"dim={m.dim}, heads={m.num_heads}, d_k={d_k}: "
            f"alpha={alpha}, expected log(d_k)={expected}"
        )
    assert seen_attn > 0, "no LpWindowAttention modules found"


def test_alpha_sqrt_dk() -> None:
    """sqrt_dk scheme: softplus(alpha_raw) ≈ sqrt(d_k)."""
    model = _build(p=3.0, feature_size=24, alpha_scheme="sqrt_dk")
    for _, m in model.named_modules():
        if not isinstance(m, LpWindowAttention):
            continue
        d_k = int(m.dim // m.num_heads)
        expected = math.sqrt(d_k)
        alpha = float(F.softplus(m.lp_qknorm.alpha_raw).detach().item())
        assert abs(alpha - expected) < 1e-6


def test_alpha_fixed_requires_value() -> None:
    """fixed scheme without a target raises LpInitError."""
    with pytest.raises(LpInitError):
        _build(p=3.0, feature_size=12, alpha_scheme="fixed", alpha_fixed=None)


def test_alpha_fixed_applies_value() -> None:
    """fixed scheme with a target yields softplus(alpha_raw) == target."""
    model = _build(p=3.0, feature_size=12, alpha_scheme="fixed", alpha_fixed=3.14)
    for _, m in model.named_modules():
        if isinstance(m, LpWindowAttention):
            alpha = float(F.softplus(m.lp_qknorm.alpha_raw).detach().item())
            assert abs(alpha - 3.14) < 1e-6


# ---------------------------------------------------------------------------
# Acceptance test 5: determinism across p
# ---------------------------------------------------------------------------


def _non_alpha_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return state_dict entries excluding LpQKNorm.alpha_raw and the p buffer.

    These two are the only tensors permitted to differ across p values when
    the init spec is held constant.
    """
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if not (k.endswith("alpha_raw") or k.endswith("lp_qknorm.p"))
    }


def test_determinism_across_p() -> None:
    """AT5: shared tensors are byte-identical across p values at fixed seed."""
    ps = [2.0, 2.5, 3.0, 3.5, 4.0]
    seed = 20260417

    # Reference: p=2.0
    ref = _non_alpha_state_dict(_build(p=ps[0], feature_size=24, seed=seed))

    for p in ps[1:]:
        sd = _non_alpha_state_dict(_build(p=p, feature_size=24, seed=seed))
        assert sd.keys() == ref.keys(), f"state_dict key drift at p={p}"
        for k in ref:
            assert torch.equal(sd[k], ref[k]), (
                f"tensor '{k}' byte-differs between p={ps[0]} and p={p}"
            )


def test_alpha_raw_identical_under_log_dk_when_dk_constant() -> None:
    """For feature_size=24 (d_k=8 at every stage), alpha_raw is identical
    across p values under the default log_dk scheme."""
    m2 = _build(p=2.0, feature_size=24)
    m3 = _build(p=3.5, feature_size=24)
    for (n2, p2), (_n3, p3) in zip(
        m2.named_parameters(), m3.named_parameters(), strict=True
    ):
        if n2.endswith("alpha_raw"):
            assert torch.equal(p2.detach(), p3.detach()), (
                f"alpha_raw at '{n2}' should be identical under log_dk when "
                f"d_k is constant across stages"
            )


# ---------------------------------------------------------------------------
# Acceptance test 6: forward-pass smoke
# ---------------------------------------------------------------------------


def test_forward_pass_smoke() -> None:
    """AT6: 16x1x224x224 input yields finite output of expected shape."""
    # feature_size=24 is the default and smallest valid for MONAI SwinUNETR.
    model = _build(p=3.0, feature_size=24)
    model.eval()
    x = torch.randn(2, 1, 224, 224)  # batch reduced from 16 for CPU speed
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 224, 224)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Guard rails on initialize_model
# ---------------------------------------------------------------------------


def test_invalid_init_scheme_raises() -> None:
    model = build_swin_unetr_lp(
        img_size=(224, 224),
        in_channels=1,
        out_channels=1,
        feature_size=12,
        lp_cfg=LpQKNormConfig(p=2.0),
    )
    with pytest.raises(LpInitError):
        initialize_model(model, init_scheme="nonsense")  # type: ignore[arg-type]


def test_invalid_linear_std_raises() -> None:
    model = build_swin_unetr_lp(
        img_size=(224, 224),
        in_channels=1,
        out_channels=1,
        feature_size=12,
        lp_cfg=LpQKNormConfig(p=2.0),
    )
    with pytest.raises(LpInitError):
        initialize_model(model, linear_init_std=-0.1)


def test_vanilla_baseline_gets_initialized() -> None:
    """lp_cfg=None path runs trunc_normal_ on the shared trunk; no alpha to seed."""
    torch.manual_seed(0)
    model = build_swin_unetr_lp(
        img_size=(224, 224),
        in_channels=1,
        out_channels=1,
        feature_size=24,
        lp_cfg=None,
    )
    for _, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                assert torch.allclose(m.weight.detach(), torch.ones_like(m.weight))
            if m.bias is not None:
                assert torch.allclose(m.bias.detach(), torch.zeros_like(m.bias))
    # Sanity: no LpQKNorm in vanilla baseline.
    assert not any(isinstance(m, LpQKNorm) for _, m in model.named_modules())
