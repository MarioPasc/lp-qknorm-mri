"""Unit tests for lpqknorm.models.lp_qknorm.

Tests cover:
- Configuration validation (ModelConfigError on bad inputs).
- _lp_normalize: unit-norm property, p=2 equivalence to Henry et al. reference.
- LpQKNorm module: buffer/parameter registration, alpha init, forward shapes.
- Critical invariant: p=2 output is numerically identical to standard L2-QKNorm
  over 100 random inputs (atol=1e-5).
- Monotone norm ordering: ||x||_p is non-increasing in p.
- Gradient safety: torch.autograd.gradcheck for p in {1.5, 2.0, 3.0, 4.0}.
- Near-zero gradient safety: no NaN gradients for p < 2 on near-zero vectors.
- Non-learnable alpha: registered as buffer, not in parameters().
- Interior-maximum sanity check: Δ(p) peaks at p* in {3, 4} on synthetic input.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lpqknorm.models.lp_qknorm import LpQKNorm, LpQKNormConfig, _lp_normalize
from lpqknorm.utils.exceptions import ModelConfigError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> torch.Generator:
    """Seeded RNG for reproducible random tensors."""
    g = torch.Generator()
    g.manual_seed(42)
    return g


def randn(*shape: int, generator: torch.Generator) -> Tensor:
    """Helper: produce a randn tensor with a given generator."""
    return torch.randn(*shape, generator=generator)


# ---------------------------------------------------------------------------
# LpQKNormConfig validation
# ---------------------------------------------------------------------------


class TestLpQKNormConfig:
    def test_valid_config_defaults(self) -> None:
        cfg = LpQKNormConfig(p=2.0)
        assert cfg.p == 2.0
        assert cfg.learnable_alpha is True
        assert cfg.init_alpha == 1.0
        assert cfg.eps == 1e-6

    def test_valid_config_custom(self) -> None:
        cfg = LpQKNormConfig(p=3.5, learnable_alpha=False, init_alpha=0.5, eps=1e-5)
        assert cfg.p == 3.5
        assert cfg.learnable_alpha is False
        assert cfg.init_alpha == 0.5
        assert cfg.eps == 1e-5

    def test_p_below_one_raises(self) -> None:
        with pytest.raises(ModelConfigError, match="p must be >= 1"):
            LpQKNormConfig(p=0.9)

    def test_p_exactly_one_is_valid(self) -> None:
        cfg = LpQKNormConfig(p=1.0)
        assert cfg.p == 1.0

    def test_eps_zero_raises(self) -> None:
        with pytest.raises(ModelConfigError, match="eps must be strictly positive"):
            LpQKNormConfig(p=2.0, eps=0.0)

    def test_eps_negative_raises(self) -> None:
        with pytest.raises(ModelConfigError, match="eps must be strictly positive"):
            LpQKNormConfig(p=2.0, eps=-1e-6)

    def test_init_alpha_zero_raises(self) -> None:
        with pytest.raises(
            ModelConfigError, match="init_alpha must be strictly positive"
        ):
            LpQKNormConfig(p=2.0, init_alpha=0.0)

    def test_init_alpha_negative_raises(self) -> None:
        with pytest.raises(
            ModelConfigError, match="init_alpha must be strictly positive"
        ):
            LpQKNormConfig(p=2.0, init_alpha=-1.0)

    def test_frozen_dataclass(self) -> None:
        from dataclasses import FrozenInstanceError

        cfg = LpQKNormConfig(p=2.0)
        with pytest.raises(FrozenInstanceError):
            cfg.p = 3.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _lp_normalize: unit-norm property
# ---------------------------------------------------------------------------


class TestLpNormalize:
    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0])
    def test_unit_norm_property(self, p: float, rng: torch.Generator) -> None:
        """Normalized output should have Lp-norm ≈ 1 for all p."""
        x = randn(4, 16, 64, generator=rng)
        eps = 1e-6
        x_hat = _lp_normalize(x, p=p, eps=eps)

        norms = x_hat.abs().pow(p).sum(dim=-1).pow(1.0 / p)
        # Allow atol accounting for eps correction: norms ≈ ||x||_p / (||x||_p + eps)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0)

    @pytest.mark.parametrize("p", [2.0, 3.0, 4.0])
    def test_shape_preserved(self, p: float, rng: torch.Generator) -> None:
        x = randn(2, 8, 32, generator=rng)
        out = _lp_normalize(x, p=p, eps=1e-6)
        assert out.shape == x.shape

    def test_dim_argument(self, rng: torch.Generator) -> None:
        """Normalization along dim=1 should give unit norms along that axis."""
        x = randn(4, 16, 32, generator=rng)
        x_hat = _lp_normalize(x, p=2.0, eps=1e-6, dim=1)
        norms = x_hat.abs().pow(2.0).sum(dim=1).pow(0.5)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0)

    def test_p2_matches_henry_et_al_reference(self, rng: torch.Generator) -> None:
        """CRITICAL: p=2 must match the Henry et al. (2020) QKNorm reference exactly.

        Reference definition:
            q_hat_ref = q / (q.norm(p=2, dim=-1, keepdim=True) + eps)

        This is the single most important numerical test in the module.
        """
        eps = 1e-6
        for _ in range(100):
            x = torch.randn(4, 16, 64, generator=rng)
            x_hat_ours = _lp_normalize(x, p=2.0, eps=eps)
            x_hat_ref = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
            torch.testing.assert_close(x_hat_ours, x_hat_ref, atol=1e-5, rtol=0.0)

    @pytest.mark.parametrize("p", [1.5, 2.0, 3.0, 4.0])
    def test_gradcheck_double_precision(self, p: float) -> None:
        """Gradient correctness via finite-difference check at double precision."""
        # Use a small tensor to keep gradcheck fast
        x = torch.randn(4, 8, 16, dtype=torch.float64, requires_grad=True)
        eps = 1e-6

        def fn(v: Tensor) -> Tensor:
            return _lp_normalize(v, p=p, eps=eps)

        assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4), (
            f"gradcheck failed for p={p}"
        )

    def test_near_zero_no_nan_gradient_p_lt_2(self) -> None:
        """Near-zero input must not produce NaN gradients for p < 2."""
        x = torch.zeros(2, 4, 8, dtype=torch.float32) + 1e-10
        x.requires_grad_(True)
        out = _lp_normalize(x, p=1.5, eps=1e-6)
        out.sum().backward()
        assert not torch.isnan(x.grad).any(), "NaN gradient for near-zero input, p=1.5"

    def test_near_zero_no_nan_gradient_p_ge_2(self) -> None:
        """Near-zero input must not produce NaN gradients for p >= 2."""
        x = torch.zeros(2, 4, 8, dtype=torch.float32) + 1e-10
        x.requires_grad_(True)
        out = _lp_normalize(x, p=3.0, eps=1e-6)
        out.sum().backward()
        assert not torch.isnan(x.grad).any(), "NaN gradient for near-zero input, p=3.0"

    @pytest.mark.parametrize("p", [2.0, 2.5, 3.0, 4.0])
    def test_exact_zero_no_nan_gradient_p_ge_2(self, p: float) -> None:
        """Exact-zero lanes must not produce NaN gradients for p >= 2.

        Regression test for the p_sweep_v1 failure: under bf16-mixed training,
        a Linear output can underflow an entire head-dim lane to 0.  With the
        former eps-post-root formulation, ``d/dS (S^{1/p}) = (1/p) S^{(1-p)/p}``
        evaluates to inf at S=0, poisoning AdamW on the first optimizer step.
        """
        x = torch.zeros(2, 4, 8, dtype=torch.float32, requires_grad=True)
        out = _lp_normalize(x, p=p, eps=1e-6)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), (
            f"Non-finite gradient for exact-zero input, p={p}"
        )


# ---------------------------------------------------------------------------
# Monotone norm ordering
# ---------------------------------------------------------------------------


class TestMonotoneNormOrdering:
    def test_lp_norm_non_increasing_in_p(self, rng: torch.Generator) -> None:
        """For fixed x, the Lp norm ||x||_p is non-increasing in p.

        This is a classical result: for p < q, ||x||_p >= ||x||_q.
        We test empirically on 50 random vectors.
        """
        p_values = [1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
        eps = 1e-6

        for _ in range(50):
            x = torch.randn(64, generator=rng)  # single vector
            norms = []
            for p in p_values:
                # Compute the ORIGINAL Lp norm (before normalization)
                if p >= 2.0:
                    raw_norm = x.abs().pow(p).sum().pow(1.0 / p)
                else:
                    raw_norm = (x.abs() + eps).pow(p).sum().pow(1.0 / p)
                norms.append(raw_norm.item())

            # Check non-increasing property with tolerance
            for i in range(len(norms) - 1):
                assert norms[i] >= norms[i + 1] - 1e-5, (
                    f"||x||_{p_values[i]:.1f}={norms[i]:.6f} < "
                    f"||x||_{p_values[i + 1]:.1f}={norms[i + 1]:.6f}: "
                    "Lp norm should be non-increasing in p"
                )


# ---------------------------------------------------------------------------
# LpQKNorm module
# ---------------------------------------------------------------------------


class TestLpQKNorm:
    def test_p_is_buffer_not_parameter(self) -> None:
        cfg = LpQKNormConfig(p=3.0)
        module = LpQKNorm(cfg)
        param_names = {name for name, _ in module.named_parameters()}
        buffer_names = {name for name, _ in module.named_buffers()}
        assert "p" not in param_names, "p must not be a learnable parameter"
        assert "p" in buffer_names, "p must be registered as a buffer"

    def test_alpha_raw_is_parameter_when_learnable(self) -> None:
        cfg = LpQKNormConfig(p=2.0, learnable_alpha=True)
        module = LpQKNorm(cfg)
        param_names = {name for name, _ in module.named_parameters()}
        assert "alpha_raw" in param_names, (
            "alpha_raw must be a parameter when learnable=True"
        )

    def test_alpha_raw_is_buffer_when_not_learnable(self) -> None:
        cfg = LpQKNormConfig(p=2.0, learnable_alpha=False)
        module = LpQKNorm(cfg)
        param_names = {name for name, _ in module.named_parameters()}
        buffer_names = {name for name, _ in module.named_buffers()}
        assert "alpha_raw" not in param_names, (
            "alpha_raw must not be a param when learnable=False"
        )
        assert "alpha_raw" in buffer_names, (
            "alpha_raw must be a buffer when learnable=False"
        )

    def test_alpha_initialization(self) -> None:
        """softplus(alpha_raw) must equal init_alpha at construction."""
        for init_alpha in [0.5, 1.0, 2.0, 5.0]:
            cfg = LpQKNormConfig(p=2.0, init_alpha=init_alpha)
            module = LpQKNorm(cfg)
            alpha_actual = F.softplus(module.alpha_raw).item()
            assert abs(alpha_actual - init_alpha) < 1e-5, (
                f"Expected softplus(alpha_raw)={init_alpha}, got {alpha_actual}"
            )

    def test_forward_shapes(self, rng: torch.Generator) -> None:
        """Forward pass must return tensors of the correct shapes."""
        cfg = LpQKNormConfig(p=3.0)
        module = LpQKNorm(cfg)
        q = randn(4, 16, 64, generator=rng)
        k = randn(4, 16, 64, generator=rng)
        q_hat, k_hat, alpha = module(q, k)
        assert q_hat.shape == q.shape
        assert k_hat.shape == k.shape
        assert alpha.shape == torch.Size([])  # scalar

    def test_alpha_is_positive(self, rng: torch.Generator) -> None:
        """The alpha output from forward must always be strictly positive."""
        cfg = LpQKNormConfig(p=2.0)
        module = LpQKNorm(cfg)
        q = randn(4, 16, 64, generator=rng)
        k = randn(4, 16, 64, generator=rng)
        _, _, alpha = module(q, k)
        assert alpha.item() > 0.0, "alpha must be positive (softplus guarantees this)"

    def test_forward_no_nan(self, rng: torch.Generator) -> None:
        """Forward pass must not produce NaN for normal inputs."""
        for p in [1.5, 2.0, 3.0, 4.0]:
            cfg = LpQKNormConfig(p=p)
            module = LpQKNorm(cfg)
            q = randn(4, 16, 64, generator=rng)
            k = randn(4, 16, 64, generator=rng)
            q_hat, k_hat, alpha = module(q, k)
            assert not torch.isnan(q_hat).any(), f"NaN in q_hat for p={p}"
            assert not torch.isnan(k_hat).any(), f"NaN in k_hat for p={p}"
            assert not torch.isnan(alpha), f"NaN in alpha for p={p}"

    @pytest.mark.parametrize("p", [1.5, 2.0, 2.5, 3.0, 4.0, 8.0])
    def test_output_unit_norms(self, p: float, rng: torch.Generator) -> None:
        """Both q_hat and k_hat must have Lp norm ≈ 1 along the last dimension."""
        cfg = LpQKNormConfig(p=p)
        module = LpQKNorm(cfg)
        x = randn(4, 16, 64, generator=rng)
        q_hat, k_hat, _ = module(x, x)
        for name, hat in [("q_hat", q_hat), ("k_hat", k_hat)]:
            norms = hat.abs().pow(p).sum(-1).pow(1.0 / p)
            torch.testing.assert_close(
                norms,
                torch.ones_like(norms),
                atol=1e-4,
                rtol=0.0,
                msg=f"Unit-norm failed for {name} at p={p}",
            )

    def test_p_value_stored_correctly(self) -> None:
        """The buffer p must store the value from config."""
        for p_val in [1.0, 2.0, 3.5, 4.0]:
            cfg = LpQKNormConfig(p=p_val)
            module = LpQKNorm(cfg)
            assert abs(module.p.item() - p_val) < 1e-7


# ---------------------------------------------------------------------------
# Critical p=2 equivalence test (module-level)
# ---------------------------------------------------------------------------


class TestP2Equivalence:
    def test_p2_module_matches_henry_et_al_100_inputs(
        self, rng: torch.Generator
    ) -> None:
        """CRITICAL: LpQKNorm(p=2) must match Henry et al. reference over 100 inputs.

        Reference formula (Henry et al., 2020, arXiv:2010.04245):
            q_hat_ref = q / (q.norm(p=2, dim=-1, keepdim=True) + eps)

        If this test fails, every downstream claim about p=2 being the
        QKNorm baseline is false.
        """
        eps = 1e-6
        cfg = LpQKNormConfig(p=2.0, eps=eps)
        module = LpQKNorm(cfg)
        module.eval()

        for trial in range(100):
            q = torch.randn(4, 16, 64, generator=rng)
            k = torch.randn(4, 16, 64, generator=rng)

            q_hat_ours, k_hat_ours, _ = module(q, k)

            q_hat_ref = q / (q.norm(p=2, dim=-1, keepdim=True) + eps)
            k_hat_ref = k / (k.norm(p=2, dim=-1, keepdim=True) + eps)

            torch.testing.assert_close(
                q_hat_ours,
                q_hat_ref,
                atol=1e-5,
                rtol=0.0,
                msg=f"q_hat mismatch at trial {trial}: LpQKNorm(p=2) != Henry et al. reference",
            )
            torch.testing.assert_close(
                k_hat_ours,
                k_hat_ref,
                atol=1e-5,
                rtol=0.0,
                msg=f"k_hat mismatch at trial {trial}: LpQKNorm(p=2) != Henry et al. reference",
            )

    def test_p2_stability_on_near_zero_vectors(self) -> None:
        """Near-zero inputs must produce finite, bounded output under the
        eps-inside-root spec form.

        The spec form (||v||_p = (Σ|v_h|^p + ε)^(1/p)) departs from Henry's
        exact form (||v||_2 + ε) when ||v|| ≪ √ε, because the additive eps
        lives inside the root. The two forms agree to relative tolerance
        5×10⁻⁷ for inputs of unit scale (covered by the random-trials test),
        but diverge when inputs are in the eps regime. This test replaces
        the old Henry-exact check at near-zero inputs with the properties
        that actually matter for training stability.
        """
        eps = 1e-6
        cfg = LpQKNormConfig(p=2.0, eps=eps)
        module = LpQKNorm(cfg)
        module.eval()

        q = torch.full((2, 4, 16), 1e-8, requires_grad=True)
        k = torch.full((2, 4, 16), 1e-8, requires_grad=True)

        q_hat_ours, k_hat_ours, _ = module(q, k)

        assert torch.isfinite(q_hat_ours).all()
        assert torch.isfinite(k_hat_ours).all()

        q_row_norm = q_hat_ours.norm(p=2, dim=-1)
        k_row_norm = k_hat_ours.norm(p=2, dim=-1)
        assert (q_row_norm <= 1.0 + 1e-6).all()
        assert (k_row_norm <= 1.0 + 1e-6).all()

        (q_hat_ours.sum() + k_hat_ours.sum()).backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()

    def test_p2_exact_zero_is_finite(self) -> None:
        """Exact-zero inputs must produce zero output and finite gradients.

        This is the production scenario: bf16 Linear outputs can underflow
        to exact zero on a whole lane, and the forward + backward must not
        introduce NaN/Inf. Regression for the p_sweep_v1 failure.
        """
        cfg = LpQKNormConfig(p=2.0, eps=1e-6)
        module = LpQKNorm(cfg)
        module.eval()

        q = torch.zeros(2, 4, 16, requires_grad=True)
        k = torch.zeros(2, 4, 16, requires_grad=True)

        q_hat, k_hat, _ = module(q, k)

        torch.testing.assert_close(q_hat, torch.zeros_like(q_hat))
        torch.testing.assert_close(k_hat, torch.zeros_like(k_hat))

        (q_hat.sum() + k_hat.sum()).backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()


# ---------------------------------------------------------------------------
# Interior-maximum sanity check
# ---------------------------------------------------------------------------


class TestInteriorMaximum:
    def test_logit_gap_peaks_at_p_star(self) -> None:
        """Verify the toy-model prediction: Δ(p) = s_qk_L - s_qk_B has its
        maximum at p* ∈ {3, 4} on a controlled synthetic input.

        Setup (matches phase-2 spec §8):
        - d_k = 64, 1 query, 1 lesion key (peaky), 1 background key (diffuse).
        - The query is aligned with the lesion key direction.
        - Peaky = energy concentrated in a few coordinates.
        - Diffuse = energy spread uniformly.

        The toy model predicts an interior maximum at p* > 2 for this type of
        input when the query/key vectors are sufficiently peaky.
        """
        torch.manual_seed(0)
        d_k = 64
        eps = 1e-6

        # Construct a peaky query: most energy in first 4 coordinates
        q = torch.zeros(d_k)
        q[:4] = 1.0  # 4 active coordinates out of 64

        # Lesion key: aligned with query (same peaky structure)
        k_lesion = torch.zeros(d_k)
        k_lesion[:4] = 1.0

        # Background key: diffuse (energy spread uniformly)
        k_bg = torch.ones(d_k) / math.sqrt(d_k)

        p_values = [1.5, 2.0, 2.5, 3.0, 4.0, 8.0]
        gaps: list[float] = []

        for p in p_values:
            q_hat = _lp_normalize(q.unsqueeze(0), p=p, eps=eps, dim=-1).squeeze(0)
            kl_hat = _lp_normalize(k_lesion.unsqueeze(0), p=p, eps=eps, dim=-1).squeeze(
                0
            )
            kb_hat = _lp_normalize(k_bg.unsqueeze(0), p=p, eps=eps, dim=-1).squeeze(0)

            s_lesion = torch.dot(q_hat, kl_hat).item()
            s_bg = torch.dot(q_hat, kb_hat).item()
            gaps.append(s_lesion - s_bg)

        # Find the p with the maximum logit gap
        max_idx = int(torch.tensor(gaps).argmax().item())
        p_star = p_values[max_idx]

        assert p_star in {3.0, 4.0}, (
            f"Expected interior maximum at p* ∈ {{3.0, 4.0}}, "
            f"got p*={p_star}. Gaps: {dict(zip(p_values, [f'{g:.4f}' for g in gaps], strict=True))}"
        )

        # Also verify that the gap strictly increases from p=2 to p* (monotone on [2, p*])
        idx_p2 = p_values.index(2.0)
        assert gaps[max_idx] > gaps[idx_p2], (
            f"Gap at p*={p_star} ({gaps[max_idx]:.4f}) must exceed gap at p=2 "
            f"({gaps[idx_p2]:.4f})"
        )
