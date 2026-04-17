"""Unit tests for mechanistic probes on synthetic attention (AT1 + AT2).

AT1: Probe correctness on controlled inputs — verify theoretical bounds.
AT2: Logit gap varies with p on synthetic peaky Q/K.
"""

from __future__ import annotations

import math

import pytest
import torch

from lpqknorm.models.hooks import AttentionCapture
from lpqknorm.models.lp_qknorm import LpQKNorm, LpQKNormConfig
from lpqknorm.probes.attention_iou import AttentionMaskIoU
from lpqknorm.probes.entropy import AttentionEntropy
from lpqknorm.probes.lesion_mass import LesionAttentionMass
from lpqknorm.probes.linear_probe import LinearProbe
from lpqknorm.probes.logit_gap import LesionBackgroundLogitGap
from lpqknorm.probes.peakiness import FeaturePeakiness
from lpqknorm.probes.spatial_loc_error import SpatialLocalizationError
from lpqknorm.probes.spectral import SpectralProbe


class TestPeakinessAT1:
    """AT1: Peakiness bounds on controlled inputs."""

    def test_one_hot_gives_peakiness_one(self) -> None:
        """One-hot vector → rho = 1."""
        v = torch.eye(64)[0].unsqueeze(0)  # (1, 64)
        rho = FeaturePeakiness.compute_value(v)
        assert rho.item() == pytest.approx(1.0, abs=1e-5)

    def test_uniform_gives_peakiness_lower_bound(self) -> None:
        """Uniform vector → rho = 1/sqrt(d)."""
        d = 64
        v = torch.ones(1, d) / math.sqrt(d)
        rho = FeaturePeakiness.compute_value(v)
        assert rho.item() == pytest.approx(1.0 / math.sqrt(d), abs=1e-4)

    def test_peakiness_in_valid_range(self) -> None:
        """Random vectors have rho in [1/sqrt(d), 1]."""
        d = 8  # head_dim for feature_size=12
        torch.manual_seed(0)
        v = torch.randn(10, 3, 49, d)
        rho = FeaturePeakiness.compute_value(v)
        lb = 1.0 / math.sqrt(d) - 1e-5
        assert (rho >= lb).all()
        assert (rho <= 1.0 + 1e-5).all()

    def test_compute_on_capture(self) -> None:
        """Full compute() returns correct shape and range."""
        torch.manual_seed(42)
        bnw, nh, n, dh = 4, 3, 49, 8
        q = torch.randn(bnw, nh, n, dh)
        cap = AttentionCapture(q=q, k=q)
        flags = torch.zeros(bnw, n, dtype=torch.bool)
        flags[:, :3] = True

        probe = FeaturePeakiness("q")
        result = probe.compute(cap, flags)
        assert result.per_token is not None
        assert result.per_token.shape == (bnw * nh * n,)
        assert (result.per_token >= 0.0).all()
        assert (result.per_token <= 1.0 + 1e-5).all()


class TestEntropyAT1:
    """AT1: Entropy bounds on controlled inputs."""

    def test_uniform_attention_gives_max_entropy(self) -> None:
        """Uniform row → H = log(W²)."""
        w2 = 49
        uniform = torch.ones(1, w2) / w2
        h_val = AttentionEntropy.compute_value(uniform)
        assert h_val.item() == pytest.approx(math.log(w2), abs=1e-5)

    def test_onehot_attention_gives_zero_entropy(self) -> None:
        """One-hot row → H = 0."""
        w2 = 49
        onehot = torch.zeros(1, w2)
        onehot[0, 7] = 1.0
        h_val = AttentionEntropy.compute_value(onehot)
        assert h_val.item() == pytest.approx(0.0, abs=1e-5)

    def test_entropy_in_valid_range(self) -> None:
        """Random softmax rows have H in [0, log(W²)]."""
        a = torch.softmax(torch.randn(20, 49), dim=-1)
        h_val = AttentionEntropy.compute_value(a)
        assert (h_val >= 0.0).all()
        assert (h_val <= math.log(49) + 1e-5).all()


class TestLesionMassAT1:
    """AT1: Lesion mass range and edge cases."""

    def test_range_is_zero_one(self) -> None:
        """Lesion mass is in [0, 1]."""
        # compute_per_query expects (n, n) attention matrix
        a = torch.softmax(torch.randn(49, 49), dim=-1)
        lesion = torch.zeros(49, dtype=torch.bool)
        lesion[:4] = True
        m = LesionAttentionMass.compute_per_query(a, lesion)
        assert (m >= 0.0).all()
        assert (m <= 1.0 + 1e-6).all()

    def test_no_lesion_returns_empty(self) -> None:
        """No lesion tokens → empty result."""
        a = torch.softmax(torch.randn(49, 49), dim=-1)
        lesion = torch.zeros(49, dtype=torch.bool)
        m = LesionAttentionMass.compute_per_query(a, lesion)
        assert m.numel() == 0

    def test_all_lesion_returns_one(self) -> None:
        """All tokens are lesion → mass = 1.0."""
        a = torch.softmax(torch.randn(49, 49), dim=-1)
        lesion = torch.ones(49, dtype=torch.bool)
        m = LesionAttentionMass.compute_per_query(a, lesion)
        torch.testing.assert_close(m, torch.ones_like(m), atol=1e-5, rtol=0.0)


class TestAttentionIoUAT1:
    """AT1: Attention-mask IoU range."""

    def test_iou_in_valid_range(self) -> None:
        """IoU values are in [0, 1]."""
        torch.manual_seed(0)
        bnw, nh, n = 2, 1, 49
        attn = torch.softmax(torch.randn(bnw, nh, n, n), dim=-1)
        flags = torch.zeros(bnw, n, dtype=torch.bool)
        flags[:, :5] = True

        cap = AttentionCapture(attention=attn)
        probe = AttentionMaskIoU()
        result = probe.compute(cap, flags)
        assert result.per_query is not None
        assert (result.per_query >= 0.0).all()
        assert (result.per_query <= 1.0 + 1e-5).all()


class TestLogitGapAT2:
    """AT2: Logit gap varies with p on synthetic peaky Q/K."""

    @staticmethod
    def _build_capture(p: float) -> AttentionCapture:
        """Build synthetic capture with peaky lesion and diffuse background."""
        n, dh = 49, 8
        torch.manual_seed(42)
        q = torch.randn(1, 1, n, dh) * 0.1
        q[0, 0, 0, 0] = 5.0  # peaky lesion query
        k = torch.randn(1, 1, n, dh) * 0.1
        k[0, 0, 0, 0] = 5.0  # aligned lesion key

        lp_norm = LpQKNorm(LpQKNormConfig(p=p, learnable_alpha=False))
        with torch.no_grad():
            q_hat, k_hat, alpha = lp_norm(q, k)
            logits = alpha * (q_hat @ k_hat.transpose(-2, -1))

        bias = torch.zeros(1, 1, n, n)
        attn = torch.softmax(logits + bias, dim=-1)
        return AttentionCapture(
            q=q,
            k=k,
            q_hat=q_hat,
            k_hat=k_hat,
            logits=logits,
            attention=attn,
            alpha=alpha,
            relative_position_bias=bias,
            stage_index=0,
            block_index=0,
        )

    def test_gap_is_positive(self) -> None:
        """Gap for peaky lesion token should be positive."""
        flags = torch.zeros(1, 49, dtype=torch.bool)
        flags[0, 0] = True
        cap = self._build_capture(3.0)
        result = LesionBackgroundLogitGap().compute(cap, flags)
        assert result.per_query is not None
        assert result.per_query[0].item() > 0

    def test_gap_varies_with_p(self) -> None:
        """Gap should vary non-trivially across p values."""
        flags = torch.zeros(1, 49, dtype=torch.bool)
        flags[0, 0] = True
        probe = LesionBackgroundLogitGap()
        gaps: list[float] = []
        for p in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            cap = self._build_capture(p)
            result = probe.compute(cap, flags)
            assert result.per_query is not None
            gaps.append(float(result.per_query[0]))
        # All positive
        assert all(g > 0 for g in gaps), f"All gaps must be positive: {gaps}"
        # Not all identical (probe is sensitive to p)
        assert max(gaps) - min(gaps) > 1e-4, f"Gaps should vary: {gaps}"


class TestSpatialLocErrorAT1:
    """AT1: Spatial localisation error bounds and centre alignment."""

    def test_peaked_on_centroid_gives_zero(self) -> None:
        """Attention peaked on the lesion centroid → SLE = 0."""
        w = 7
        attn = torch.zeros(1, w * w)
        attn[0, 24] = 1.0  # centre of a 7x7 window
        lesion_mask = torch.zeros(w * w, dtype=torch.bool)
        lesion_mask[24] = True
        sle = SpatialLocalizationError.compute_per_query(attn, lesion_mask, w)
        assert sle.item() == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self) -> None:
        """SLE is always non-negative."""
        torch.manual_seed(0)
        w = 7
        attn = torch.softmax(torch.randn(4, w * w), dim=-1)
        lesion_mask = torch.zeros(w * w, dtype=torch.bool)
        lesion_mask[:5] = True
        sle = SpatialLocalizationError.compute_per_query(attn, lesion_mask, w)
        assert (sle >= 0.0).all()

    def test_bounded_by_window_diagonal(self) -> None:
        """SLE never exceeds W*sqrt(2)."""
        w = 7
        attn = torch.zeros(1, w * w)
        attn[0, 0] = 1.0  # corner
        lesion_mask = torch.zeros(w * w, dtype=torch.bool)
        lesion_mask[w * w - 1] = True  # opposite corner
        sle = SpatialLocalizationError.compute_per_query(attn, lesion_mask, w)
        assert sle.item() <= w * math.sqrt(2)


class TestSpectralAT1:
    """AT1: Participation ratio sanity."""

    def test_isotropic_pr_equals_dk(self) -> None:
        """Isotropic Gaussian features have PR ≈ d_k."""
        torch.manual_seed(0)
        x = torch.randn(10_000, 16)
        pr = SpectralProbe._participation_ratio(x)
        assert pr == pytest.approx(16.0, rel=0.1)

    def test_rank_one_pr_equals_one(self) -> None:
        """Rank-1 features have PR ≈ 1."""
        torch.manual_seed(0)
        u = torch.randn(16)
        x = torch.randn(10_000, 1) * u
        pr = SpectralProbe._participation_ratio(x)
        assert pr == pytest.approx(1.0, abs=0.1)

    def test_eigenvalues_length_and_order(self) -> None:
        """Eigenvalue output has length d_k and is sorted descending."""
        torch.manual_seed(0)
        x = torch.randn(200, 8)
        ev = SpectralProbe._eigenvalues(x)
        assert ev.shape == (8,)
        diff = ev[:-1] - ev[1:]
        assert (diff >= -1e-5).all()


class TestLinearProbeAT1:
    """AT1: Linear probe separability."""

    def test_separable_features_give_high_ba(self) -> None:
        """Clearly separable Gaussian clusters → BA ≥ 0.95."""
        torch.manual_seed(0)
        x_lesion = torch.randn(100, 16) + 3.0
        x_bg = torch.randn(100, 16) - 3.0
        metrics = LinearProbe(n_splits=5).compute_value(x_lesion, x_bg)
        assert metrics.balanced_accuracy >= 0.95

    def test_non_separable_features_give_chance(self) -> None:
        """Identical clusters → BA near 0.5."""
        torch.manual_seed(0)
        x_lesion = torch.randn(100, 16)
        x_bg = torch.randn(100, 16)
        metrics = LinearProbe(n_splits=5).compute_value(x_lesion, x_bg)
        assert metrics.balanced_accuracy <= 0.7
