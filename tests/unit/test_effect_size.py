"""Tests for ``lpqknorm.analysis.effect_size``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lpqknorm.analysis.bootstrap import AnalysisError
from lpqknorm.analysis.effect_size import hedges_g, paired_cohen_d


def test_cohen_d_exact_constant_diff_is_inf() -> None:
    """Diff exactly representable as a constant -> std == 0 -> ±inf."""
    treatment = np.array([0.5, 0.5, 0.5, 0.5])
    control = np.zeros(4)
    assert paired_cohen_d(treatment, control) == math.inf


def test_cohen_d_floating_point_constant_diff_is_huge() -> None:
    """Float subtraction may leave a near-zero std; magnitude must still
    dominate any reasonable practical effect (>>> 1)."""
    treatment = np.array([0.7, 0.8, 0.6, 0.9])
    control = np.array([0.5, 0.6, 0.4, 0.7])
    assert paired_cohen_d(treatment, control) > 1e10


def test_cohen_d_constant_zero_diff_is_zero() -> None:
    a = np.array([0.5, 0.5, 0.5])
    assert paired_cohen_d(a, a) == 0.0


def test_cohen_d_negative_direction() -> None:
    treatment = np.array([0.5, 0.5, 0.5, 0.5])
    control = np.array([0.7, 0.7, 0.7, 0.7])
    assert paired_cohen_d(treatment, control) == -math.inf


def test_cohen_d_random_matches_manual() -> None:
    rng = np.random.default_rng(123)
    n = 80
    control = rng.normal(0.7, 0.1, n)
    treatment = control + rng.normal(0.05, 0.03, n)
    expected = (treatment - control).mean() / (treatment - control).std(ddof=1)
    assert paired_cohen_d(treatment, control) == pytest.approx(expected, rel=1e-12)


def test_cohen_d_known_large_effect() -> None:
    """Spec-style sanity test: nearly-constant 0.1 diff with tiny noise."""
    rng = np.random.default_rng(0)
    diff = np.full(100, 0.1) + 1e-9 * rng.standard_normal(100)
    e = np.zeros(100)
    assert paired_cohen_d(diff, e) > 1.0


def test_cohen_d_rejects_length_mismatch() -> None:
    with pytest.raises(AnalysisError):
        paired_cohen_d(np.zeros(5), np.zeros(4))


def test_cohen_d_rejects_too_few_pairs() -> None:
    with pytest.raises(AnalysisError):
        paired_cohen_d(np.array([0.5]), np.array([0.4]))


def test_hedges_g_smaller_magnitude_than_cohen() -> None:
    rng = np.random.default_rng(7)
    n = 30
    control = rng.normal(0.6, 0.1, n)
    treatment = control + rng.normal(0.04, 0.03, n)
    d = paired_cohen_d(treatment, control)
    g = hedges_g(treatment, control)
    # |g| <= |d| because the small-sample correction shrinks toward zero.
    assert abs(g) <= abs(d) * 1.0 + 1e-12
    # Expected correction factor.
    df = n - 1
    expected_correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    assert g == pytest.approx(d * expected_correction, rel=1e-12)


def test_hedges_g_passes_through_inf() -> None:
    treatment = np.array([0.5, 0.5, 0.5, 0.5])
    control = np.zeros(4)  # diff exactly 0.5 -> std == 0 -> +inf
    assert hedges_g(treatment, control) == math.inf


def test_hedges_g_falls_back_to_d_for_n_below_4() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 1.5, 2.0])
    assert hedges_g(a, b) == paired_cohen_d(a, b)
