"""Tests for ``lpqknorm.analysis.bootstrap``."""

from __future__ import annotations

import numpy as np
import pytest

from lpqknorm.analysis.bootstrap import (
    AnalysisError,
    BootstrapResult,
    holm_bonferroni,
    paired_patient_bootstrap,
)


# ---------------------------------------------------------------------------
# Paired patient bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_recovers_known_effect() -> None:
    """Spec acceptance test 1: known mean diff 0.05 ± 0.02 SD."""
    rng = np.random.default_rng(0)
    n = 200
    control = rng.normal(0.7, 0.1, n)
    treatment = control + rng.normal(0.05, 0.02, n)
    res = paired_patient_bootstrap(treatment, control, n_resamples=5000, seed=0)
    assert res.mean == pytest.approx(0.05, abs=0.01)
    assert res.p_value_one_sided < 0.01
    assert res.ci_low > 0
    assert res.distribution.shape == (5000,)
    assert res.n_patients == n


def test_bootstrap_returns_null_for_zero_effect() -> None:
    """Spec acceptance test 2: CI brackets zero, p > 0.05."""
    rng = np.random.default_rng(1)
    n = 200
    a = rng.normal(0.7, 0.1, n)
    b = a + rng.normal(0.0, 0.02, n)
    res = paired_patient_bootstrap(a, b, n_resamples=5000, seed=1)
    assert res.ci_low < 0 < res.ci_high
    assert res.p_value_one_sided > 0.05


def test_bootstrap_ci_brackets_observed_mean() -> None:
    rng = np.random.default_rng(7)
    n = 100
    diffs = rng.normal(0.1, 0.05, n)
    res = paired_patient_bootstrap(diffs, np.zeros(n), n_resamples=2000, seed=42)
    assert res.ci_low <= res.mean <= res.ci_high


def test_bootstrap_seed_determinism() -> None:
    a = np.linspace(0.5, 0.9, 50)
    b = np.linspace(0.4, 0.8, 50)
    r1 = paired_patient_bootstrap(a, b, n_resamples=1000, seed=123)
    r2 = paired_patient_bootstrap(a, b, n_resamples=1000, seed=123)
    np.testing.assert_array_equal(r1.distribution, r2.distribution)
    assert r1.mean == r2.mean
    assert r1.p_value_one_sided == r2.p_value_one_sided


def test_bootstrap_drops_nan_pairs() -> None:
    a = np.array([0.5, np.nan, 0.7, 0.8])
    b = np.array([0.4, 0.6, np.nan, 0.7])
    res = paired_patient_bootstrap(a, b, n_resamples=500, seed=0)
    assert res.n_patients == 2  # only positions 0 and 3


def test_bootstrap_rejects_length_mismatch() -> None:
    with pytest.raises(AnalysisError):
        paired_patient_bootstrap(np.zeros(5), np.zeros(4))


def test_bootstrap_rejects_too_few_observations() -> None:
    with pytest.raises(AnalysisError):
        paired_patient_bootstrap(np.array([1.0]), np.array([0.5]))


def test_bootstrap_rejects_invalid_ci() -> None:
    a = np.linspace(0, 1, 10)
    with pytest.raises(AnalysisError):
        paired_patient_bootstrap(a, a, ci=1.0)
    with pytest.raises(AnalysisError):
        paired_patient_bootstrap(a, a, ci=0.0)


def test_bootstrap_result_is_frozen() -> None:
    rng = np.random.default_rng(9)
    res = paired_patient_bootstrap(
        rng.normal(size=20), rng.normal(size=20), n_resamples=200, seed=0
    )
    assert isinstance(res, BootstrapResult)
    with pytest.raises((AttributeError, Exception)):
        res.mean = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Holm–Bonferroni
# ---------------------------------------------------------------------------


def test_holm_against_textbook_example() -> None:
    """Worked example: p = [0.01, 0.04, 0.03, 0.005].

    Sorted ascending: 0.005 (4), 0.01 (3), 0.03 (2), 0.04 (1)
    Multipliers      :    4   ,    3  ,    2  ,    1
    Adjusted (raw)   : 0.020 , 0.030 , 0.060 , 0.040
    Running max      : 0.020 , 0.030 , 0.060 , 0.060
    """
    p = np.array([0.01, 0.04, 0.03, 0.005])
    expected = np.array([0.030, 0.060, 0.060, 0.020])
    np.testing.assert_allclose(holm_bonferroni(p), expected, atol=1e-10)


def test_holm_monotone_in_sorted_order() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, size=15)
    adj = holm_bonferroni(p)
    sorted_p = np.sort(p)
    sorted_adj = adj[np.argsort(p, kind="stable")]
    diff = np.diff(sorted_adj)
    assert (diff >= -1e-12).all(), "Holm-adjusted p-values must be non-decreasing"
    # And bounded by 1.
    assert (sorted_adj <= 1.0 + 1e-12).all()
    # And >= the raw p-value.
    assert (sorted_adj >= sorted_p - 1e-12).all()


def test_holm_single_value_passthrough() -> None:
    np.testing.assert_allclose(holm_bonferroni(np.array([0.07])), [0.07])


def test_holm_handles_nan() -> None:
    p = np.array([0.01, np.nan, 0.04])
    adj = holm_bonferroni(p)
    assert np.isnan(adj[1])
    # Two finite p-values: 0.01*2 = 0.02, 0.04*1 = 0.04 (running max).
    np.testing.assert_allclose(adj[[0, 2]], [0.02, 0.04], atol=1e-12)


def test_holm_rejects_out_of_range() -> None:
    with pytest.raises(AnalysisError):
        holm_bonferroni(np.array([0.5, 1.1]))
    with pytest.raises(AnalysisError):
        holm_bonferroni(np.array([-0.1, 0.5]))
