"""Patient-level paired bootstrap and Holm–Bonferroni correction.

References
----------
- Efron & Tibshirani. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
- Holm. *A Simple Sequentially Rejective Multiple Test Procedure*.
  Scand. J. Stat. 6(2), 1979.
- Nadeau & Bengio. *Inference for the Generalization Error*. Mach. Learn. 2003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lpqknorm.utils.exceptions import LpQKNormError


class AnalysisError(LpQKNormError):
    """Raised on misuse of the analysis utilities."""


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a paired patient-level bootstrap.

    Attributes
    ----------
    mean : float
        Observed mean of the paired differences ``treatment - control``.
    ci_low : float
        Lower bound of the percentile bootstrap CI on the mean.
    ci_high : float
        Upper bound of the percentile bootstrap CI on the mean.
    p_value_one_sided : float
        Fraction of bootstrap means with sign **opposite** to the
        observed mean.  Under the null of zero mean difference this
        approximates a one-sided p-value (Efron & Tibshirani, 1993).
    distribution : np.ndarray
        The bootstrap distribution of the mean, shape ``(n_resamples,)``.
    n_patients : int
        Number of paired observations entering the bootstrap.
    """

    mean: float
    ci_low: float
    ci_high: float
    p_value_one_sided: float
    distribution: np.ndarray
    n_patients: int


def paired_patient_bootstrap(
    values_treatment: np.ndarray,
    values_control: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 20260216,
    ci: float = 0.95,
) -> BootstrapResult:
    """Patient-level paired bootstrap of the mean difference.

    Parameters
    ----------
    values_treatment, values_control : np.ndarray
        Paired per-patient values, both shape ``(n,)``.  Pairing is
        positional: row ``i`` of the two arrays must refer to the
        same patient.
    n_resamples : int, default 10_000
        Number of bootstrap replicates.
    seed : int, default 20260216
        RNG seed for reproducibility.
    ci : float, default 0.95
        Two-sided percentile CI level.

    Returns
    -------
    BootstrapResult

    Raises
    ------
    AnalysisError
        On length mismatch, empty input, or out-of-range ``ci``.
    """
    t = np.asarray(values_treatment, dtype=float).reshape(-1)
    c = np.asarray(values_control, dtype=float).reshape(-1)
    if t.shape != c.shape:
        raise AnalysisError(f"length mismatch: treatment={t.shape}, control={c.shape}")
    # Drop pairs with NaN in either arm — patients with undefined metric
    # (e.g. lesion-recall on a patient without small lesions) cannot
    # contribute to a paired comparison.
    finite = np.isfinite(t) & np.isfinite(c)
    t = t[finite]
    c = c[finite]
    n = t.size
    if n < 2:
        raise AnalysisError(f"need at least 2 paired finite observations, got {n}")
    if not 0.0 < ci < 1.0:
        raise AnalysisError(f"ci must be in (0, 1), got {ci}")
    if n_resamples < 1:
        raise AnalysisError(f"n_resamples must be >= 1, got {n_resamples}")

    diffs = t - c
    observed_mean = float(diffs.mean())

    rng = np.random.default_rng(seed)
    # Vectorised resampling: sample a (B, n) index matrix once.
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)

    alpha = 1.0 - ci
    ci_low, ci_high = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])

    # One-sided p-value: fraction of bootstrap means with sign opposite
    # to the observed direction.  ``observed_mean == 0`` is rare on
    # continuous data; degenerate to 0.5 to avoid implying significance.
    if observed_mean > 0:
        p_one = float(np.mean(boot_means <= 0.0))
    elif observed_mean < 0:
        p_one = float(np.mean(boot_means >= 0.0))
    else:
        p_one = 0.5

    return BootstrapResult(
        mean=observed_mean,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value_one_sided=p_one,
        distribution=boot_means,
        n_patients=n,
    )


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values.

    Implements the algorithm of Holm (1979): sort the *m* raw p-values
    ascending, compare ``p_(k)`` to ``alpha / (m - k + 1)``.  The
    adjusted p-values are the running maximum of ``(m - k + 1) p_(k)``,
    clipped to ``[0, 1]``, returned in the original order.

    Parameters
    ----------
    p_values : np.ndarray
        Raw two-sided p-values.  NaN entries are propagated unchanged.

    Returns
    -------
    np.ndarray
        Holm-adjusted p-values, same shape as input.

    Raises
    ------
    AnalysisError
        If any non-NaN p is outside ``[0, 1]``.
    """
    p = np.asarray(p_values, dtype=float).copy()
    finite = np.isfinite(p)
    if (p[finite] < 0).any() or (p[finite] > 1).any():
        raise AnalysisError("p_values must lie in [0, 1]")

    out = np.full_like(p, np.nan, dtype=float)
    if not finite.any():
        return out

    p_finite = p[finite]
    m = p_finite.size
    order = np.argsort(p_finite, kind="stable")
    ranked = p_finite[order]
    multipliers = np.arange(m, 0, -1, dtype=float)  # m, m-1, ..., 1
    # Running max over the sequence (m - k + 1) * p_(k).
    raw_adjusted = ranked * multipliers
    adjusted_sorted = np.clip(np.maximum.accumulate(raw_adjusted), 0.0, 1.0)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(m)
    out[finite] = adjusted_sorted[inverse]
    return out
