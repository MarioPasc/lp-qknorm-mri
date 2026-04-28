"""Effect-size measures for paired observations.

References
----------
- Cohen. *Statistical Power Analysis for the Behavioral Sciences*.
  2nd ed., Routledge, 1988.
- Hedges. *Distribution Theory for Glass's Estimator of Effect Size*.
  J. Educ. Stat. 6(2), 1981.
"""

from __future__ import annotations

import math

import numpy as np

from lpqknorm.analysis.bootstrap import AnalysisError


def _paired_finite(
    treatment: np.ndarray, control: np.ndarray
) -> tuple[np.ndarray, int]:
    t = np.asarray(treatment, dtype=float).reshape(-1)
    c = np.asarray(control, dtype=float).reshape(-1)
    if t.shape != c.shape:
        raise AnalysisError(f"length mismatch: treatment={t.shape}, control={c.shape}")
    diffs = t - c
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        raise AnalysisError(f"need >= 2 paired finite differences, got {diffs.size}")
    return diffs, diffs.size


def paired_cohen_d(values_treatment: np.ndarray, values_control: np.ndarray) -> float:
    """Paired Cohen's d on the per-pair differences.

    ``d = mean(d_i) / std(d_i, ddof=1)``, computed on
    ``d_i = treatment_i - control_i``.

    Parameters
    ----------
    values_treatment, values_control : np.ndarray
        Paired per-patient values of equal length.

    Returns
    -------
    float
        Paired Cohen's d.  Returns ``+inf`` (resp. ``-inf``) when the
        sample standard deviation is zero and the mean is positive
        (resp. negative).  Returns ``0.0`` when both are zero.
    """
    diffs, _n = _paired_finite(values_treatment, values_control)
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        if mean == 0.0:
            return 0.0
        return math.inf if mean > 0 else -math.inf
    return mean / sd


def hedges_g(values_treatment: np.ndarray, values_control: np.ndarray) -> float:
    """Hedges' g — small-sample-corrected Cohen's d.

    ``g = J(n - 1) * d`` where ``J(df) ≈ 1 - 3 / (4 df - 1)``
    (Hedges, 1981).  Falls back to plain ``d`` when ``n < 4``.

    Parameters
    ----------
    values_treatment, values_control : np.ndarray
        Paired per-patient values of equal length.

    Returns
    -------
    float
        Hedges' g.
    """
    d = paired_cohen_d(values_treatment, values_control)
    if not math.isfinite(d):
        return d
    _, n = _paired_finite(values_treatment, values_control)
    df = n - 1
    if df < 3:
        return d
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return correction * d
