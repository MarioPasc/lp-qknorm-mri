"""Aggregate probe values for the mechanistic-chain figures and tests.

The probe HDF5s written by Phase 4 contain per-token / per-query
trajectories.  For the across-`p` curves and per-patient correlations
we need a single scalar per patient: the mean of the probe over all
*lesion* tokens of that patient.  This module performs that
reduction starting from the long-format frame returned by
:func:`lpqknorm.analysis.aggregation.load_probes`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from lpqknorm.analysis.bootstrap import AnalysisError


def probe_curve(
    probes: pd.DataFrame,
    probe_name: str,
    aggregate: Literal["patient", "token"] = "patient",
    *,
    block: str | None = None,
) -> pd.DataFrame:
    """Aggregate probe ``probe_name`` per ``(p, fold[, subject_id])``.

    Parameters
    ----------
    probes : pd.DataFrame
        Output of :func:`lpqknorm.analysis.aggregation.load_probes`.
        Required columns: ``[p, p_label, fold, block, subject_id,
        probe_name, value]``.
    probe_name : str
        Probe to aggregate (must match a value of ``probe_name``).
    aggregate : {"patient", "token"}
        ``"patient"`` averages over patients per ``(p, fold)``.
        ``"token"`` returns a per-patient row instead.
    block : str, optional
        Restrict to a single block (e.g. ``"block_0_wmsa"``).

    Returns
    -------
    pd.DataFrame
        Long-format with columns ``[p, p_label, fold, value]`` (mode
        ``"patient"``) or ``[p, p_label, fold, subject_id, value]``
        (mode ``"token"``).

    Raises
    ------
    AnalysisError
        If *probe_name* is unknown or *probes* is missing columns.
    """
    required = {"p", "p_label", "fold", "block", "subject_id", "probe_name", "value"}
    missing = required - set(probes.columns)
    if missing:
        raise AnalysisError(f"probes is missing columns: {sorted(missing)}")
    sub = probes[probes["probe_name"] == probe_name]
    if block is not None:
        sub = sub[sub["block"] == block]
    if sub.empty:
        raise AnalysisError(f"no rows for probe_name={probe_name!r}, block={block!r}")

    # First reduce to one scalar per (run, subject): mean over slices.
    per_patient = sub.groupby(
        ["p", "p_label", "fold", "subject_id"],
        as_index=False,
        dropna=False,
    )["value"].mean(numeric_only=True)
    if aggregate == "token":
        return per_patient
    if aggregate != "patient":
        raise AnalysisError(
            f"aggregate must be 'patient' or 'token', got {aggregate!r}"
        )
    return per_patient.groupby(["p", "p_label", "fold"], as_index=False, dropna=False)[
        "value"
    ].mean(numeric_only=True)


def probe_outcome_correlation(
    probes: pd.DataFrame,
    per_patient: pd.DataFrame,
    *,
    probe_name: str,
    metric: str = "lesion_recall",
    block: str | None = None,
) -> pd.DataFrame:
    """Compute Pearson correlation of a probe with a per-patient metric.

    For each ``(p, fold)`` the function computes the Pearson r and the
    corresponding two-sided p-value using a t-approximation
    (``t = r sqrt((n-2) / (1 - r^2))``).  Results are emitted in long
    format, one row per ``(p, fold)``.

    Parameters
    ----------
    probes : pd.DataFrame
        Long-format probes frame (as in :func:`probe_curve`).
    per_patient : pd.DataFrame
        Patient-level segmentation metrics (output of
        :func:`lpqknorm.analysis.aggregation.load_per_patient`,
        ``aggregate=True``).
    probe_name : str
        Probe to correlate.
    metric : str, default ``"lesion_recall"``
        Column from *per_patient* to correlate against.
    block : str, optional
        Restrict to one block.

    Returns
    -------
    pd.DataFrame
        Columns ``[p, p_label, fold, n, pearson_r, p_value]``.
    """
    if metric not in per_patient.columns:
        raise AnalysisError(
            f"per_patient lacks metric {metric!r}; columns={list(per_patient.columns)}"
        )
    probe_pp = probe_curve(probes, probe_name, aggregate="token", block=block)
    join = probe_pp.merge(
        per_patient[["p", "p_label", "fold", "subject_id", metric]].rename(
            columns={metric: "_y"}
        ),
        on=["p", "p_label", "fold", "subject_id"],
        how="inner",
    )

    rows: list[dict[str, object]] = []
    for (p, p_label, fold), grp in join.groupby(["p", "p_label", "fold"], dropna=False):
        x = grp["value"].to_numpy(dtype=float)
        y = grp["_y"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        n = x.size
        if n < 3 or x.std() == 0 or y.std() == 0:
            r = float("nan")
            pval = float("nan")
        else:
            r = float(np.corrcoef(x, y)[0, 1])
            # Two-sided t-approximation for Pearson's r.
            t_stat = r * np.sqrt((n - 2) / max(1.0 - r * r, 1e-12))
            from scipy.stats import t as student_t  # local import keeps top fast

            pval = float(2 * (1 - student_t.cdf(abs(t_stat), df=n - 2)))
        rows.append(
            {
                "p": p,
                "p_label": p_label,
                "fold": fold,
                "n": int(n),
                "pearson_r": r,
                "p_value": pval,
            }
        )
    return pd.DataFrame(rows)
