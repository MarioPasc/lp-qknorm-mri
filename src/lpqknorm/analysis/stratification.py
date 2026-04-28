"""Patient-level lesion-volume stratum attachment.

Phase 1 stores ``volume_stratum`` (small / medium / large) inside each
run's ``test_per_patient.parquet`` directly, so most production
analyses do not need a separate join.  The function below remains the
canonical entry point because it (a) supplies a ``stratum_path`` join
when an external CSV is preferred and (b) validates the post-join
schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lpqknorm.analysis.bootstrap import AnalysisError


VALID_STRATA: frozenset[str] = frozenset({"small", "medium", "large"})


def attach_strata(
    per_patient: pd.DataFrame,
    strata_path: Path | str | None = None,
    *,
    column: str = "volume_stratum",
) -> pd.DataFrame:
    """Attach (or validate) the per-patient lesion-volume stratum.

    Behaviour
    ---------
    * If ``strata_path`` is provided, it must point to a CSV /
      parquet table with columns ``[subject_id, <column>]``; the
      table is left-joined onto *per_patient* by ``subject_id``.
      Any existing ``column`` is overwritten.
    * If ``strata_path`` is None, *per_patient* must already carry the
      strata column; the function then validates that values lie in
      ``{"small", "medium", "large"}`` (case-insensitive).
    * Row count is invariant across the join (no patients are
      dropped).  A patient missing from the strata table is left with
      ``NaN`` and a ``StratificationError`` would be raised by callers
      that need every patient classified.

    Parameters
    ----------
    per_patient : pd.DataFrame
        Patient-level frame; must contain ``subject_id``.
    strata_path : Path or str or None
        Optional external strata file (CSV or parquet).
    column : str, default ``"volume_stratum"``
        Name of the stratum column in both inputs and output.

    Returns
    -------
    pd.DataFrame
        Same length as *per_patient*, guaranteed to carry *column*.

    Raises
    ------
    AnalysisError
        If required columns are missing or strata values are invalid.
    """
    if "subject_id" not in per_patient.columns:
        raise AnalysisError("per_patient must contain a 'subject_id' column")
    n_in = len(per_patient)
    out = per_patient.copy()

    if strata_path is not None:
        path = Path(strata_path)
        if not path.exists():
            raise AnalysisError(f"strata file not found: {path}")
        if path.suffix.lower() in {".parquet", ".pq"}:
            ext = pd.read_parquet(path)
        else:
            ext = pd.read_csv(path)
        if "subject_id" not in ext.columns or column not in ext.columns:
            raise AnalysisError(
                f"strata file {path} must have columns ['subject_id', '{column}']"
            )
        if column in out.columns:
            out = out.drop(columns=[column])
        out = out.merge(ext[["subject_id", column]], on="subject_id", how="left")
    elif column not in out.columns:
        raise AnalysisError(
            f"per_patient is missing '{column}' and no strata_path was given"
        )

    if len(out) != n_in:
        raise AnalysisError(
            f"row count changed during strata attachment: {n_in} -> {len(out)}"
        )

    raw = out[column].dropna().astype(str).str.lower().unique()
    bad = set(raw) - VALID_STRATA
    if bad:
        raise AnalysisError(
            f"invalid stratum value(s): {sorted(bad)} (allowed: {sorted(VALID_STRATA)})"
        )
    out[column] = (
        out[column].astype(str).str.lower().where(out[column].notna(), out[column])
    )
    return out
