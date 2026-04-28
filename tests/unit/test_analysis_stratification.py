"""Tests for ``lpqknorm.analysis.stratification`` (Phase 5 join layer).

The Phase 1 stratification module already has its own tests under
``test_stratification.py``; this file targets the analysis-side join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from lpqknorm.analysis.bootstrap import AnalysisError
from lpqknorm.analysis.stratification import attach_strata


if TYPE_CHECKING:
    from pathlib import Path


def _mock_per_patient(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "subject_id": [f"S{i:03d}" for i in range(n)],
            "dice": rng.uniform(0.5, 0.95, n),
            "volume_stratum": rng.choice(["small", "medium", "large"], size=n),
        }
    )


def test_attach_strata_passthrough_when_column_present() -> None:
    pp = _mock_per_patient(20)
    out = attach_strata(pp)
    assert len(out) == len(pp)
    assert set(out["volume_stratum"]).issubset({"small", "medium", "large"})


def test_attach_strata_external_csv_join(tmp_path: Path) -> None:
    pp = _mock_per_patient(30).drop(columns=["volume_stratum"])
    strata = pd.DataFrame(
        {
            "subject_id": pp["subject_id"],
            "volume_stratum": np.where(
                np.arange(30) < 10,
                "small",
                np.where(np.arange(30) < 20, "medium", "large"),
            ),
        }
    )
    csv = tmp_path / "strata.csv"
    strata.to_csv(csv, index=False)
    out = attach_strata(pp, strata_path=csv)
    assert len(out) == len(pp)
    assert (out["volume_stratum"].iloc[:10] == "small").all()
    assert (out["volume_stratum"].iloc[10:20] == "medium").all()
    assert (out["volume_stratum"].iloc[20:30] == "large").all()


def test_attach_strata_external_parquet(tmp_path: Path) -> None:
    pp = _mock_per_patient(8).drop(columns=["volume_stratum"])
    strata = pd.DataFrame(
        {"subject_id": pp["subject_id"], "volume_stratum": ["small"] * 8}
    )
    pq = tmp_path / "strata.parquet"
    strata.to_parquet(pq)
    out = attach_strata(pp, strata_path=pq)
    assert (out["volume_stratum"] == "small").all()


def test_attach_strata_overrides_existing_column(tmp_path: Path) -> None:
    pp = _mock_per_patient(5)
    pp["volume_stratum"] = ["small"] * 5
    strata = pd.DataFrame(
        {"subject_id": pp["subject_id"], "volume_stratum": ["large"] * 5}
    )
    csv = tmp_path / "s.csv"
    strata.to_csv(csv, index=False)
    out = attach_strata(pp, strata_path=csv)
    assert (out["volume_stratum"] == "large").all()


def test_attach_strata_rejects_missing_subject_id() -> None:
    bad = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(AnalysisError):
        attach_strata(bad)


def test_attach_strata_rejects_invalid_value() -> None:
    bad = pd.DataFrame({"subject_id": ["A", "B"], "volume_stratum": ["small", "huge"]})
    with pytest.raises(AnalysisError):
        attach_strata(bad)


def test_attach_strata_missing_when_no_column_or_path() -> None:
    pp = _mock_per_patient(5).drop(columns=["volume_stratum"])
    with pytest.raises(AnalysisError):
        attach_strata(pp)
