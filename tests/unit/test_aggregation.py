"""Tests for ``lpqknorm.analysis.aggregation`` — results-tree walker."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import pytest

from lpqknorm.analysis.aggregation import (
    load_per_patient,
    load_probes,
    load_runs,
)
from lpqknorm.analysis.bootstrap import AnalysisError


if TYPE_CHECKING:
    from pathlib import Path


def _build_mock_results(
    root: Path,
    *,
    p_values: tuple[str, ...] = ("vanilla", "2.0", "3.0"),
    folds: tuple[int, ...] = (0, 1),
    n_patients: int = 6,
    seed: int = 0,
    write_probes: bool = False,
) -> Path:
    """Create a synthetic results tree compatible with Phase 3/4 outputs."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for p in p_values:
        for f in folds:
            run = root / f"p={p}" / f"fold={f}" / "seed=20260216"
            (run / "metrics").mkdir(parents=True, exist_ok=True)
            (run / "checkpoints").mkdir(parents=True, exist_ok=True)
            if write_probes:
                (run / "probes").mkdir(parents=True, exist_ok=True)

            # Per-patient parquet with multiple slices per patient
            # so the aggregator's mean reduction has work to do.
            rows = []
            for sid_i in range(n_patients):
                sid = f"S{sid_i:03d}"
                stratum = ("small", "medium", "large")[sid_i % 3]
                base_dice = 0.6 + 0.05 * (p_values.index(p) - 1) + rng.normal(0, 0.02)
                for _slice_i in range(4):
                    rows.append(
                        {
                            "epoch": 86,
                            "subject_id": sid,
                            "volume_stratum": stratum,
                            "dice": float(
                                np.clip(base_dice + rng.normal(0, 0.01), 0, 1)
                            ),
                            "iou": float(
                                np.clip(base_dice - 0.02 + rng.normal(0, 0.01), 0, 1)
                            ),
                            "lesion_recall": float(
                                np.clip(base_dice + 0.05 + rng.normal(0, 0.02), 0, 1)
                            ),
                            "false_positives_per_slice": int(rng.integers(0, 3)),
                        }
                    )
            df = pd.DataFrame(rows)
            df.to_parquet(run / "metrics" / "test_per_patient.parquet")
            df.to_parquet(run / "metrics" / "val_per_patient.parquet")

            (run / "manifest.json").write_text(
                json.dumps(
                    {
                        "experiment": "p_sweep_test",
                        "dataset_name": "brats_men",
                        "walltime_sec": 100.0,
                        "final_epoch": 86,
                        "best_val_dice": 0.8,
                        "best_small_recall": 0.7,
                    }
                )
            )

            if write_probes:
                with h5py.File(run / "probes" / "epoch_best_dice.h5", "w") as h:
                    inputs = h.create_group("inputs")
                    sids = np.array([f"S{i:03d}".encode() for i in range(n_patients)])
                    inputs.create_dataset("subject_id", data=sids)
                    for block in ("block_0_wmsa", "block_1_swmsa"):
                        grp = h.create_group(block)
                        for probe in (
                            "feature_peakiness_q",
                            "attention_entropy",
                            "lesion_attention_mass",
                            "lesion_background_logit_gap",
                            "attention_mask_iou",
                        ):
                            arr = rng.uniform(0.1, 0.9, size=(n_patients, 8)).astype(
                                np.float32
                            )
                            grp.create_dataset(probe, data=arr)
    return root


@pytest.fixture
def mock_results(tmp_path: Path) -> Path:
    return _build_mock_results(tmp_path / "results", write_probes=True)


def test_load_runs_discovers_all_pairs(mock_results: Path) -> None:
    df = load_runs(mock_results)
    # 3 p-values × 2 folds = 6 runs.
    assert len(df) == 6
    assert set(df["fold"].unique()) == {0, 1}
    assert "vanilla" in df["p_label"].unique()
    assert (df["dataset_name"] == "brats_men").all()
    assert (df["experiment"] == "p_sweep_test").all()


def test_load_runs_filters_by_experiment(mock_results: Path) -> None:
    df = load_runs(mock_results, experiment="p_sweep_test")
    assert len(df) == 6
    df = load_runs(mock_results, experiment="other")
    assert len(df) == 0


def test_load_runs_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError):
        load_runs(tmp_path / "does_not_exist")


def test_load_per_patient_aggregates_to_one_row_per_patient(mock_results: Path) -> None:
    pp = load_per_patient(mock_results)
    assert not pp.empty
    # 6 runs × 6 patients = 36 rows after aggregation.
    assert len(pp) == 36
    assert {"dice", "iou", "lesion_recall"}.issubset(pp.columns)
    # Each (run_id, subject_id) appears exactly once.
    counts = pp.groupby(["run_id", "subject_id"]).size()
    assert (counts == 1).all()


def test_load_per_patient_raw_keeps_slice_rows(mock_results: Path) -> None:
    raw = load_per_patient(mock_results, aggregate=False)
    # 6 runs × 6 patients × 4 slices = 144 rows.
    assert len(raw) == 144


def test_load_per_patient_handles_missing_files(tmp_path: Path) -> None:
    """Empty results root yields an empty frame, not an exception."""
    empty = tmp_path / "empty"
    empty.mkdir()
    out = load_per_patient(empty)
    assert out.empty


def test_load_probes_returns_long_format(mock_results: Path) -> None:
    df = load_probes(mock_results)
    assert not df.empty
    assert {"p_label", "fold", "block", "probe_name", "subject_id", "value"}.issubset(
        df.columns
    )
    assert df["block"].isin({"block_0_wmsa", "block_1_swmsa"}).all()
    # 6 runs × 2 blocks × 5 probes × 6 patients = 360 rows.
    assert len(df) == 360


def test_load_probes_returns_empty_when_missing(tmp_path: Path) -> None:
    no_probe = _build_mock_results(tmp_path / "noprobe", write_probes=False)
    out = load_probes(no_probe)
    assert out.empty
