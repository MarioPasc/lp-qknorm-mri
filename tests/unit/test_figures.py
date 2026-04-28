"""Tests for ``lpqknorm.analysis.figures`` and ``style``."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest


matplotlib.use("Agg")

from lpqknorm.analysis.bootstrap import AnalysisError
from lpqknorm.analysis.figures import (
    _toy_delta,
    fig_mechanism_chain,
    fig_per_patient_effect,
    fig_probe_trajectory,
    fig_small_recall_vs_p,
    fig_stratified_dice,
    fig_toy_model_prediction,
)


# ---------------------------------------------------------------------------
# Toy model
# ---------------------------------------------------------------------------


def test_toy_delta_known_values() -> None:
    """``Δ(2, s, d_k) = 1 − sqrt(s/d_k)`` (since s^{1−2/2} = 1)."""
    p = 2.0
    d_k = 16
    s = 4
    expected = 1.0 - np.sqrt(4.0 / 16.0)
    np.testing.assert_allclose(_toy_delta(np.array([p]), s, d_k), [expected])


def test_toy_delta_has_interior_max() -> None:
    """For sparse inputs (s=2, d_k=24), Δ peaks at p > 2."""
    p_grid = np.linspace(1.5, 8.0, 401)
    delta = _toy_delta(p_grid, s=2, d_k=24)
    p_max = p_grid[int(np.argmax(delta))]
    assert p_max > 2.0


def test_fig_toy_model_writes_pdf(tmp_path: Path) -> None:
    out = fig_toy_model_prediction([2, 4, 8, 16], d_k=24, out=tmp_path / "fig1.pdf")
    assert out.exists()
    assert out.suffix == ".pdf"
    assert out.stat().st_size > 100


def test_fig_toy_model_writes_png(tmp_path: Path) -> None:
    out = fig_toy_model_prediction([4, 8], d_k=24, out=tmp_path / "fig1.png")
    assert out.exists()


def test_fig_toy_model_pdf_byte_determinism(tmp_path: Path) -> None:
    """Spec acceptance test 6: two consecutive runs produce equal bytes.

    Even when two PDFs differ in incidental metadata bytes, the
    matplotlib content stream should be identical for the same input
    when timestamps are suppressed.  This test compares SHA-256 hashes.
    """
    p1 = fig_toy_model_prediction([2, 4, 8, 16], d_k=24, out=tmp_path / "a.pdf")
    p2 = fig_toy_model_prediction([2, 4, 8, 16], d_k=24, out=tmp_path / "b.pdf")
    h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
    assert h1 == h2


def test_fig_toy_model_rejects_invalid_d_k(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError):
        fig_toy_model_prediction([2, 4], d_k=1, out=tmp_path / "x.pdf")


def test_fig_toy_model_rejects_oversized_sparsity(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError):
        fig_toy_model_prediction([4, 30], d_k=24, out=tmp_path / "x.pdf")


# ---------------------------------------------------------------------------
# Empirical figures
# ---------------------------------------------------------------------------


def _mock_per_patient(n_patients: int = 18) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for fold in (0, 1, 2):
        for p, p_label in [
            ("vanilla", "vanilla"),
            (2.0, "2.0"),
            (3.0, "3.0"),
            (4.0, "4.0"),
        ]:
            for sid_i in range(n_patients):
                sid = f"S{sid_i:03d}"
                stratum = ("small", "medium", "large")[sid_i % 3]
                base = 0.6 + (0 if p_label == "vanilla" else 0.05)
                rows.append(
                    {
                        "p": p,
                        "p_label": p_label,
                        "fold": fold,
                        "subject_id": sid,
                        "volume_stratum": stratum,
                        "dice": float(np.clip(base + rng.normal(0, 0.05), 0, 1)),
                        "iou": float(np.clip(base - 0.02 + rng.normal(0, 0.05), 0, 1)),
                        "lesion_recall": float(
                            np.clip(base + 0.05 + rng.normal(0, 0.05), 0, 1)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _mock_probes(n_patients: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for fold in (0, 1, 2):
        for p, p_label in [(2.0, "2.0"), (3.0, "3.0"), (4.0, "4.0")]:
            for block in ("block_0_wmsa", "block_1_swmsa"):
                for probe in (
                    "feature_peakiness_q",
                    "attention_entropy",
                    "lesion_attention_mass",
                    "lesion_background_logit_gap",
                    "attention_mask_iou",
                ):
                    for sid_i in range(n_patients):
                        rows.append(
                            {
                                "p": p,
                                "p_label": p_label,
                                "fold": fold,
                                "block": block,
                                "subject_id": f"S{sid_i:03d}",
                                "probe_name": probe,
                                "value": float(rng.uniform(0.1, 0.9)),
                            }
                        )
    return pd.DataFrame(rows)


def test_fig_stratified_dice_outputs_file(tmp_path: Path) -> None:
    out = fig_stratified_dice(_mock_per_patient(), tmp_path / "fig2.pdf")
    assert out.exists()


def test_fig_small_recall_vs_p_outputs_file(tmp_path: Path) -> None:
    out = fig_small_recall_vs_p(_mock_per_patient(), tmp_path / "fig3.pdf")
    assert out.exists()


def test_fig_probe_trajectory_outputs_file(tmp_path: Path) -> None:
    out = fig_probe_trajectory(_mock_probes(), tmp_path / "fig4.pdf")
    assert out.exists()


def test_fig_mechanism_chain_outputs_file(tmp_path: Path) -> None:
    out = fig_mechanism_chain(
        _mock_per_patient(), _mock_probes(), tmp_path / "fig5.pdf"
    )
    assert out.exists()


def test_fig_per_patient_effect_outputs_file(tmp_path: Path) -> None:
    out = fig_per_patient_effect(_mock_per_patient(), tmp_path / "fig6.pdf")
    assert out.exists()


def test_fig_probe_trajectory_rejects_empty() -> None:
    with pytest.raises(AnalysisError):
        fig_probe_trajectory(pd.DataFrame(), Path("/tmp/x.pdf"))


def test_fig_per_patient_effect_rejects_missing_p2(tmp_path: Path) -> None:
    pp = _mock_per_patient()
    pp = pp[pp["p_label"] != "2.0"]
    with pytest.raises(AnalysisError):
        fig_per_patient_effect(pp, tmp_path / "x.pdf")
