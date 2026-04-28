"""End-to-end test for ``lpqknorm.cli.analyze`` (Phase 5 acceptance test 5)."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from lpqknorm.cli.analyze import main as analyze_main

# Reuse the mock-tree builder from the aggregation test module.
from tests.unit.test_aggregation import _build_mock_results


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
def test_analyze_cli_in_process(tmp_path: Path) -> None:
    """Run the analyze CLI in-process; check tables, figures, and manifest."""
    mock_root = _build_mock_results(tmp_path / "results", write_probes=True)
    out = tmp_path / "outputs"
    rc = analyze_main(
        [
            "--results-root",
            str(mock_root),
            "--output",
            str(out),
            "--n-resamples",
            "200",
            "--seed",
            "0",
        ]
    )
    assert rc == 0

    assert (out / "tables" / "stratified_metrics.parquet").exists()
    assert (out / "tables" / "per_fold_argmax.parquet").exists()
    assert (out / "figures" / "fig1_toy_model_prediction.pdf").exists()
    assert (out / "figures" / "fig2_stratified_dice.pdf").exists()
    assert (out / "figures" / "fig3_small_recall_vs_p.pdf").exists()
    assert (out / "figures" / "fig4_probe_trajectory.pdf").exists()

    manifest_path = out / "analysis_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_runs"] == 6
    assert manifest["p_baseline"] == "2.0"
    assert manifest["seed"] == 0
    assert manifest["d_k"] == 24
    assert manifest["figures"]["fig1_toy_model_prediction"].endswith(
        "fig1_toy_model_prediction.pdf"
    )

    # Bootstrap raw distributions.
    headline = list((out / "bootstrap").glob("lesion_recall_small_*_vs_p=2.0.npz"))
    assert headline, "no bootstrap distributions written for the headline metric"


@pytest.mark.integration
def test_analyze_cli_accepts_hydra_style_kv(tmp_path: Path) -> None:
    """The Hydra-style ``key=value`` form must be accepted."""
    mock_root = _build_mock_results(tmp_path / "results", write_probes=False)
    out = tmp_path / "outputs"
    rc = analyze_main(
        [
            f"results_root={mock_root}",
            f"output={out}",
            "n_resamples=100",
            "seed=0",
        ]
    )
    assert rc == 0
    assert (out / "analysis_manifest.json").exists()
    assert (out / "figures" / "fig1_toy_model_prediction.pdf").exists()


@pytest.mark.integration
def test_analyze_cli_subprocess(tmp_path: Path) -> None:
    """Same flow via a child process — exercises ``python -m`` invocation."""
    mock_root = _build_mock_results(tmp_path / "results", write_probes=False)
    out = tmp_path / "outputs"
    cp = subprocess.run(
        [
            sys.executable,
            "-m",
            "lpqknorm.cli.analyze",
            f"results_root={mock_root}",
            f"output={out}",
            "n_resamples=100",
            "seed=0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr
    assert (out / "analysis_manifest.json").exists()
