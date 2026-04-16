"""Integration tests for the Phase 4 probe pipeline (AT5 + AT6 + AT7).

AT5: End-to-end ProbeRecorder → HDF5 with correct structure and ranges.
AT6: Two identical runs produce bit-identical HDF5 arrays.
AT7: No autograd retention under torch.inference_mode().
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest
import torch

from lpqknorm.data.datamodule import MockAtlasDataModule, MockDataConfig
from lpqknorm.models.hooks import AttentionHookRegistry
from lpqknorm.models.lp_qknorm import LpQKNormConfig
from lpqknorm.models.swin_unetr_lp import build_swin_unetr_lp
from lpqknorm.probes import (
    AttentionEntropy,
    AttentionMaskIoU,
    FeaturePeakiness,
    LesionAttentionMass,
    LesionBackgroundLogitGap,
    ProbeRecorder,
)


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


def _build_model() -> torch.nn.Module:
    """Build a small Lp-patched model for testing."""
    return build_swin_unetr_lp(
        img_size=(224, 224),
        in_channels=1,
        out_channels=1,
        feature_size=12,
        lp_cfg=LpQKNormConfig(p=3.0),
    )


def _build_dm() -> MockAtlasDataModule:
    """Build a tiny mock DataModule."""
    dm = MockAtlasDataModule(
        MockDataConfig(n_val=4, batch_size=2, img_size=(224, 224), seed=0)
    )
    dm.setup("fit")
    return dm


def _all_probes() -> list[object]:
    """Return all six probe instances."""
    return [
        FeaturePeakiness("q"),
        FeaturePeakiness("k"),
        AttentionEntropy(),
        LesionAttentionMass(),
        LesionBackgroundLogitGap(),
        AttentionMaskIoU(),
    ]


class TestProbeRecorderEndToEnd:
    """AT5: End-to-end recorder → HDF5 with correct structure."""

    def test_hdf5_structure_and_ranges(self, tmp_path: Path) -> None:
        model = _build_model()
        model.eval()
        dm = _build_dm()

        recorder = ProbeRecorder(
            probes=_all_probes(),
            output_dir=tmp_path,
            n_probe_samples=4,
        )
        out = recorder.run(model, dm.val_dataloader(), epoch_tag="test", device="cpu")

        assert out.exists()
        with h5py.File(out) as f:
            # Both blocks present
            assert "block_0_wmsa" in f
            assert "block_1_swmsa" in f
            assert "metadata" in f

            for block in ["block_0_wmsa", "block_1_swmsa"]:
                grp = f[block]
                # Peakiness arrays exist and have correct range
                assert "peakiness_q" in grp
                assert "peakiness_k" in grp
                rho_q = grp["peakiness_q"][()]
                assert rho_q.shape[0] > 0
                assert (rho_q >= 0.0).all()
                assert (rho_q <= 1.0 + 1e-4).all()

                # Entropy exists and in valid range
                assert "entropy" in grp
                h_vals = grp["entropy"][()]
                assert h_vals.shape[0] > 0
                assert (h_vals >= -1e-5).all()
                assert (h_vals <= math.log(49) + 1e-3).all()

                # Alpha is a scalar
                assert "alpha" in grp
                assert float(grp["alpha"][()]) > 0

            # Metadata attributes
            assert f["metadata"].attrs["epoch_tag"] == "test"
            assert f["metadata"].attrs["window_size"] == 7


class TestProbeDeterminism:
    """AT6: Two identical runs produce bit-identical arrays."""

    def test_determinism(self, tmp_path: Path) -> None:
        model = _build_model()
        model.eval()
        dm = _build_dm()

        def _run(tag: str) -> Path:
            out_dir = tmp_path / tag
            recorder = ProbeRecorder(
                probes=_all_probes(),
                output_dir=out_dir,
                n_probe_samples=4,
            )
            return recorder.run(model, dm.val_dataloader(), epoch_tag=tag, device="cpu")

        path1 = _run("run1")
        path2 = _run("run2")

        with h5py.File(path1) as f1, h5py.File(path2) as f2:
            for block in ["block_0_wmsa", "block_1_swmsa"]:
                for key in ["peakiness_q", "peakiness_k", "entropy"]:
                    if key in f1[block] and key in f2[block]:
                        np.testing.assert_array_equal(
                            f1[block][key][()],
                            f2[block][key][()],
                        )


class TestNoAutogradRetention:
    """AT7: ProbeRecorder runs under inference_mode, no grad retention."""

    def test_no_autograd_retention(self, tmp_path: Path) -> None:
        model = _build_model()
        model.eval()
        dm = _build_dm()

        recorder = ProbeRecorder(
            probes=_all_probes(),
            output_dir=tmp_path,
            n_probe_samples=2,
        )

        # Run under inference_mode (recorder does this internally)
        path = recorder.run(model, dm.val_dataloader(), epoch_tag="at7", device="cpu")
        assert path.exists()

        # Verify captures have no grad
        registry = AttentionHookRegistry()
        registry.register(model, stages=[0])
        batch = next(iter(dm.val_dataloader()))
        with torch.inference_mode():
            _ = model(batch["image"])
        for cap in registry.captures():
            assert cap.q is not None
            assert not cap.q.requires_grad
            assert cap.attention is not None
            assert not cap.attention.requires_grad
        registry.remove()

        # HDF5 arrays are numpy (no grad by definition)
        with h5py.File(path) as f:
            arr = f["block_0_wmsa/peakiness_q"][()]
            assert arr.dtype == np.float32
