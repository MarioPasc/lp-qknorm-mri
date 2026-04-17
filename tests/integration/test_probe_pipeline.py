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
    LinearProbe,
    ProbeRecorder,
    SpatialLocalizationError,
    SpectralProbe,
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
    """Return all eight probe instances."""
    return [
        FeaturePeakiness("q"),
        FeaturePeakiness("k"),
        AttentionEntropy(),
        LesionAttentionMass(),
        LesionBackgroundLogitGap(),
        AttentionMaskIoU(),
        SpatialLocalizationError(),
        LinearProbe(n_splits=3, min_samples_per_class=5),
        SpectralProbe(min_samples=4),
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

                # New Phase-4 keys (Probes 6, 7, 8 + controls).
                for key in [
                    "spatial_localization_error",
                    "lp_balanced_accuracy",
                    "lp_weight_sparsity",
                    "lp_margin",
                    "pr_lesion",
                    "pr_background",
                    "eigenvalues_lesion",
                    "eigenvalues_background",
                    "rel_pos_bias",
                    "rel_pos_bias_entropy",
                    "attention_full",
                    "logits_full",
                ]:
                    assert key in grp, f"{key} missing from {block}"

            # Metadata attributes
            assert f["metadata"].attrs["epoch_tag"] == "test"
            assert f["metadata"].attrs["window_size"] == 7

            # /inputs group with image, mask, subject_id, slice_index.
            assert "inputs" in f
            for key in ("image", "mask", "subject_id", "slice_index"):
                assert key in f["inputs"]


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


class TestFloat16RoundTrip:
    """AT10: float16 storage of attention preserves row-sum and entropy."""

    def test_attention_row_sum_and_entropy_preserved(self, tmp_path: Path) -> None:
        model = _build_model()
        model.eval()
        dm = _build_dm()
        recorder = ProbeRecorder(
            probes=_all_probes(),
            output_dir=tmp_path,
            n_probe_samples=2,
        )
        out = recorder.run(model, dm.val_dataloader(), epoch_tag="fp16", device="cpu")
        with h5py.File(out) as f:
            attn = f["block_0_wmsa/attention_full"][()]
            assert attn.dtype == np.float16
            # Row sums close to 1.
            row_sum = attn.astype(np.float32).sum(axis=-1)
            assert np.allclose(row_sum, 1.0, atol=1e-2)

            # Entropy recomputed from fp16 attention vs stored fp32 entropy.
            safe = np.clip(attn.astype(np.float32), 1e-9, None)
            h16 = -(safe * np.log(safe)).sum(axis=-1)
            h_stored = f["block_0_wmsa/entropy"][()]
            # Median abs relative error is small.
            rel = np.abs(h16.ravel()[: h_stored.shape[0]] - h_stored) / (
                np.abs(h_stored) + 1e-6
            )
            assert np.median(rel) < 5e-3


class TestAlphaLoggerAT11:
    """AT11: AlphaLogger produces strictly increasing steps."""

    def test_alpha_log_monotonic(self, tmp_path: Path) -> None:
        import json

        from lpqknorm.training.callbacks import AlphaLogger

        class _FakeTrainer:
            global_step = 0
            current_epoch = 0

        class _FakeModule:
            def __init__(self, model: torch.nn.Module) -> None:
                self.model = model

        model = _build_model()
        trainer = _FakeTrainer()
        mod = _FakeModule(model)
        cb = AlphaLogger(run_dir=tmp_path, p_value=3.0, fold=0)
        cb.on_fit_start(trainer, mod)  # type: ignore[arg-type]
        for s in range(5):
            trainer.global_step = s
            cb.on_train_batch_end(trainer, mod, None, None, s)  # type: ignore[arg-type]

        jsonl = tmp_path / "probes" / "alpha_trajectory.jsonl"
        assert jsonl.exists()
        records = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert len(records) > 0
        steps = [r["step"] for r in records]
        # Steps are non-decreasing (two blocks per step → equal pairs).
        from itertools import pairwise

        assert all(b >= a for a, b in pairwise(steps))
        # Alpha values are finite and positive.
        assert all(r["alpha"] > 0 for r in records)
