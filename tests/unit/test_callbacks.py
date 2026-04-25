"""Tests for ManifestCallback periodic flush and resume behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from lpqknorm.training.callbacks import AlphaLogger, ManifestCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(epoch: int = 5, metrics: dict | None = None) -> MagicMock:
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.callback_metrics = metrics or {}
    return trainer


def _make_manifest_init() -> dict:
    return {
        "run_id": "test-run-id",
        "experiment": "test",
        "p": 2.0,
        "fold": 0,
        "seed": 42,
        "git_sha": "abc123",
        "git_dirty": False,
        "git_branch": "main",
        "host": "test-host",
        "gpu_model": "N/A",
        "cuda_version": "N/A",
        "torch_version": torch.__version__,
        "monai_version": "1.5.2",
        "lpqknorm_version": "0.1.0",
        "config_hash": "deadbeef",
        "split_hash": "cafebabe",
        "n_train": 100,
        "n_val": 50,
        "n_test": 30,
        "init_scheme": "scratch_trunc_normal",
        "linear_init_std": 0.02,
        "alpha_init_scheme": "log_dk",
        "alpha_init_fixed": None,
        "init_spec_hash": "0" * 64,
    }


# ---------------------------------------------------------------------------
# ManifestCallback — fresh start
# ---------------------------------------------------------------------------


class TestManifestCallbackFresh:
    """Test ManifestCallback on a fresh (non-resumed) run."""

    def test_on_fit_start_writes_manifest(self, tmp_path: Path) -> None:
        cb = ManifestCallback(tmp_path, _make_manifest_init())
        cb.on_fit_start(_make_trainer(), MagicMock())

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["run_id"] == "test-run-id"
        assert manifest["started_utc"] is not None
        assert manifest["finished_utc"] is None
        assert manifest["final_epoch"] is None
        assert manifest["best_val_dice"] is None

    def test_on_validation_epoch_end_flushes_progress(self, tmp_path: Path) -> None:
        cb = ManifestCallback(tmp_path, _make_manifest_init())
        trainer = _make_trainer(
            epoch=10,
            metrics={
                "val_dice_mean": torch.tensor(0.75),
                "val_lesion_recall_small": torch.tensor(0.60),
            },
        )
        cb.on_fit_start(trainer, MagicMock())
        cb.on_validation_epoch_end(trainer, MagicMock())

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["final_epoch"] == 10
        assert manifest["best_val_dice"] == pytest.approx(0.75, abs=1e-6)
        assert manifest["best_small_recall"] == pytest.approx(0.60, abs=1e-6)
        assert manifest["walltime_sec"] is not None
        assert manifest["finished_utc"] is None

    def test_best_val_dice_tracks_maximum(self, tmp_path: Path) -> None:
        cb = ManifestCallback(tmp_path, _make_manifest_init())
        cb.on_fit_start(_make_trainer(), MagicMock())

        # Epoch 1: dice=0.5
        trainer1 = _make_trainer(
            epoch=1,
            metrics={
                "val_dice_mean": torch.tensor(0.5),
            },
        )
        cb.on_validation_epoch_end(trainer1, MagicMock())

        # Epoch 2: dice=0.7 (improvement)
        trainer2 = _make_trainer(
            epoch=2,
            metrics={
                "val_dice_mean": torch.tensor(0.7),
            },
        )
        cb.on_validation_epoch_end(trainer2, MagicMock())

        # Epoch 3: dice=0.6 (regression — best should stay 0.7)
        trainer3 = _make_trainer(
            epoch=3,
            metrics={
                "val_dice_mean": torch.tensor(0.6),
            },
        )
        cb.on_validation_epoch_end(trainer3, MagicMock())

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["final_epoch"] == 3
        assert manifest["best_val_dice"] == pytest.approx(0.7, abs=1e-6)

    def test_on_fit_end_sets_finished_utc(self, tmp_path: Path) -> None:
        cb = ManifestCallback(tmp_path, _make_manifest_init())
        trainer = _make_trainer(
            epoch=42,
            metrics={
                "val_dice_mean": torch.tensor(0.80),
            },
        )
        cb.on_fit_start(trainer, MagicMock())
        cb.on_fit_end(trainer, MagicMock())

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["finished_utc"] is not None
        assert manifest["final_epoch"] == 42

    def test_nan_metric_does_not_overwrite_best(self, tmp_path: Path) -> None:
        cb = ManifestCallback(tmp_path, _make_manifest_init())
        cb.on_fit_start(_make_trainer(), MagicMock())

        trainer1 = _make_trainer(
            epoch=1,
            metrics={
                "val_dice_mean": torch.tensor(0.65),
            },
        )
        cb.on_validation_epoch_end(trainer1, MagicMock())

        trainer2 = _make_trainer(
            epoch=2,
            metrics={
                "val_dice_mean": torch.tensor(float("nan")),
            },
        )
        cb.on_validation_epoch_end(trainer2, MagicMock())

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["best_val_dice"] == pytest.approx(0.65, abs=1e-6)


# ---------------------------------------------------------------------------
# ManifestCallback — resume
# ---------------------------------------------------------------------------


class TestManifestCallbackResume:
    """Test ManifestCallback resume behaviour."""

    def _write_prior_manifest(self, run_dir: Path) -> None:
        """Write a manifest as if a prior run was killed mid-training."""
        manifest = {
            **_make_manifest_init(),
            "started_utc": "2026-04-22T17:00:00Z",
            "finished_utc": None,
            "walltime_sec": 36000.0,
            "peak_gpu_memory_mb": 1200.0,
            "final_epoch": 30,
            "best_val_dice": 0.72,
            "best_small_recall": 0.55,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def test_resume_preserves_started_utc(self, tmp_path: Path) -> None:
        self._write_prior_manifest(tmp_path)
        cb = ManifestCallback(tmp_path, _make_manifest_init(), resuming=True)
        cb.on_fit_start(_make_trainer(), MagicMock())

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["started_utc"] == "2026-04-22T17:00:00Z"
        assert manifest["best_val_dice"] == pytest.approx(0.72, abs=1e-6)

    def test_resume_accumulates_walltime(self, tmp_path: Path) -> None:
        self._write_prior_manifest(tmp_path)
        cb = ManifestCallback(tmp_path, _make_manifest_init(), resuming=True)
        cb.on_fit_start(_make_trainer(), MagicMock())

        assert cb._walltime_offset == pytest.approx(36000.0, abs=1e-2)

    def test_resume_preserves_best_dice_unless_improved(self, tmp_path: Path) -> None:
        self._write_prior_manifest(tmp_path)
        cb = ManifestCallback(tmp_path, _make_manifest_init(), resuming=True)
        cb.on_fit_start(_make_trainer(), MagicMock())

        # Epoch with lower dice — best should stay 0.72
        trainer = _make_trainer(
            epoch=31,
            metrics={
                "val_dice_mean": torch.tensor(0.70),
            },
        )
        cb.on_validation_epoch_end(trainer, MagicMock())
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["best_val_dice"] == pytest.approx(0.72, abs=1e-6)

        # Epoch with higher dice — best should update
        trainer2 = _make_trainer(
            epoch=35,
            metrics={
                "val_dice_mean": torch.tensor(0.78),
            },
        )
        cb.on_validation_epoch_end(trainer2, MagicMock())
        manifest2 = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest2["best_val_dice"] == pytest.approx(0.78, abs=1e-6)

    def test_resume_fit_end_sets_finished_utc(self, tmp_path: Path) -> None:
        self._write_prior_manifest(tmp_path)
        cb = ManifestCallback(tmp_path, _make_manifest_init(), resuming=True)
        trainer = _make_trainer(
            epoch=50,
            metrics={
                "val_dice_mean": torch.tensor(0.80),
            },
        )
        cb.on_fit_start(trainer, MagicMock())
        cb.on_fit_end(trainer, MagicMock())

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["finished_utc"] is not None
        assert manifest["started_utc"] == "2026-04-22T17:00:00Z"


# ---------------------------------------------------------------------------
# AlphaLogger — resume
# ---------------------------------------------------------------------------


class TestAlphaLoggerResume:
    """Test that AlphaLogger does not truncate on resume."""

    def test_fresh_start_truncates(self, tmp_path: Path) -> None:
        path = tmp_path / "probes" / "alpha_trajectory.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"step": 0, "alpha": 1.0}\n')

        al = AlphaLogger(run_dir=tmp_path, resuming=False)
        al.on_fit_start(_make_trainer(), MagicMock())

        assert path.read_text() == ""

    def test_resume_preserves_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "probes" / "alpha_trajectory.jsonl"
        path.parent.mkdir(parents=True)
        prior = '{"step": 0, "alpha": 1.0}\n{"step": 100, "alpha": 1.5}\n'
        path.write_text(prior)

        al = AlphaLogger(run_dir=tmp_path, resuming=True)
        al.on_fit_start(_make_trainer(), MagicMock())

        assert path.read_text() == prior


# ---------------------------------------------------------------------------
# Resume detection (train.py logic, tested in isolation)
# ---------------------------------------------------------------------------


class TestResumeDetection:
    """Test the resume detection logic from train.py."""

    @staticmethod
    def _detect_resume(run_dir: Path) -> tuple[bool, str | None]:
        """Replicate the resume detection logic from train.py."""
        resume_ckpt: str | None = None
        resuming = False
        last_ckpt = run_dir / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                prior = json.loads(manifest_path.read_text())
                if prior.get("finished_utc") is None:
                    resume_ckpt = str(last_ckpt)
                    resuming = True
        return resuming, resume_ckpt

    def test_no_checkpoint_no_resume(self, tmp_path: Path) -> None:
        resuming, ckpt = self._detect_resume(tmp_path)
        assert resuming is False
        assert ckpt is None

    def test_checkpoint_with_incomplete_manifest_resumes(self, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").write_text("dummy")
        manifest = {"finished_utc": None, "best_val_dice": 0.5}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        resuming, ckpt = self._detect_resume(tmp_path)
        assert resuming is True
        assert ckpt == str(ckpt_dir / "last.ckpt")

    def test_checkpoint_with_completed_manifest_does_not_resume(
        self, tmp_path: Path
    ) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").write_text("dummy")
        manifest = {"finished_utc": "2026-04-24T19:32:07Z", "best_val_dice": 0.75}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        resuming, ckpt = self._detect_resume(tmp_path)
        assert resuming is False
        assert ckpt is None

    def test_checkpoint_without_manifest_does_not_resume(self, tmp_path: Path) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "last.ckpt").write_text("dummy")

        resuming, ckpt = self._detect_resume(tmp_path)
        assert resuming is False
        assert ckpt is None
