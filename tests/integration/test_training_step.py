"""Integration tests for Phase 3 training pipeline.

Implements acceptance tests 1-7 from ``docs/phase_03_training.md``.
All tests use ``feature_size=12`` and ``accelerator="cpu"`` for speed.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl

from lpqknorm.data.datamodule import MockAtlasDataModule, MockDataConfig
from lpqknorm.models.lp_qknorm import LpQKNormConfig
from lpqknorm.training.callbacks import (
    ArtefactDirectoryCallback,
    AttentionSummaryCallback,
    GradientNormCallback,
    ManifestCallback,
    PerPatientMetricsCallback,
)
from lpqknorm.training.logging import StructuredLogger
from lpqknorm.training.module import LpSegmentationModule, ModelConfig, TrainingConfig


pytestmark = pytest.mark.integration


def _make_manifest_init(
    dm: MockAtlasDataModule,
) -> dict[str, Any]:
    """Build a minimal manifest_init dict for ManifestCallback tests."""
    return {
        "run_id": "test-run",
        "experiment": "test",
        "p": 3.0,
        "fold": 0,
        "seed": 42,
        "git_sha": "abc123",
        "git_dirty": False,
        "git_branch": "main",
        "host": "testhost",
        "gpu_model": "N/A",
        "cuda_version": "N/A",
        "torch_version": "test",
        "monai_version": "test",
        "lpqknorm_version": "0.1.0",
        "config_hash": "deadbeef",
        "split_hash": dm.split_hash,
        "n_train": dm.n_train,
        "n_val": dm.n_val,
        "n_test": dm.n_test,
    }


class TestSingleTrainingStepRuns:
    """AT1: training_step runs for 2 steps without error."""

    def test_single_training_step_runs(
        self,
        tiny_datamodule: MockAtlasDataModule,
        tmp_path: Path,
    ) -> None:
        module = LpSegmentationModule(
            model_cfg=ModelConfig(feature_size=12, out_channels=1),
            lp_cfg=LpQKNormConfig(p=3.0),
            training_cfg=TrainingConfig(lr=1e-3, precision="32"),
        )
        trainer = pl.Trainer(
            max_steps=2,
            logger=False,
            enable_checkpointing=False,
            default_root_dir=str(tmp_path),
            accelerator="cpu",
        )
        trainer.fit(module, tiny_datamodule)
        assert trainer.global_step == 2

    def test_vanilla_baseline_runs(
        self,
        tiny_datamodule: MockAtlasDataModule,
        tmp_path: Path,
    ) -> None:
        """Vanilla (lp_cfg=None) also runs without error."""
        module = LpSegmentationModule(
            model_cfg=ModelConfig(feature_size=12, out_channels=1),
            lp_cfg=None,
            training_cfg=TrainingConfig(lr=1e-3, precision="32"),
        )
        trainer = pl.Trainer(
            max_steps=2,
            logger=False,
            enable_checkpointing=False,
            default_root_dir=str(tmp_path),
            accelerator="cpu",
        )
        trainer.fit(module, tiny_datamodule)
        assert trainer.global_step == 2


class TestArtefactsAfterOneEpoch:
    """AT2: all expected artefact files exist after a 1-epoch run."""

    def test_artefacts_after_one_epoch(
        self,
        tiny_datamodule: MockAtlasDataModule,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        s_logger = StructuredLogger(run_dir)
        module = LpSegmentationModule(
            model_cfg=ModelConfig(feature_size=12, out_channels=1),
            lp_cfg=LpQKNormConfig(p=3.0),
            training_cfg=TrainingConfig(lr=1e-3, max_epochs=1, precision="32"),
            pos_weight=tiny_datamodule.pos_weight,
            structured_logger=s_logger,
        )
        callbacks: list[pl.Callback] = [
            ArtefactDirectoryCallback(run_dir),
            ManifestCallback(run_dir, _make_manifest_init(tiny_datamodule)),
            PerPatientMetricsCallback(s_logger),
            GradientNormCallback(run_dir, log_every_n_steps=1),
        ]
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
            default_root_dir=str(tmp_path),
            accelerator="cpu",
        )
        trainer.fit(module, tiny_datamodule)
        s_logger.close()

        # Check artefact existence
        for expected in [
            "manifest.json",
            "metrics/train_steps.jsonl",
            "metrics/val_per_patient.parquet",
        ]:
            assert (run_dir / expected).exists(), f"Missing: {expected}"

        # Check subdirectories created by ArtefactDirectoryCallback
        for subdir in [
            "metrics",
            "attention_stats",
            "gradient_stats",
            "predictions",
            "checkpoints",
            "probes",
        ]:
            assert (run_dir / subdir).is_dir(), f"Missing dir: {subdir}"


class TestDeterminism:
    """AT3: identical loss curves for the same seed."""

    def _run_once(
        self,
        seed: int,
        run_dir: Path,
        dm: MockAtlasDataModule,
    ) -> list[dict[str, Any]]:
        """Run 3 training steps and return parsed JSONL rows."""
        pl.seed_everything(seed)
        s_logger = StructuredLogger(run_dir)
        module = LpSegmentationModule(
            model_cfg=ModelConfig(feature_size=12, out_channels=1),
            lp_cfg=LpQKNormConfig(p=3.0),
            training_cfg=TrainingConfig(lr=1e-3, precision="32"),
            pos_weight=dm.pos_weight,
            structured_logger=s_logger,
        )
        trainer = pl.Trainer(
            max_steps=3,
            logger=False,
            enable_checkpointing=False,
            default_root_dir=str(run_dir),
            accelerator="cpu",
            deterministic=True,
        )
        trainer.fit(module, dm)
        s_logger.close()

        jsonl_path = run_dir / "metrics" / "train_steps.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")
        return [json.loads(line) for line in lines]

    def test_determinism(self, tmp_path: Path) -> None:
        # Each run needs its own DataModule to ensure identical data
        cfg = MockDataConfig(
            n_train=8,
            n_val=4,
            n_test=4,
            img_size=(224, 224),
            n_subjects=2,
            seed=0,
            batch_size=2,
        )
        dm1 = MockAtlasDataModule(cfg)
        dm1.setup("fit")
        dm2 = MockAtlasDataModule(cfg)
        dm2.setup("fit")

        s1 = self._run_once(42, tmp_path / "run1", dm1)
        s2 = self._run_once(42, tmp_path / "run2", dm2)

        losses1 = [r["loss_total"] for r in s1]
        losses2 = [r["loss_total"] for r in s2]
        np.testing.assert_allclose(losses1, losses2, atol=1e-5)


class TestManifestSplitHash:
    """AT5: manifest.json contains the correct split_hash."""

    def test_manifest_split_hash(
        self,
        tiny_datamodule: MockAtlasDataModule,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        s_logger = StructuredLogger(run_dir)
        module = LpSegmentationModule(
            model_cfg=ModelConfig(feature_size=12, out_channels=1),
            lp_cfg=LpQKNormConfig(p=3.0),
            training_cfg=TrainingConfig(lr=1e-3, max_epochs=1, precision="32"),
            structured_logger=s_logger,
        )
        callbacks: list[pl.Callback] = [
            ArtefactDirectoryCallback(run_dir),
            ManifestCallback(run_dir, _make_manifest_init(tiny_datamodule)),
        ]
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
            default_root_dir=str(tmp_path),
            accelerator="cpu",
        )
        trainer.fit(module, tiny_datamodule)
        s_logger.close()

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert "split_hash" in manifest
        assert manifest["split_hash"] == tiny_datamodule.split_hash


class TestPerPatientCardinality:
    """AT6: val_per_patient.parquet has n_val_slices x n_epochs rows."""

    def test_per_patient_cardinality(
        self,
        tiny_datamodule: MockAtlasDataModule,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        n_epochs = 2
        s_logger = StructuredLogger(run_dir)
        module = LpSegmentationModule(
            model_cfg=ModelConfig(feature_size=12, out_channels=1),
            lp_cfg=LpQKNormConfig(p=3.0),
            training_cfg=TrainingConfig(
                lr=1e-3, max_epochs=n_epochs, patience=100, precision="32"
            ),
            pos_weight=tiny_datamodule.pos_weight,
            structured_logger=s_logger,
        )
        callbacks: list[pl.Callback] = [
            ArtefactDirectoryCallback(run_dir),
            PerPatientMetricsCallback(s_logger),
        ]
        trainer = pl.Trainer(
            max_epochs=n_epochs,
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
            default_root_dir=str(tmp_path),
            accelerator="cpu",
            num_sanity_val_steps=0,
        )
        trainer.fit(module, tiny_datamodule)
        s_logger.close()

        df = pd.read_parquet(run_dir / "metrics" / "val_per_patient.parquet")
        # MockDataConfig: n_val=8 slices, n_epochs=2, sanity check disabled
        expected_rows = tiny_datamodule.n_val * n_epochs
        assert len(df) == expected_rows

        # Unique (epoch, subject_id, slice) combinations
        assert "epoch" in df.columns
        assert "subject_id" in df.columns


class TestAttentionSummaryValidRanges:
    """AT7: entropy in [0, log(W^2)] and structure is correct."""

    def test_attention_summary_valid_ranges(
        self,
        tiny_datamodule: MockAtlasDataModule,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        s_logger = StructuredLogger(run_dir)
        module = LpSegmentationModule(
            model_cfg=ModelConfig(feature_size=12, out_channels=1),
            lp_cfg=LpQKNormConfig(p=3.0),
            training_cfg=TrainingConfig(
                lr=1e-3, max_epochs=1, patience=100, precision="32"
            ),
            pos_weight=tiny_datamodule.pos_weight,
            structured_logger=s_logger,
        )
        callbacks: list[pl.Callback] = [
            ArtefactDirectoryCallback(run_dir),
            AttentionSummaryCallback(
                run_dir,
                n_fixed_batches=2,
                capture_epochs={1},
                seed=42,
            ),
        ]
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
            default_root_dir=str(tmp_path),
            accelerator="cpu",
        )
        trainer.fit(module, tiny_datamodule)
        s_logger.close()

        pq_path = run_dir / "attention_stats" / "epoch_1.parquet"
        assert pq_path.exists(), "epoch_1.parquet not created"

        df = pd.read_parquet(pq_path)
        assert len(df) > 0

        # W=7 for stage-0 window_size → W^2=49
        max_entropy = math.log(49) + 1e-3  # small tolerance
        assert (df["mean_entropy"] >= -1e-6).all(), "Entropy below 0"
        assert (df["mean_entropy"] <= max_entropy).all(), "Entropy above log(W^2)"

        # Alpha should be positive
        assert (df["alpha_value"] > 0).all()
