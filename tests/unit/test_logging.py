"""Unit tests for StructuredLogger."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd


if TYPE_CHECKING:
    from pathlib import Path

from lpqknorm.training.logging import StructuredLogger


class TestStructuredLogger:
    """Tests for JSONL and Parquet logging."""

    def test_log_step_creates_jsonl(self, tmp_path: Path) -> None:
        """After log_step, JSONL file exists and parses correctly."""
        sl = StructuredLogger(tmp_path)
        sl.log_step({"step": 0, "epoch": 0, "loss_total": 1.23})
        sl.log_step({"step": 1, "epoch": 0, "loss_total": 1.10})

        jsonl_path = tmp_path / "metrics" / "train_steps.jsonl"
        assert jsonl_path.exists()

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 2

        row0 = json.loads(lines[0])
        assert row0["step"] == 0
        assert abs(row0["loss_total"] - 1.23) < 1e-6

        sl.close()

    def test_flush_parquet_train(self, tmp_path: Path) -> None:
        """After log_epoch + flush_parquet, train_epochs.parquet is readable."""
        sl = StructuredLogger(tmp_path)
        sl.log_epoch("train", {"epoch": 0, "loss": 1.5})
        sl.log_epoch("train", {"epoch": 1, "loss": 1.2})
        sl.flush_parquet("train")

        pq_path = tmp_path / "metrics" / "train_epochs.parquet"
        assert pq_path.exists()

        df = pd.read_parquet(pq_path)
        assert len(df) == 2
        assert list(df.columns) == ["epoch", "loss"]

        sl.close()

    def test_flush_parquet_val(self, tmp_path: Path) -> None:
        """Val epoch rows flushed correctly."""
        sl = StructuredLogger(tmp_path)
        sl.log_epoch("val", {"epoch": 0, "val_dice": 0.5})
        sl.flush_parquet("val")

        df = pd.read_parquet(tmp_path / "metrics" / "val_epochs.parquet")
        assert len(df) == 1
        assert abs(df["val_dice"].iloc[0] - 0.5) < 1e-6

        sl.close()

    def test_per_patient_accumulation(self, tmp_path: Path) -> None:
        """Multiple log_per_patient calls accumulate correctly."""
        sl = StructuredLogger(tmp_path)
        sl.log_per_patient(
            "val",
            epoch=0,
            rows=[
                {"subject_id": "sub-0001", "dice": 0.8},
                {"subject_id": "sub-0002", "dice": 0.7},
            ],
        )
        sl.log_per_patient(
            "val",
            epoch=1,
            rows=[
                {"subject_id": "sub-0001", "dice": 0.85},
                {"subject_id": "sub-0002", "dice": 0.75},
            ],
        )
        sl.flush_parquet("val_per_patient")

        df = pd.read_parquet(tmp_path / "metrics" / "val_per_patient.parquet")
        assert len(df) == 4
        assert set(df["epoch"].unique()) == {0, 1}
        assert set(df["subject_id"].unique()) == {"sub-0001", "sub-0002"}

        sl.close()

    def test_close_flushes_all(self, tmp_path: Path) -> None:
        """close() flushes all pending parquet files."""
        sl = StructuredLogger(tmp_path)
        sl.log_epoch("train", {"epoch": 0, "loss": 1.0})
        sl.log_epoch("val", {"epoch": 0, "val_dice": 0.5})
        sl.close()

        assert (tmp_path / "metrics" / "train_epochs.parquet").exists()
        assert (tmp_path / "metrics" / "val_epochs.parquet").exists()

    def test_invalid_stage_raises(self, tmp_path: Path) -> None:
        """log_epoch with invalid stage raises ValueError."""
        import pytest

        sl = StructuredLogger(tmp_path)
        with pytest.raises(ValueError, match="Unknown stage"):
            sl.log_epoch("invalid", {"epoch": 0})
        sl.close()

    def test_per_lesion_accumulation(self, tmp_path: Path) -> None:
        """Per-lesion rows accumulate and flush correctly."""
        sl = StructuredLogger(tmp_path)
        sl.log_per_lesion(
            [
                {"subject_id": "sub-0001", "lesion_id": 0, "was_detected": True},
                {"subject_id": "sub-0001", "lesion_id": 1, "was_detected": False},
            ]
        )
        sl.flush_parquet("test_per_lesion")

        df = pd.read_parquet(tmp_path / "metrics" / "test_per_lesion.parquet")
        assert len(df) == 2

        sl.close()
