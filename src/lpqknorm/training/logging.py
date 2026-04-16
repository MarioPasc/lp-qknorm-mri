"""Structured JSONL + Parquet logger for per-step and per-epoch metrics.

Design rationale: Parquet tables are the native format for pandas/polars
downstream analysis.  JSONL is append-only and crash-safe for per-step logs.
TensorBoard is lossy (summarised).  W&B is fine as a secondary viewer but
cannot be the source of truth because Picasso compute nodes may lack
outbound network access.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

import pandas as pd


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


class StructuredLogger:
    """Write per-step metrics to JSONL and accumulate epoch/patient rows for Parquet.

    Parameters
    ----------
    run_dir : Path
        Root of the per-run artefact directory.

    Notes
    -----
    JSONL files are opened in append mode and ``fsync``-ed after every
    write to ensure crash-safety.  Parquet files are written atomically
    by accumulating rows in memory and calling :meth:`flush_parquet`.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._metrics_dir = run_dir / "metrics"
        self._metrics_dir.mkdir(parents=True, exist_ok=True)

        self._step_path = self._metrics_dir / "train_steps.jsonl"
        self._step_file = self._step_path.open("a", encoding="utf-8")

        # Accumulated rows for parquet writes
        self._train_epoch_rows: list[dict[str, Any]] = []
        self._val_epoch_rows: list[dict[str, Any]] = []
        self._val_per_patient_rows: list[dict[str, Any]] = []
        self._test_per_patient_rows: list[dict[str, Any]] = []
        self._test_per_lesion_rows: list[dict[str, Any]] = []

    def log_step(self, payload: dict[str, Any]) -> None:
        """Append one line to ``train_steps.jsonl`` and ``fsync``.

        Parameters
        ----------
        payload : dict
            Must include ``"step"`` and ``"epoch"`` keys.
        """
        self._step_file.write(json.dumps(payload) + "\n")
        self._step_file.flush()
        os.fsync(self._step_file.fileno())

    def log_epoch(self, stage: str, payload: dict[str, Any]) -> None:
        """Accumulate one row for the per-epoch Parquet file.

        Parameters
        ----------
        stage : str
            ``"train"`` or ``"val"``.
        payload : dict
            Metrics dict for this epoch.

        Raises
        ------
        ValueError
            If ``stage`` is not ``"train"`` or ``"val"``.
        """
        if stage == "train":
            self._train_epoch_rows.append(payload)
        elif stage == "val":
            self._val_epoch_rows.append(payload)
        else:
            msg = f"Unknown stage: {stage!r}"
            raise ValueError(msg)

    def log_per_patient(
        self, stage: str, epoch: int, rows: list[dict[str, Any]]
    ) -> None:
        """Accumulate per-patient rows (prepends ``epoch`` field to each row).

        Parameters
        ----------
        stage : str
            ``"val"`` or ``"test"``.
        epoch : int
            Current epoch number.
        rows : list[dict]
            One dict per patient / slice.  Must include ``"subject_id"``.
        """
        for row in rows:
            enriched = {"epoch": epoch, **row}
            if stage == "val":
                self._val_per_patient_rows.append(enriched)
            elif stage == "test":
                self._test_per_patient_rows.append(enriched)

    def log_per_lesion(self, rows: list[dict[str, Any]]) -> None:
        """Accumulate per-lesion rows for ``test_per_lesion.parquet``.

        Parameters
        ----------
        rows : list[dict]
            One dict per connected lesion component.
        """
        self._test_per_lesion_rows.extend(rows)

    def flush_parquet(self, stage: str) -> None:
        """Write accumulated rows to the appropriate Parquet file.

        Parameters
        ----------
        stage : str
            One of ``"train"``, ``"val"``, ``"val_per_patient"``,
            ``"test_per_patient"``, ``"test_per_lesion"``.
        """
        mapping: dict[str, tuple[list[dict[str, Any]], Path]] = {
            "train": (
                self._train_epoch_rows,
                self._metrics_dir / "train_epochs.parquet",
            ),
            "val": (
                self._val_epoch_rows,
                self._metrics_dir / "val_epochs.parquet",
            ),
            "val_per_patient": (
                self._val_per_patient_rows,
                self._metrics_dir / "val_per_patient.parquet",
            ),
            "test_per_patient": (
                self._test_per_patient_rows,
                self._metrics_dir / "test_per_patient.parquet",
            ),
            "test_per_lesion": (
                self._test_per_lesion_rows,
                self._metrics_dir / "test_per_lesion.parquet",
            ),
        }
        rows, path = mapping[stage]
        if not rows:
            logger.warning("No rows to flush for stage=%r", stage)
            return
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        logger.info("Flushed %d rows to %s", len(rows), path)

    def close(self) -> None:
        """Flush all pending Parquet files and close the JSONL handle."""
        for stage in [
            "train",
            "val",
            "val_per_patient",
            "test_per_patient",
            "test_per_lesion",
        ]:
            self.flush_parquet(stage)
        self._step_file.close()
        logger.info("StructuredLogger closed for %s", self._run_dir)
