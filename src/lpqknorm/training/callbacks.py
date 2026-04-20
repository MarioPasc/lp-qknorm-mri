"""PyTorch Lightning callbacks for exhaustive artefact logging.

Six callbacks implement the "log everything, aggregate later" principle:

- :class:`ArtefactDirectoryCallback` — ensures the run directory structure.
- :class:`ManifestCallback` — writes ``manifest.json`` at start and end.
- :class:`PerPatientMetricsCallback` — flushes per-patient validation rows.
- :class:`GradientNormCallback` — logs QKV/alpha grad norms every K steps.
- :class:`AttentionSummaryCallback` — stage-0 attention stats on fixed batches.
- :class:`ProbeCallback` — runs mechanistic probes at scheduled epochs.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytorch_lightning as pl
import torch

from lpqknorm.models.hooks import AttentionHookRegistry


if TYPE_CHECKING:
    from pathlib import Path

    from lpqknorm.training.logging import StructuredLogger


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunManifest:
    """Full run metadata written to ``manifest.json``.

    Parameters
    ----------
    run_id : str
        UUID identifying this run.
    experiment : str
        Experiment label, e.g. ``"p_sweep_v1"``.
    p : float
        Lp norm exponent (``-1.0`` for vanilla baseline).
    fold : int
        Cross-validation fold index.
    seed : int
        Global random seed.
    git_sha : str
        Commit SHA at training time.
    git_dirty : bool
        Whether the working tree had uncommitted changes.
    git_branch : str
        Branch name at training time.
    started_utc : str
        ISO 8601 timestamp of training start.
    finished_utc : str or None
        ISO 8601 timestamp of training end (``None`` if still running).
    host : str
        Hostname of the machine.
    gpu_model : str
        GPU model name or ``"N/A"``.
    cuda_version : str
        CUDA version string or ``"N/A"``.
    torch_version : str
        PyTorch version.
    monai_version : str
        MONAI version.
    lpqknorm_version : str
        Package version.
    config_hash : str
        SHA256 of the composed Hydra config YAML.
    split_hash : str
        SHA256 of the data splits descriptor.
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    n_test : int
        Number of test samples.
    walltime_sec : float or None
        Wall-clock training time in seconds.
    peak_gpu_memory_mb : float or None
        Peak GPU memory allocated in MB.
    final_epoch : int or None
        Last completed epoch index.
    best_val_dice : float or None
        Best validation Dice achieved.
    best_small_recall : float or None
        Best small-stratum lesion recall achieved.
    """

    run_id: str
    experiment: str
    p: float
    fold: int
    seed: int
    git_sha: str
    git_dirty: bool
    git_branch: str
    started_utc: str
    finished_utc: str | None
    host: str
    gpu_model: str
    cuda_version: str
    torch_version: str
    monai_version: str
    lpqknorm_version: str
    config_hash: str
    split_hash: str
    n_train: int
    n_val: int
    n_test: int
    walltime_sec: float | None
    peak_gpu_memory_mb: float | None
    final_epoch: int | None
    best_val_dice: float | None
    best_small_recall: float | None


# ---------------------------------------------------------------------------
# ArtefactDirectoryCallback
# ---------------------------------------------------------------------------


class ArtefactDirectoryCallback(pl.Callback):
    """Ensure the run artefact directory structure exists at training start.

    Creates subdirectories: ``metrics/``, ``attention_stats/``,
    ``gradient_stats/``, ``predictions/``, ``checkpoints/``, ``probes/``.

    Parameters
    ----------
    run_dir : Path
        Root of the per-run artefact directory.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create subdirectories and sentinel files."""
        for subdir in [
            "metrics",
            "attention_stats",
            "gradient_stats",
            "predictions",
            "checkpoints",
            "probes",
        ]:
            d = self._run_dir / subdir
            d.mkdir(parents=True, exist_ok=True)
            (d / ".keep").touch()
        logger.info("ArtefactDirectoryCallback: run_dir=%s ensured.", self._run_dir)


# ---------------------------------------------------------------------------
# ManifestCallback
# ---------------------------------------------------------------------------


class ManifestCallback(pl.Callback):
    """Write and update ``manifest.json`` at start and end of training.

    Parameters
    ----------
    run_dir : Path
        Per-run artefact directory.
    manifest_init : dict
        Initial manifest fields (run_id, experiment, p, fold, seed,
        git_*, config_hash, split_hash, n_train, n_val, n_test, host,
        gpu_model, cuda_version, torch_version, monai_version,
        lpqknorm_version).
    """

    def __init__(self, run_dir: Path, manifest_init: dict[str, Any]) -> None:
        self._run_dir = run_dir
        self._manifest_init = manifest_init
        self._start_time: float = 0.0
        self._manifest_path = run_dir / "manifest.json"

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Write initial manifest with started_utc."""
        self._start_time = time.monotonic()
        init = {
            **self._manifest_init,
            "started_utc": datetime.datetime.now(tz=datetime.UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "finished_utc": None,
            "walltime_sec": None,
            "peak_gpu_memory_mb": None,
            "final_epoch": None,
            "best_val_dice": None,
            "best_small_recall": None,
        }
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(init, indent=2))
        logger.info(
            "ManifestCallback: wrote initial manifest to %s",
            self._manifest_path,
        )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update manifest with final metrics and timing."""
        elapsed = time.monotonic() - self._start_time
        data = json.loads(self._manifest_path.read_text())
        data["finished_utc"] = (
            datetime.datetime.now(tz=datetime.UTC).isoformat().replace("+00:00", "Z")
        )
        data["walltime_sec"] = round(elapsed, 2)
        data["final_epoch"] = trainer.current_epoch
        if torch.cuda.is_available():
            data["peak_gpu_memory_mb"] = round(
                torch.cuda.max_memory_allocated() / 1024**2, 2
            )
        cb_metrics = trainer.callback_metrics
        data["best_val_dice"] = float(cb_metrics.get("val_dice_mean", math.nan))
        data["best_small_recall"] = float(
            cb_metrics.get("val_lesion_recall_small", math.nan)
        )
        self._manifest_path.write_text(json.dumps(data, indent=2))
        logger.info("ManifestCallback: updated final manifest.")


# ---------------------------------------------------------------------------
# PerPatientMetricsCallback
# ---------------------------------------------------------------------------


class PerPatientMetricsCallback(pl.Callback):
    """Flush per-patient validation metrics to Parquet at each epoch end.

    The :class:`~lpqknorm.training.module.LpSegmentationModule` stores
    per-batch patient rows in ``self._per_patient_buffer``.  This callback
    collects them at epoch end and flushes to :class:`StructuredLogger`.

    Parameters
    ----------
    structured_logger : StructuredLogger
        Logger instance writing to the run's ``metrics/`` directory.
    """

    def __init__(self, structured_logger: StructuredLogger) -> None:
        self._logger = structured_logger

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Collect per-patient rows from module buffer and flush."""
        buffer: list[dict[str, Any]] = getattr(pl_module, "_per_patient_buffer", [])
        if not buffer:
            return
        # Snapshot and clear before flushing
        rows = list(buffer)
        buffer.clear()
        epoch = trainer.current_epoch
        self._logger.log_per_patient("val", epoch, rows)
        self._logger.flush_parquet("val_per_patient")
        logger.debug(
            "PerPatientMetricsCallback: flushed %d rows at epoch %d",
            len(rows),
            epoch,
        )


# ---------------------------------------------------------------------------
# GradientNormCallback
# ---------------------------------------------------------------------------


class GradientNormCallback(pl.Callback):
    """Log per-layer gradient L2 norms for QKV and alpha_raw parameters.

    Parameters
    ----------
    run_dir : Path
        Per-run artefact directory.
    log_every_n_steps : int
        How often to capture gradient norms.  Default ``50``.

    Notes
    -----
    Writes ``gradient_stats/layer_norms.parquet`` at training end.
    Only captures gradients for parameters whose name contains ``"qkv"``
    or ``"alpha_raw"``.
    """

    def __init__(self, run_dir: Path, log_every_n_steps: int = 50) -> None:
        self._run_dir = run_dir
        self._every_n = log_every_n_steps
        self._rows: list[dict[str, float | int]] = []

    def on_after_backward(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Capture gradient norms every N steps."""
        step = trainer.global_step
        if step % self._every_n != 0:
            return
        row: dict[str, float | int] = {
            "step": step,
            "epoch": trainer.current_epoch,
        }
        for name, param in pl_module.named_parameters():
            if ("qkv" in name or "alpha_raw" in name) and param.grad is not None:
                grad_norm = float(param.grad.detach().norm(2).item())
                col = name.replace(".", "_")
                row[f"grad_norm_{col}"] = grad_norm
        self._rows.append(row)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Flush accumulated gradient norms to Parquet."""
        if not self._rows:
            return
        out_dir = self._run_dir / "gradient_stats"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._rows)
        df.to_parquet(out_dir / "layer_norms.parquet", index=False)
        logger.info(
            "GradientNormCallback: wrote %d rows to layer_norms.parquet",
            len(self._rows),
        )


# ---------------------------------------------------------------------------
# AttentionSummaryCallback
# ---------------------------------------------------------------------------


class AttentionSummaryCallback(pl.Callback):
    """Capture attention summary stats on a fixed validation batch subset.

    Runs at epochs ``{1, 5, 10, final}`` using a seed-fixed batch subset
    of the validation DataLoader.  Writes
    ``attention_stats/epoch_{N}.parquet``.

    Parameters
    ----------
    run_dir : Path
        Per-run artefact directory.
    n_fixed_batches : int
        Number of batches from the val loader to use.  Default ``10``.
    capture_epochs : set[int] or None
        Epochs at which to capture (1-indexed).  Default ``{1, 5, 10}``.
        The ``"final"`` epoch is always captured in ``on_fit_end``.
    seed : int
        Seed for fixed-batch selection.

    Notes
    -----
    ``mean_lesion_mass_on_lesion_queries`` and peakiness fields are left
    as ``NaN`` — full probe data is collected in Phase 4.  This callback
    provides a coarse trajectory (entropy, max-prob, alpha) for free.
    """

    def __init__(
        self,
        run_dir: Path,
        n_fixed_batches: int = 10,
        capture_epochs: set[int] | None = None,
        seed: int = 42,
    ) -> None:
        self._run_dir = run_dir
        self._n_fixed_batches = n_fixed_batches
        self._capture_epochs = (
            capture_epochs if capture_epochs is not None else {1, 5, 10}
        )
        self._seed = seed
        # Registry is constructed on-demand inside _capture_and_write so that
        # hooks fire ONLY during the capture events, not on every training
        # forward.  Leaving the registry registered across training caused
        # an unbounded leak: each training forward appended raw captures
        # (references to activation tensors) into _raw_captures, pinning
        # activation memory and the autograd sub-graph until fit end.
        self._fixed_batches: list[dict[str, Any]] | None = None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Capture attention stats at configured epochs."""
        epoch = trainer.current_epoch + 1  # 1-indexed
        if epoch not in self._capture_epochs:
            return
        self._capture_and_write(trainer, pl_module, epoch_label=str(epoch))

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Capture the final epoch (no persistent hooks to clean up)."""
        self._capture_and_write(trainer, pl_module, epoch_label="final")

    def _capture_and_write(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        epoch_label: str,
    ) -> None:
        """Run model on fixed batches and collect attention summary stats.

        Hooks are registered for the scope of this method only and removed
        in a ``finally`` block — crucial for memory: a leaked registry
        accumulates one raw-capture entry per attention block per training
        forward and pins the autograd sub-graph until fit end.
        """
        val_loader = trainer.val_dataloaders
        if val_loader is None:
            return

        # Lazily build fixed batch list (seed-fixed, same across epochs)
        if self._fixed_batches is None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(self._seed)
            self._fixed_batches = []
            for i, batch in enumerate(val_loader):
                if i >= self._n_fixed_batches:
                    break
                self._fixed_batches.append(batch)
            torch.set_rng_state(rng_state)

        model_fn: torch.nn.Module = pl_module.model  # type: ignore[assignment]

        registry = AttentionHookRegistry()
        try:
            registry.register(model_fn, stages=[0])
        except Exception:
            # Vanilla baseline has no LpWindowAttention — quietly skip.
            logger.warning(
                "AttentionSummaryCallback: could not register hooks "
                "(expected for vanilla baseline); skipping epoch=%s.",
                epoch_label,
            )
            return

        rows: list[dict[str, Any]] = []
        pl_module.eval()
        try:
            with torch.no_grad():
                for batch in self._fixed_batches:
                    images = batch["image"].to(pl_module.device)
                    _ = model_fn(images)
                    captures = registry.captures()
                    for cap in captures:
                        if cap.attention is None:
                            continue
                        attn = cap.attention  # (B*nW, nh, n, n)

                        # Entropy: H = -sum A_ij log A_ij
                        attn_safe = attn.clamp(min=1e-9)
                        entropy = -(attn_safe * attn_safe.log()).sum(-1)

                        # Max prob per row
                        mean_max_prob = float(attn.max(dim=-1).values.mean().item())

                        # Alpha
                        alpha_val = (
                            float(cap.alpha.item())
                            if cap.alpha is not None
                            else math.nan
                        )

                        rows.append(
                            {
                                "block_id": (
                                    f"stage{cap.stage_index}_block{cap.block_index}"
                                ),
                                "stage_index": cap.stage_index,
                                "block_index": cap.block_index,
                                "mean_entropy": float(entropy.mean().item()),
                                "median_entropy": float(entropy.median().item()),
                                "mean_max_prob": mean_max_prob,
                                "alpha_value": alpha_val,
                                # Phase 4 probes — NaN placeholders
                                "mean_lesion_mass_on_lesion_queries": math.nan,
                                "mean_q_peakiness": math.nan,
                                "mean_k_peakiness": math.nan,
                            }
                        )
                    registry.clear()
        finally:
            registry.remove()

        if not rows:
            logger.warning(
                "AttentionSummaryCallback: no rows at epoch=%s",
                epoch_label,
            )
            return

        out_dir = self._run_dir / "attention_stats"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / f"epoch_{epoch_label}.parquet", index=False)
        logger.info(
            "AttentionSummaryCallback: wrote epoch_%s.parquet (%d rows)",
            epoch_label,
            len(rows),
        )


# ---------------------------------------------------------------------------
# ProbeCallback
# ---------------------------------------------------------------------------


class ProbeCallback(pl.Callback):
    """Invoke :class:`~lpqknorm.probes.recorder.ProbeRecorder` at scheduled epochs.

    Parameters
    ----------
    recorder : ProbeRecorder
        Configured recorder with all probe instances.
    probe_epochs : frozenset[int]
        0-indexed epochs at which to run probes.
        Default ``{0, 1, 5, 10, 25, 50}``.

    Notes
    -----
    The ``"final"`` epoch is always captured in ``on_fit_end``.
    The probe loader is passed at recorder construction time.
    """

    _DEFAULT_SCHEDULE: frozenset[int] = frozenset({0, 1, 5, 10, 25, 50})

    def __init__(
        self,
        recorder: object,
        probe_loader: Any,
        probe_epochs: frozenset[int] | None = None,
    ) -> None:
        from lpqknorm.probes.recorder import ProbeRecorder

        assert isinstance(recorder, ProbeRecorder)
        self._recorder: ProbeRecorder = recorder
        self._probe_loader: Any = probe_loader
        self._probe_epochs = (
            probe_epochs if probe_epochs is not None else self._DEFAULT_SCHEDULE
        )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Run probes at scheduled epochs."""
        epoch = trainer.current_epoch
        if epoch in self._probe_epochs:
            self._run(pl_module, epoch_tag=str(epoch))

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Run probes at the final epoch."""
        self._run(pl_module, epoch_tag="final")

    def _run(self, pl_module: pl.LightningModule, epoch_tag: str) -> None:
        """Execute probe recording."""
        model: torch.nn.Module = pl_module.model  # type: ignore[assignment]
        model.eval()
        self._recorder.run(
            model=model,
            dataloader=self._probe_loader,
            epoch_tag=epoch_tag,
            device=pl_module.device,
        )


# ---------------------------------------------------------------------------
# AlphaLogger
# ---------------------------------------------------------------------------


class AlphaLogger(pl.Callback):
    """Append ``softplus(alpha_raw)`` to ``alpha_trajectory.jsonl`` per step.

    One line per (step, block) is written so the Phase-5 control analysis
    can correlate the learnable scale ``alpha`` with the empirical logit
    gap ``Delta(p)``.  Covers stage-0 blocks 0 and 1 by default.

    Parameters
    ----------
    run_dir : Path
        Per-run artefact directory.  Output lives under
        ``run_dir / "probes" / "alpha_trajectory.jsonl"``.
    stage : int
        Swin stage whose alpha values are logged.  Default ``0``.
    blocks : tuple[int, ...]
        Block indices within the stage.  Default ``(0, 1)``.
    p_value : float
        The Lp norm exponent for the current run (for provenance).
    fold : int
        Cross-validation fold index for the current run.

    Notes
    -----
    Writing is append-only and unbuffered (``flush=True``) so mid-run
    crashes do not lose the trajectory.  The file is deleted on
    ``on_fit_start`` so re-running a fold clears the old log.
    """

    def __init__(
        self,
        run_dir: Path,
        stage: int = 0,
        blocks: tuple[int, ...] = (0, 1),
        p_value: float = 2.0,
        fold: int = 0,
    ) -> None:
        self._run_dir = run_dir
        self._stage = stage
        self._blocks = blocks
        self._p = p_value
        self._fold = fold
        self._path = run_dir / "probes" / "alpha_trajectory.jsonl"

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Create the output file (truncating any previous content)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("")
        logger.info("AlphaLogger: initialised trajectory at %s", self._path)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Append one JSONL record per block after each optimizer step."""
        model: torch.nn.Module = pl_module.model  # type: ignore[assignment]
        step = trainer.global_step
        epoch = trainer.current_epoch
        try:
            layer = getattr(model.swinViT, f"layers{self._stage + 1}")[0]
        except AttributeError:
            return
        with self._path.open("a") as fh:
            for b in self._blocks:
                try:
                    attn = layer.blocks[b].attn
                    alpha_raw = attn.lp_qknorm.alpha_raw
                except AttributeError:
                    continue
                alpha = float(torch.nn.functional.softplus(alpha_raw).item())
                rec = {
                    "step": int(step),
                    "epoch": int(epoch),
                    "block": int(b),
                    "alpha": alpha,
                    "p": float(self._p),
                    "fold": int(self._fold),
                }
                fh.write(json.dumps(rec) + "\n")
            fh.flush()


# ---------------------------------------------------------------------------
# PatchingCallback
# ---------------------------------------------------------------------------


class PatchingCallback(pl.Callback):
    """Run :class:`~lpqknorm.probes.patching.ActivationPatcher` on fit end.

    Parameters
    ----------
    config : PatchingConfig
        Already-populated patching config (source/target checkpoints etc).
    probe_loader : Any
        Fixed probe DataLoader to reuse.
    output_dir : Path
        Directory for the ``patching_best_dice.h5`` file.  Created on
        ``on_fit_end``.
    model_loader : Callable[[Path], nn.Module] or None
        Optional factory for loading source/target models.  Passed through
        to :class:`ActivationPatcher`.

    Notes
    -----
    If the config references checkpoints that do not yet exist on disk the
    callback silently skips the patching run and emits a warning; this
    keeps the train loop resilient when the best-small-recall model has
    not been selected yet.
    """

    def __init__(
        self,
        config: Any,
        probe_loader: Any,
        output_dir: Path,
        model_loader: Any = None,
    ) -> None:
        self._config = config
        self._probe_loader = probe_loader
        self._output_dir = output_dir
        self._model_loader = model_loader

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Trigger the patching sweep at end-of-fit."""
        del trainer, pl_module  # unused
        from pathlib import Path as _Path

        from lpqknorm.probes.patching import ActivationPatcher

        src = _Path(self._config.source_checkpoint)
        tgt = _Path(self._config.target_checkpoint)
        if not src.exists() or not tgt.exists():
            logger.warning(
                "PatchingCallback: checkpoints missing (src=%s, tgt=%s); "
                "skipping patching.",
                src,
                tgt,
            )
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        patcher = ActivationPatcher(self._config, model_loader=self._model_loader)
        out = patcher.run(self._probe_loader, self._output_dir)
        logger.info("PatchingCallback: wrote %s", out)
