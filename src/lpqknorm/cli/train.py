"""Hydra CLI entry point for Phase 3 training sweep.

Each invocation runs a single training job defined by the composed Hydra
configuration.  The 18-run sweep is orchestrated by the SLURM array job
in ``scripts/submit_sweep.sbatch``.

Usage (local, mock data)::

    lpqknorm-train data.use_mock=true model.p=3.0 data.fold=0

Usage (Picasso, real data)::

    lpqknorm-train model.p=3.0 data.fold=0 training.seed=20260216
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import socket
import uuid
from pathlib import Path

import hydra
import monai
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from lpqknorm import __version__
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
from lpqknorm.utils.git import capture_git_state
from lpqknorm.utils.seeding import set_global_seed


logger = logging.getLogger(__name__)


def _pkg_version(name: str) -> str:
    """Return the installed version of a package, or ``"N/A"``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "N/A"


@hydra.main(config_path="../../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run a single training job from the composed Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Composed Hydra config.  See ``configs/train.yaml``.
    """
    # --- Seeding ---
    seed = int(cfg.training.seed)
    set_global_seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # --- Git state ---
    repo_root = Path(__file__).resolve().parents[3]
    git_state = capture_git_state(str(repo_root))

    # --- Run directory ---
    p_label = "vanilla" if cfg.model.p is None else str(cfg.model.p)
    run_dir = (
        Path(cfg.run_dir) / f"p={p_label}" / f"fold={cfg.data.fold}" / f"seed={seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Config snapshot ---
    config_str = OmegaConf.to_yaml(cfg)
    (run_dir / "config.yaml").write_text(config_str)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    # --- Git diff ---
    (run_dir / "code.diff").write_text(git_state.diff or "")

    # --- Environment JSON ---
    env_info = {
        pkg: _pkg_version(pkg)
        for pkg in ["torch", "monai", "pytorch-lightning", "lpqknorm"]
    }
    (run_dir / "env.json").write_text(json.dumps(env_info, indent=2))

    # --- DataModule ---
    dm: pl.LightningDataModule
    if cfg.data.use_mock:
        from lpqknorm.data.datamodule import MockAtlasDataModule, MockDataConfig

        dm = MockAtlasDataModule(
            MockDataConfig(
                batch_size=cfg.training.batch_size,
                seed=seed,
            )
        )
    else:
        from lpqknorm.data.datamodule import SegmentationDataModule

        dm = SegmentationDataModule(
            h5_path=Path(cfg.data.h5_path),
            fold=cfg.data.fold,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
        )
    dm.setup("fit")

    # --- Derive channel counts from the dataset header (invariant #9) ---
    # Downstream code must read n_modalities / n_label_classes from the HDF5
    # header rather than a hardcoded config default (which only matches the
    # ATLAS binary case).  For mock data we keep the config values.
    in_channels = int(cfg.model.in_channels)
    out_channels = int(cfg.model.out_channels)
    if not cfg.data.use_mock:
        header = dm.header  # type: ignore[attr-defined]
        if in_channels != header.n_modalities:
            logger.info(
                "Overriding model.in_channels: cfg=%d -> header.n_modalities=%d",
                in_channels,
                header.n_modalities,
            )
            in_channels = int(header.n_modalities)
        if out_channels != header.n_label_classes:
            logger.info(
                "Overriding model.out_channels: cfg=%d -> header.n_label_classes=%d",
                out_channels,
                header.n_label_classes,
            )
            out_channels = int(header.n_label_classes)

    # --- Configs ---
    model_cfg = ModelConfig(
        img_size=tuple(cfg.model.img_size),
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=cfg.model.feature_size,
        init_scheme=cfg.model.get("init_scheme", "scratch_trunc_normal"),
        linear_init_std=float(cfg.model.get("linear_init_std", 0.02)),
        alpha_init_scheme=cfg.model.get("alpha_init_scheme", "log_dk"),
        alpha_init_fixed=(
            float(cfg.model.alpha_init_fixed)
            if cfg.model.get("alpha_init_fixed", None) is not None
            else None
        ),
    )
    lp_cfg: LpQKNormConfig | None = (
        LpQKNormConfig(p=float(cfg.model.p)) if cfg.model.p is not None else None
    )
    training_cfg = TrainingConfig(
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
        batch_size=cfg.training.batch_size,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        bce_weight=cfg.training.bce_weight,
        dice_weight=cfg.training.dice_weight,
        threshold=cfg.training.threshold,
        gradient_log_every_n_steps=cfg.training.gradient_log_every_n_steps,
    )

    # --- Structured logger ---
    s_logger = StructuredLogger(run_dir)

    # --- Manifest init payload ---
    gpu_model = "N/A"
    cuda_version = "N/A"
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda or "N/A"

    # Init-spec hash: isolates the weight-initialization fields so the
    # Manifest can assert byte-equality across runs grouped by experiment.
    # The primary-sweep controlled experiment requires every p / fold run
    # to share identical init_scheme / linear_init_std / alpha_init_scheme /
    # alpha_init_fixed; drift in any of these would confound the p effect.
    init_spec = {
        "init_scheme": model_cfg.init_scheme,
        "linear_init_std": model_cfg.linear_init_std,
        "alpha_init_scheme": model_cfg.alpha_init_scheme,
        "alpha_init_fixed": model_cfg.alpha_init_fixed,
    }
    init_spec_hash = hashlib.sha256(
        json.dumps(init_spec, sort_keys=True).encode()
    ).hexdigest()

    manifest_init: dict[str, object] = {
        "run_id": str(uuid.uuid4()),
        "experiment": cfg.get("experiment", "p_sweep_v1"),
        "p": float(cfg.model.p) if cfg.model.p is not None else -1.0,
        "fold": int(cfg.data.fold),
        "seed": seed,
        "git_sha": git_state.sha,
        "git_dirty": git_state.dirty,
        "git_branch": git_state.branch,
        "host": socket.gethostname(),
        "gpu_model": gpu_model,
        "cuda_version": cuda_version,
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "lpqknorm_version": __version__,
        "config_hash": config_hash,
        "split_hash": getattr(dm, "split_hash", ""),
        "n_train": getattr(dm, "n_train", 0),
        "n_val": getattr(dm, "n_val", 0),
        "n_test": getattr(dm, "n_test", 0),
        "init_scheme": model_cfg.init_scheme,
        "linear_init_std": model_cfg.linear_init_std,
        "alpha_init_scheme": model_cfg.alpha_init_scheme,
        "alpha_init_fixed": model_cfg.alpha_init_fixed,
        "init_spec_hash": init_spec_hash,
    }

    # --- LightningModule ---
    module = LpSegmentationModule(
        model_cfg=model_cfg,
        lp_cfg=lp_cfg,
        training_cfg=training_cfg,
        pos_weight=getattr(dm, "pos_weight", None),
        structured_logger=s_logger,
    )

    # --- Callbacks ---
    ckpt_dir = str(run_dir / "checkpoints")
    callbacks: list[pl.Callback] = [
        ArtefactDirectoryCallback(run_dir),
        ManifestCallback(run_dir, manifest_init),
        PerPatientMetricsCallback(s_logger),
        GradientNormCallback(
            run_dir,
            log_every_n_steps=cfg.training.gradient_log_every_n_steps,
        ),
        AttentionSummaryCallback(run_dir, seed=seed),
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="last",
            monitor=None,
            save_last=True,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best_val_dice",
            monitor="val_dice_mean",
            mode="max",
            save_top_k=1,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best_small_recall",
            monitor="val_lesion_recall_small",
            mode="max",
            save_top_k=1,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_dice_mean",
            mode="max",
            patience=training_cfg.patience,
            verbose=True,
        ),
    ]

    # --- Trainer ---
    precision = training_cfg.precision if torch.cuda.is_available() else "32"
    # Optional smoke-test limits: keep the primary sweep runs untouched but
    # allow local CLI-driven truncation via `+training.limit_train_batches=N`.
    limit_train_batches = cfg.training.get("limit_train_batches", 1.0)
    limit_val_batches = cfg.training.get("limit_val_batches", 1.0)
    trainer = pl.Trainer(
        max_epochs=training_cfg.max_epochs,
        precision=precision,  # type: ignore[arg-type]
        gradient_clip_val=training_cfg.gradient_clip_val,
        callbacks=callbacks,
        logger=False,
        deterministic=True,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        default_root_dir=str(run_dir),
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )

    trainer.fit(module, dm)
    trainer.test(module, dm)
    s_logger.close()
    logger.info("Training complete. Artefacts at %s", run_dir)


if __name__ == "__main__":
    main()
