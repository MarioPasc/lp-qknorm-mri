"""PyTorch Lightning module for Lp-QKNorm segmentation training.

Encapsulates the 2D SwinUNETR model, compound loss, optimizer/scheduler
configuration, and per-step / per-epoch metric computation.  Per-patient
rows are accumulated in ``_per_patient_buffer`` for the
:class:`~lpqknorm.training.callbacks.PerPatientMetricsCallback` to flush.

References
----------
- Hatamizadeh et al. *Swin UNETR*. BrainLes 2021. arXiv:2201.01266.
- Loshchilov & Hutter. *Decoupled Weight Decay Regularization*.
  ICLR 2019. arXiv:1711.05101.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lpqknorm.models.attention import LpWindowAttention, set_capture_enabled
from lpqknorm.models.swin_unetr_lp import build_swin_unetr_lp
from lpqknorm.training.losses import CompoundSegLoss
from lpqknorm.training.metrics import (
    dice_score,
    hd95,
    iou_score,
    lesion_wise_detection,
)


if TYPE_CHECKING:
    from lpqknorm.models.init import AlphaInitScheme, InitScheme
    from lpqknorm.models.lp_qknorm import LpQKNormConfig
    from lpqknorm.training.logging import StructuredLogger


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Frozen configuration for the 2D SwinUNETR model.

    Parameters
    ----------
    img_size : tuple[int, int]
        Input spatial size.  Default ``(224, 224)``.
    in_channels : int
        Number of input channels.  Default ``1`` (single T1w MRI).
    out_channels : int
        Number of output channels.  Default ``1`` (binary segmentation).
    feature_size : int
        Base embedding dimension.  Must be divisible by 12 (MONAI
        constraint).  Default ``24``.
    init_scheme : {"scratch_trunc_normal", "pretrained_ssl"}
        Weight-initialization regime.  Default ``"scratch_trunc_normal"``
        (primary from-scratch run).  ``"pretrained_ssl"`` loads the Tang
        et al. (2022) SSL checkpoint and is reserved for the single
        ablation row.
    linear_init_std : float
        Standard deviation of the truncated-normal used for ``nn.Linear``,
        ``nn.Conv{2,3}d``, and ``relative_position_bias_table`` weights.
        Default ``0.02`` per Swin (Liu et al., 2021) and ViT
        (Dosovitskiy et al., 2021).
    alpha_init_scheme : {"log_dk", "sqrt_dk", "fixed"}
        Scheme for initializing ``LpQKNorm.alpha_raw``.  Default
        ``"log_dk"``: at ``p = 2`` this recovers the Henry et al. (2020)
        logit scale ``alpha <q_hat, k_hat> in [-log d_k, log d_k]``.
    alpha_init_fixed : float or None
        Required iff ``alpha_init_scheme == "fixed"``; ignored otherwise.
        Must be ``> 0``.
    """

    img_size: tuple[int, int] = (224, 224)
    in_channels: int = 1
    out_channels: int = 1
    feature_size: int = 24
    init_scheme: InitScheme = "scratch_trunc_normal"
    linear_init_std: float = 0.02
    alpha_init_scheme: AlphaInitScheme = "log_dk"
    alpha_init_fixed: float | None = None
    use_checkpoint: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen training hyperparameters.

    Parameters
    ----------
    lr : float
        AdamW learning rate.  Default ``3e-4``.
    weight_decay : float
        AdamW weight decay.  Default ``1e-5``.
    betas : tuple[float, float]
        AdamW beta coefficients.  Default ``(0.9, 0.999)``.
    max_epochs : int
        Maximum training epochs.  Default ``100``.
    patience : int
        EarlyStopping patience (epochs).  Default ``15``.
    eta_min : float
        CosineAnnealingLR minimum learning rate.  Default ``1e-6``.
    batch_size : int
        Batch size (for logging reference).  Default ``16``.
    precision : str
        Lightning precision string.  Default ``"bf16-mixed"``.
    gradient_clip_val : float
        Gradient clipping value.  Default ``1.0``.
    bce_weight : float
        BCE loss weight.  Default ``0.5``.
    dice_weight : float
        Dice loss weight.  Default ``0.5``.
    threshold : float
        Sigmoid threshold for binary prediction.  Default ``0.5``.
    gradient_log_every_n_steps : int
        Gradient norm logging frequency (steps).  Default ``50``.
    """

    lr: float = 3e-4
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    max_epochs: int = 100
    patience: int = 15
    eta_min: float = 1e-6
    batch_size: int = 16
    precision: str = "bf16-mixed"
    gradient_clip_val: float = 1.0
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    threshold: float = 0.5
    gradient_log_every_n_steps: int = 50
    skip_on_nonfinite_loss: bool = True


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------


class LpSegmentationModule(pl.LightningModule):
    """Lightning module for binary stroke lesion segmentation with Lp-QKNorm.

    Parameters
    ----------
    model_cfg : ModelConfig
        Model architecture configuration.
    lp_cfg : LpQKNormConfig or None
        Lp-QKNorm configuration.  ``None`` = vanilla baseline.
    training_cfg : TrainingConfig
        Training hyperparameters.
    pos_weight : Tensor or None
        BCE positive class weight from the DataModule.
    structured_logger : StructuredLogger or None
        For per-step JSONL logging.  ``None`` disables structured logging
        (useful for tests).

    Attributes
    ----------
    model : nn.Module
        The 2D SwinUNETR (patched or vanilla).
    loss_fn : CompoundSegLoss
        Compound BCE + Dice loss.
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        lp_cfg: LpQKNormConfig | None = None,
        training_cfg: TrainingConfig | None = None,
        pos_weight: Tensor | None = None,
        structured_logger: StructuredLogger | None = None,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg if model_cfg is not None else ModelConfig()
        self.lp_cfg = lp_cfg
        self.training_cfg = (
            training_cfg if training_cfg is not None else TrainingConfig()
        )
        self._structured_logger = structured_logger

        # Save hyperparameters (excludes non-serializable objects)
        self.save_hyperparameters(ignore=["structured_logger", "pos_weight"])

        # Build model
        self.model: nn.Module = build_swin_unetr_lp(
            img_size=self.model_cfg.img_size,
            in_channels=self.model_cfg.in_channels,
            out_channels=self.model_cfg.out_channels,
            feature_size=self.model_cfg.feature_size,
            lp_cfg=lp_cfg,
            init_scheme=self.model_cfg.init_scheme,
            linear_init_std=self.model_cfg.linear_init_std,
            alpha_init_scheme=self.model_cfg.alpha_init_scheme,
            alpha_init_fixed=self.model_cfg.alpha_init_fixed,
            use_checkpoint=self.model_cfg.use_checkpoint,
        )

        # Disable the per-module ``_capture`` dict on every LpWindowAttention
        # instance.  The capture dict pins references to ``q, k, q_hat,
        # k_hat, logits, attention, relative_position_bias`` — all of which
        # are intermediate activations whose autograd sub-graph would
        # otherwise survive past ``backward()``.  Callbacks and probes that
        # need the captures re-enable them locally via AttentionHookRegistry.
        n_attn = set_capture_enabled(self.model, False)
        logger.info(
            "LpSegmentationModule: disabled capture on %d LpWindowAttention "
            "module(s); probes/callbacks re-enable per event.",
            n_attn,
        )

        # Loss
        self.loss_fn = CompoundSegLoss(
            bce_weight=self.training_cfg.bce_weight,
            dice_weight=self.training_cfg.dice_weight,
            pos_weight=pos_weight,
        )

        # Per-patient buffer (read by PerPatientMetricsCallback)
        self._per_patient_buffer: list[dict[str, Any]] = []

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : Tensor
            Input image tensor, shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Raw logits, shape ``(B, out_channels, H, W)``.
        """
        return self.model(x)  # type: ignore[no-any-return]

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Single training step.

        Parameters
        ----------
        batch : dict
            Batch dict with ``"image"`` and ``"mask"`` keys.
        batch_idx : int
            Batch index within the epoch.

        Returns
        -------
        Tensor
            Scalar loss for backpropagation.
        """
        images: Tensor = batch["image"]
        masks: Tensor = batch["mask"]

        step_start = time.perf_counter()
        logits = self.model(images)
        loss_total: Tensor
        loss_total, loss_bce, loss_dice = self.loss_fn(logits, masks)

        # Non-finite-loss guard.  Under bf16-mixed precision with Lp norms
        # of non-integer p, a malformed micro-batch (e.g. a slice with an
        # extreme logit magnitude) can still produce NaN/Inf even after the
        # fp32 islands inside _lp_normalize and CompoundSegLoss.  If we
        # forwarded the NaN through backward, AdamW would poison every
        # parameter with NaN on the first optimiser step.  Instead we
        # substitute a zero leaf tensor: backward runs (PL requires it),
        # the optimiser sees all-zero gradients and performs a no-op step.
        if self.training_cfg.skip_on_nonfinite_loss and not bool(
            torch.isfinite(loss_total)
        ):
            logger.warning(
                "Non-finite training loss at step=%d (bce=%s, dice=%s); "
                "replacing with zero leaf and skipping parameter update.",
                self.global_step,
                float(loss_bce.detach().item())
                if torch.isfinite(loss_bce)
                else float("nan"),
                float(loss_dice.detach().item())
                if torch.isfinite(loss_dice)
                else float("nan"),
            )
            loss_total = torch.zeros((), device=loss_total.device, requires_grad=True)

        # Collect alpha stats per stage (vanilla has no lp_qknorm)
        alpha_stats = self._collect_alpha_stats()

        # Log to Lightning (for EarlyStopping, checkpointing)
        self.log(
            "train_loss",
            loss_total,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_loss_bce", loss_bce, on_step=True, on_epoch=False)
        self.log("train_loss_dice", loss_dice, on_step=True, on_epoch=False)

        # Log to StructuredLogger (JSONL)
        if self._structured_logger is not None:
            elapsed = time.perf_counter() - step_start
            throughput = images.size(0) / max(elapsed, 1e-9)
            gpu_mem = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0.0
            )
            opt = self.optimizers()
            lr = (
                opt.param_groups[0]["lr"]  # type: ignore[union-attr]
                if opt is not None
                else 0.0
            )
            payload: dict[str, Any] = {
                "step": self.global_step,
                "epoch": self.current_epoch,
                "loss_total": float(loss_total.detach().item()),
                "loss_bce": float(loss_bce.detach().item()),
                "loss_dice": float(loss_dice.detach().item()),
                "lr": float(lr),
                "throughput_samples_per_sec": float(throughput),
                "gpu_mem_mb": float(gpu_mem),
                **alpha_stats,
            }
            self._structured_logger.log_step(payload)

        return loss_total

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Validation step: compute loss + metrics, accumulate per-patient rows.

        Parameters
        ----------
        batch : dict
            Batch dict with ``"image"``, ``"mask"``, ``"subject_id"``,
            ``"volume_stratum"`` keys.
        batch_idx : int
            Batch index within the epoch.
        """
        images: Tensor = batch["image"]
        masks: Tensor = batch["mask"]
        subject_ids: list[str] = batch["subject_id"]
        strata: list[str] = batch["volume_stratum"]

        # Keep the entire validation step inside ``no_grad``.  Previously the
        # loss was computed outside the no_grad scope, which built a short
        # autograd graph that was dropped without backward — wasting memory
        # proportional to the model depth during every validation batch.
        with torch.no_grad():
            logits = self.model(images)
            loss_total, loss_bce, loss_dice = self.loss_fn(logits, masks)

        # Binary predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= self.training_cfg.threshold).float()
        target = masks

        # Batch-level metrics
        dice = dice_score(preds, target)  # (B,)
        iou = iou_score(preds, target)  # (B,)

        self.log("val_loss", loss_total, on_epoch=True, prog_bar=True)
        self.log("val_loss_bce", loss_bce, on_epoch=True)
        self.log("val_loss_dice", loss_dice, on_epoch=True)
        self.log("val_dice_mean", dice.mean(), on_epoch=True, prog_bar=True)
        self.log("val_iou_mean", iou.mean(), on_epoch=True)

        # Per-patient rows for PerPatientMetricsCallback
        small_recalls: list[float] = []
        for i in range(images.size(0)):
            det = lesion_wise_detection(preds[i], target[i])
            try:
                hd_val = float(hd95(preds[i : i + 1], target[i : i + 1]).item())
            except Exception:
                hd_val = math.nan

            self._per_patient_buffer.append(
                {
                    "subject_id": subject_ids[i],
                    "volume_stratum": strata[i],
                    "dice": float(dice[i].item()),
                    "iou": float(iou[i].item()),
                    "precision": self._compute_precision(preds[i], target[i]),
                    "recall": self._compute_recall(preds[i], target[i]),
                    "lesion_recall": det.lesion_recall,
                    "false_positives_per_slice": det.false_positives,
                    "hd95": hd_val,
                    "n_gt_lesions": det.n_gt_lesions,
                }
            )

            # Track small-stratum recall for checkpoint
            if strata[i] == "small" and not math.isnan(det.lesion_recall):
                small_recalls.append(det.lesion_recall)

        # Always log val_lesion_recall_small so ModelCheckpoint(monitor=...) can
        # see it every epoch (NaN when no small-stratum patient is in the batch;
        # PL aggregates over the epoch and handles NaN safely).
        self.log(
            "val_lesion_recall_small",
            float(np.mean(small_recalls)) if small_recalls else float("nan"),
            on_epoch=True,
        )

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Test step: compute metrics and accumulate per-patient rows.

        Parameters
        ----------
        batch : dict
            Batch dict with ``"image"``, ``"mask"``, ``"subject_id"``,
            ``"volume_stratum"`` keys.
        batch_idx : int
            Batch index.
        """
        images: Tensor = batch["image"]
        masks: Tensor = batch["mask"]
        subject_ids: list[str] = batch["subject_id"]
        strata: list[str] = batch["volume_stratum"]

        with torch.no_grad():
            logits = self.model(images)

        probs = torch.sigmoid(logits.detach())
        preds = (probs >= self.training_cfg.threshold).float()
        target = masks.detach()

        dice = dice_score(preds, target)
        iou = iou_score(preds, target)

        self.log("test_dice_mean", dice.mean(), on_epoch=True)
        self.log("test_iou_mean", iou.mean(), on_epoch=True)

        for i in range(images.size(0)):
            det = lesion_wise_detection(preds[i], target[i])
            self._per_patient_buffer.append(
                {
                    "subject_id": subject_ids[i],
                    "volume_stratum": strata[i],
                    "dice": float(dice[i].item()),
                    "iou": float(iou[i].item()),
                    "lesion_recall": det.lesion_recall,
                    "false_positives_per_slice": det.false_positives,
                }
            )

    def configure_optimizers(self) -> pl.utilities.types.OptimizerLRSchedulerConfig:
        """Configure AdamW + CosineAnnealingLR with epoch-level stepping.

        Returns
        -------
        dict
            Optimizer and LR scheduler configuration for Lightning.
        """
        cfg = self.training_cfg
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.max_epochs,
            eta_min=cfg.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_dice_mean",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_alpha_stats(self) -> dict[str, float]:
        """Collect ``alpha = softplus(alpha_raw)`` mean/std per Swin stage.

        Returns
        -------
        dict
            Keys like ``"alpha_mean_stage0"`` and ``"alpha_std_stage0"``.
            Empty dict for vanilla models.
        """
        stats: dict[str, float] = {}
        if not hasattr(self.model, "swinViT"):
            return stats
        swin = self.model.swinViT
        for i, attr in enumerate(["layers1", "layers2", "layers3", "layers4"]):
            if not hasattr(swin, attr):
                continue
            stage = getattr(swin, attr)[0]
            alphas: list[float] = []
            for block in stage.blocks:
                attn = block.attn
                if isinstance(attn, LpWindowAttention):
                    alpha = float(F.softplus(attn.lp_qknorm.alpha_raw).detach().item())
                    alphas.append(alpha)
            if alphas:
                stats[f"alpha_mean_stage{i}"] = float(np.mean(alphas))
                stats[f"alpha_std_stage{i}"] = float(np.std(alphas))
        return stats

    @staticmethod
    def _compute_precision(pred: Tensor, target: Tensor, eps: float = 1e-6) -> float:
        """Compute pixel-wise precision for a single sample."""
        tp = float((pred * target).sum().item())
        pp = float(pred.sum().item())
        return tp / (pp + eps)

    @staticmethod
    def _compute_recall(pred: Tensor, target: Tensor, eps: float = 1e-6) -> float:
        """Compute pixel-wise recall for a single sample."""
        tp = float((pred * target).sum().item())
        ap = float(target.sum().item())
        return tp / (ap + eps)
