"""Compound segmentation loss for binary stroke lesion segmentation.

Combines binary cross-entropy with logits and soft Dice loss in a weighted
sum.  The BCE component supports ``pos_weight`` to address class imbalance
(lesion voxels are a small fraction of each slice).

Mathematical specification::

    L = w_bce * BCE(logits, target) + w_dice * DiceLoss(logits, target)

where ``DiceLoss`` is MONAI's implementation with ``sigmoid=True`` applied
internally to the raw logits.

References
----------
- Isensee et al. *nnU-Net*. Nat. Methods 2021.
  doi:10.1038/s41592-020-01008-z (loss convention).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from monai.losses import DiceLoss  # type: ignore[attr-defined]
from torch import Tensor


logger = logging.getLogger(__name__)


class CompoundSegLoss(nn.Module):
    """Weighted BCE-with-logits + soft Dice loss for binary segmentation.

    Parameters
    ----------
    bce_weight : float
        Weight for the BCE component.  Default ``0.5``.
    dice_weight : float
        Weight for the Dice component.  Default ``0.5``.
    pos_weight : Tensor or None
        Positive-class weight tensor for ``BCEWithLogitsLoss`` (addresses
        class imbalance).  Shape ``()`` scalar or ``(1,)``.  Computed from
        the training set at ``DataModule.setup()`` time.

    Notes
    -----
    ``DiceLoss`` is configured with ``sigmoid=True`` so it applies sigmoid
    to raw logits internally.  ``BCEWithLogitsLoss`` also applies sigmoid
    internally.  Both losses expect raw logit inputs.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # Reshape a 1D (K,) per-class pos_weight to (K, 1, 1) so it broadcasts
        # correctly against (B, K, H, W) logits/targets.  Without this, a
        # (K,) tensor aligns with the trailing width dim and errors.
        if pos_weight is not None and pos_weight.dim() == 1 and pos_weight.numel() > 1:
            pos_weight = pos_weight.view(-1, 1, 1)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)

    def forward(self, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the compound loss.

        Parameters
        ----------
        logits : Tensor
            Raw unnormalized predictions, shape ``(B, 1, H, W)``.
        target : Tensor
            Binary ground-truth masks, shape ``(B, 1, H, W)``, float.

        Returns
        -------
        loss_total : Tensor
            Scalar.  Weighted sum of BCE and Dice.
        loss_bce : Tensor
            Scalar.  BCE component (before weighting).
        loss_dice : Tensor
            Scalar.  Dice component (before weighting).
        """
        # Disable autocast around BCE + Dice.  BCE-with-logits in bf16
        # becomes unstable when |logits| grows (the log-sum-exp internals
        # saturate) and MONAI's soft Dice uses 1e-5 smoothing which can
        # underflow in bf16 (< 6e-5 smallest normal).  The loss cost is a
        # tiny fraction of the forward, so fp32 evaluation is free in
        # practice and removes a class of bf16 NaNs.
        device_type = (
            logits.device.type if logits.device.type in {"cuda", "cpu"} else "cpu"
        )
        with torch.amp.autocast(device_type=device_type, enabled=False):
            logits_f32 = logits.float()
            target_f32 = target.float()
            loss_bce = self.bce(logits_f32, target_f32)
            loss_dice = self.dice(logits_f32, target_f32)
            loss_total = self.bce_weight * loss_bce + self.dice_weight * loss_dice
        return loss_total, loss_bce, loss_dice
