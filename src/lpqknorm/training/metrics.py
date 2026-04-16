"""Segmentation metrics for binary stroke lesion evaluation.

Provides Dice coefficient, IoU, 95th-percentile Hausdorff distance, and
connected-component lesion-wise detection statistics.  These metrics are
used in both the validation loop (per-epoch monitoring) and the final test
evaluation (per-patient and per-lesion tables).

References
----------
- Isensee et al. *nnU-Net*. Nat. Methods 2021 (metric conventions).
- Liew et al. *ATLAS v2.0*. Scientific Data 2022 (lesion-wise recall).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance  # type: ignore[attr-defined]
from scipy.ndimage import label as ndimage_label
from torch import Tensor


logger = logging.getLogger(__name__)

LESION_DETECTION_IOU_THRESHOLD: float = 0.1
"""Default IoU threshold for lesion-wise detection."""


@dataclass(frozen=True)
class LesionDetectionResult:
    """Per-image lesion-wise detection result.

    Parameters
    ----------
    n_gt_lesions : int
        Total ground-truth connected components.
    n_detected : int
        Number of GT lesions detected (IoU with any pred >= threshold).
    lesion_recall : float
        Fraction detected.  ``NaN`` if ``n_gt_lesions == 0``.
    false_positives : int
        Predicted components with no matching GT lesion.
    pred_ious : list[float]
        Best IoU of each predicted component against GT.
    gt_ious : list[float]
        Best IoU of each GT lesion against predicted components.
    """

    n_gt_lesions: int
    n_detected: int
    lesion_recall: float
    false_positives: int
    pred_ious: list[float]
    gt_ious: list[float]


def dice_score(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute per-sample Dice coefficient for binary predictions.

    Parameters
    ----------
    pred : Tensor
        Shape ``(B, 1, H, W)``.  Binary predictions (0 or 1).
    target : Tensor
        Shape ``(B, 1, H, W)``.  Binary ground truth (0 or 1).
    eps : float
        Numerical stability constant.

    Returns
    -------
    Tensor
        Shape ``(B,)``.  Per-sample Dice scores in ``[0, 1]``.
        Returns ``1.0`` for samples where both pred and target are all-zero.
    """
    pred_flat = pred.float().view(pred.size(0), -1)
    target_flat = target.float().view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return torch.where(
        union == 0,
        torch.ones_like(intersection),
        (2.0 * intersection + eps) / (union + eps),
    )


def iou_score(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute per-sample Intersection-over-Union for binary predictions.

    Parameters
    ----------
    pred : Tensor
        Shape ``(B, 1, H, W)``.  Binary predictions.
    target : Tensor
        Shape ``(B, 1, H, W)``.  Binary ground truth.
    eps : float
        Numerical stability constant.

    Returns
    -------
    Tensor
        Shape ``(B,)``.  Per-sample IoU in ``[0, 1]``.
        Returns ``1.0`` for samples where both pred and target are all-zero.
    """
    pred_flat = pred.float().view(pred.size(0), -1)
    target_flat = target.float().view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = (pred_flat + target_flat).clamp(max=1).sum(dim=1)
    return torch.where(
        union == 0,
        torch.ones_like(intersection),
        (intersection + eps) / (union + eps),
    )


def lesion_wise_detection(
    pred: Tensor,
    target: Tensor,
    iou_threshold: float = LESION_DETECTION_IOU_THRESHOLD,
) -> LesionDetectionResult:
    """Compute lesion-wise detection statistics via connected components.

    Parameters
    ----------
    pred : Tensor
        Shape ``(1, H, W)`` or ``(H, W)``.  Binary prediction for one image.
    target : Tensor
        Shape ``(1, H, W)`` or ``(H, W)``.  Binary ground truth for one image.
    iou_threshold : float
        Minimum IoU for a GT lesion to be considered detected.

    Returns
    -------
    LesionDetectionResult
        Frozen dataclass with detection statistics.

    Notes
    -----
    Uses ``scipy.ndimage.label`` for 2D connected-component labeling with
    the default structuring element (4-connectivity).

    A GT lesion is *detected* if any predicted component overlaps it with
    ``IoU >= iou_threshold``.  A predicted component is a *false positive*
    if it does not match any GT lesion at the threshold.
    """
    pred_np = pred.detach().squeeze().cpu().numpy().astype(bool)
    target_np = target.detach().squeeze().cpu().numpy().astype(bool)

    pred_labeled, n_pred = ndimage_label(pred_np)
    target_labeled, n_gt = ndimage_label(target_np)

    if n_gt == 0:
        return LesionDetectionResult(
            n_gt_lesions=0,
            n_detected=0,
            lesion_recall=math.nan,
            false_positives=n_pred,
            pred_ious=[],
            gt_ious=[],
        )

    gt_detected = [False] * n_gt
    pred_matched = [False] * n_pred
    gt_ious: list[float] = [0.0] * n_gt
    pred_ious: list[float] = [0.0] * n_pred

    # For each GT lesion, find best-matching predicted component
    for gt_idx in range(n_gt):
        gt_mask = target_labeled == (gt_idx + 1)
        best_iou = 0.0
        for pred_idx in range(n_pred):
            pred_mask = pred_labeled == (pred_idx + 1)
            inter = int(np.sum(gt_mask & pred_mask))
            union = int(np.sum(gt_mask | pred_mask))
            cur_iou = inter / union if union > 0 else 0.0
            if cur_iou > best_iou:
                best_iou = cur_iou
            if cur_iou >= iou_threshold:
                pred_matched[pred_idx] = True
        gt_ious[gt_idx] = best_iou
        gt_detected[gt_idx] = best_iou >= iou_threshold

    # For each predicted component, find best-matching GT lesion
    for pred_idx in range(n_pred):
        pred_mask = pred_labeled == (pred_idx + 1)
        best_iou = 0.0
        for gt_idx in range(n_gt):
            gt_mask = target_labeled == (gt_idx + 1)
            inter = int(np.sum(pred_mask & gt_mask))
            union = int(np.sum(pred_mask | gt_mask))
            cur_iou = inter / union if union > 0 else 0.0
            best_iou = max(best_iou, cur_iou)
        pred_ious[pred_idx] = best_iou

    n_detected = sum(gt_detected)
    fp = sum(1 for m in pred_matched if not m)

    return LesionDetectionResult(
        n_gt_lesions=n_gt,
        n_detected=n_detected,
        lesion_recall=n_detected / n_gt,
        false_positives=fp,
        pred_ious=pred_ious,
        gt_ious=gt_ious,
    )


def hd95(pred: Tensor, target: Tensor) -> Tensor:
    """Compute 95th-percentile Hausdorff distance using MONAI.

    Parameters
    ----------
    pred : Tensor
        Shape ``(B, 1, H, W)``.  Binary predictions.
    target : Tensor
        Shape ``(B, 1, H, W)``.  Binary ground truth.

    Returns
    -------
    Tensor
        Shape ``(B, 1)``.  HD95 values in pixels.  Returns ``NaN`` for
        samples where either pred or target is empty (MONAI behaviour).

    Notes
    -----
    Wraps ``monai.metrics.compute_hausdorff_distance(percentile=95)``.
    """
    with torch.no_grad():
        return compute_hausdorff_distance(
            pred.float(),
            target.float(),
            include_background=False,
            distance_metric="euclidean",
            percentile=95,
        )
