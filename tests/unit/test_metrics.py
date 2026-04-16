"""Unit tests for segmentation metrics."""

from __future__ import annotations

import math

import numpy as np
import torch

from lpqknorm.training.metrics import (
    dice_score,
    hd95,
    iou_score,
    lesion_wise_detection,
)


class TestDiceScore:
    """Tests for dice_score."""

    def test_perfect_prediction(self) -> None:
        """Pred == target → Dice ≈ 1.0."""
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 10:20, 10:20] = 1.0
        dice = dice_score(target, target)
        np.testing.assert_allclose(dice.numpy(), 1.0, atol=1e-5)

    def test_both_empty(self) -> None:
        """Both pred and target empty → Dice = 1.0 (convention)."""
        pred = torch.zeros(2, 1, 16, 16)
        target = torch.zeros(2, 1, 16, 16)
        dice = dice_score(pred, target)
        np.testing.assert_allclose(dice.numpy(), 1.0, atol=1e-6)

    def test_no_overlap(self) -> None:
        """Non-overlapping pred and target → Dice ≈ 0."""
        pred = torch.zeros(1, 1, 16, 16)
        pred[0, 0, 0:4, 0:4] = 1.0
        target = torch.zeros(1, 1, 16, 16)
        target[0, 0, 12:16, 12:16] = 1.0
        dice = dice_score(pred, target)
        assert dice.item() < 0.01

    def test_shape(self) -> None:
        """Output shape is (B,)."""
        pred = torch.zeros(3, 1, 8, 8)
        target = torch.zeros(3, 1, 8, 8)
        dice = dice_score(pred, target)
        assert dice.shape == (3,)


class TestIoUScore:
    """Tests for iou_score."""

    def test_perfect_prediction(self) -> None:
        """Pred == target → IoU ≈ 1.0."""
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1.0
        iou = iou_score(target, target)
        np.testing.assert_allclose(iou.numpy(), 1.0, atol=1e-5)

    def test_both_empty(self) -> None:
        """Both empty → IoU = 1.0 (convention)."""
        pred = torch.zeros(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)
        iou = iou_score(pred, target)
        assert abs(iou.item() - 1.0) < 1e-5

    def test_half_overlap(self) -> None:
        """50% overlap yields IoU ≈ 1/3."""
        pred = torch.zeros(1, 1, 20, 20)
        target = torch.zeros(1, 1, 20, 20)
        pred[0, 0, 0:10, 0:10] = 1.0
        target[0, 0, 5:15, 0:10] = 1.0
        # intersection = 5*10=50, union = 10*10 + 10*10 - 50 = 150
        iou = iou_score(pred, target)
        np.testing.assert_allclose(iou.numpy(), 50.0 / 150.0, atol=0.02)


class TestLesionWiseDetection:
    """Tests for lesion_wise_detection."""

    def test_two_lesions_one_detected(self) -> None:
        """Two GT lesions, pred overlaps only one → recall = 0.5."""
        target = torch.zeros(1, 32, 32)
        target[0, 2:6, 2:6] = 1.0  # lesion 1
        target[0, 20:24, 20:24] = 1.0  # lesion 2

        pred = torch.zeros(1, 32, 32)
        pred[0, 2:6, 2:6] = 1.0  # matches lesion 1 only

        result = lesion_wise_detection(pred, target)
        assert result.n_gt_lesions == 2
        assert result.n_detected == 1
        assert abs(result.lesion_recall - 0.5) < 1e-6

    def test_no_gt_lesions(self) -> None:
        """No GT lesions → recall is NaN, all pred are false positives."""
        target = torch.zeros(32, 32)
        pred = torch.zeros(32, 32)
        pred[5:10, 5:10] = 1.0

        result = lesion_wise_detection(pred, target)
        assert result.n_gt_lesions == 0
        assert math.isnan(result.lesion_recall)
        assert result.false_positives == 1

    def test_perfect_detection(self) -> None:
        """Pred matches all GT exactly → recall = 1.0, fp = 0."""
        target = torch.zeros(32, 32)
        target[5:10, 5:10] = 1.0
        pred = target.clone()
        result = lesion_wise_detection(pred, target)
        assert result.n_gt_lesions == 1
        assert result.n_detected == 1
        assert abs(result.lesion_recall - 1.0) < 1e-6
        assert result.false_positives == 0

    def test_empty_pred_and_target(self) -> None:
        """Both empty → 0 GT, NaN recall, 0 FP."""
        pred = torch.zeros(16, 16)
        target = torch.zeros(16, 16)
        result = lesion_wise_detection(pred, target)
        assert result.n_gt_lesions == 0
        assert math.isnan(result.lesion_recall)
        assert result.false_positives == 0


class TestHD95:
    """Tests for hd95."""

    def test_perfect_prediction(self) -> None:
        """Pred == target → HD95 ≈ 0."""
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, 10:20, 10:20] = 1.0
        hd_val = hd95(target, target)
        assert hd_val.shape == (1, 1)
        assert hd_val.item() < 1.0

    def test_empty_pred_returns_nan(self) -> None:
        """Empty pred with non-empty target → NaN (MONAI behaviour)."""
        pred = torch.zeros(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)
        target[0, 0, 4:8, 4:8] = 1.0
        hd_val = hd95(pred, target)
        assert math.isnan(hd_val.item())

    def test_output_shape(self) -> None:
        """Output shape is (B, 1)."""
        pred = torch.zeros(3, 1, 16, 16)
        pred[:, :, 2:6, 2:6] = 1.0
        target = pred.clone()
        hd_val = hd95(pred, target)
        assert hd_val.shape == (3, 1)
