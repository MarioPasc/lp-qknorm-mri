"""Unit tests for CompoundSegLoss."""

from __future__ import annotations

import torch
import torch.testing

from lpqknorm.training.losses import CompoundSegLoss


class TestCompoundSegLoss:
    """Tests for the compound BCE + Dice loss."""

    def test_loss_is_non_negative(self) -> None:
        """All loss components should be non-negative."""
        loss_fn = CompoundSegLoss()
        logits = torch.randn(4, 1, 32, 32)
        target = (torch.randn(4, 1, 32, 32) > 0).float()
        total, bce, dice = loss_fn(logits, target)
        assert total.item() >= 0.0
        assert bce.item() >= 0.0
        assert dice.item() >= 0.0

    def test_weighted_sum(self) -> None:
        """Total equals bce_weight * bce + dice_weight * dice."""
        loss_fn = CompoundSegLoss(bce_weight=0.3, dice_weight=0.7)
        logits = torch.randn(2, 1, 16, 16)
        target = (torch.randn(2, 1, 16, 16) > 0).float()
        total, bce, dice = loss_fn(logits, target)
        expected = 0.3 * bce + 0.7 * dice
        torch.testing.assert_close(total, expected, atol=1e-6, rtol=0.0)

    def test_perfect_prediction_low_dice_loss(self) -> None:
        """For pred == target (large positive logits), Dice loss is near 0."""
        loss_fn = CompoundSegLoss()
        target = torch.ones(2, 1, 16, 16)
        logits = torch.full_like(target, 5.0)  # sigmoid(5) ≈ 0.993
        _, _, dice = loss_fn(logits, target)
        assert dice.item() < 0.05

    def test_empty_masks_both_zero(self) -> None:
        """All-zero target with large-negative logits → loss is finite."""
        loss_fn = CompoundSegLoss()
        target = torch.zeros(2, 1, 16, 16)
        logits = torch.full_like(target, -5.0)  # sigmoid(-5) ≈ 0.007
        total, _bce, dice = loss_fn(logits, target)
        assert total.isfinite()
        # MONAI DiceLoss returns ~1.0 for empty masks (no foreground);
        # the important thing is finiteness and non-NaN.
        assert dice.isfinite()

    def test_pos_weight_accepted(self) -> None:
        """CompoundSegLoss accepts pos_weight parameter."""
        pw = torch.tensor(10.0)
        loss_fn = CompoundSegLoss(pos_weight=pw)
        logits = torch.randn(2, 1, 16, 16)
        target = (torch.randn(2, 1, 16, 16) > 0).float()
        total, _, _ = loss_fn(logits, target)
        assert total.isfinite()

    def test_pos_weight_increases_fn_penalty(self) -> None:
        """With high pos_weight, BCE loss on false negatives is higher."""
        logits = torch.full((1, 1, 8, 8), -3.0)  # pred ≈ 0
        target = torch.ones(1, 1, 8, 8)  # all positive

        loss_no_weight = CompoundSegLoss(bce_weight=1.0, dice_weight=0.0)
        loss_weighted = CompoundSegLoss(
            bce_weight=1.0, dice_weight=0.0, pos_weight=torch.tensor(10.0)
        )

        total_no, _, _ = loss_no_weight(logits, target)
        total_w, _, _ = loss_weighted(logits, target)
        assert total_w.item() > total_no.item()

    def test_gradient_flows(self) -> None:
        """Loss is differentiable w.r.t. logits."""
        loss_fn = CompoundSegLoss()
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = (torch.randn(2, 1, 16, 16) > 0).float()
        total, _, _ = loss_fn(logits, target)
        total.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
