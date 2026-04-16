"""Unit tests for lesion-mask tokenization and window partition (AT3 + AT4)."""

from __future__ import annotations

import torch

from lpqknorm.probes.tokenization import mask_to_token_flags, window_partition_flags


class TestMaskToTokenFlags:
    """AT3: Mask-to-token-flags roundtrip."""

    def test_lesion_token_tagging_roundtrip(self) -> None:
        """2x2 lesion at pixel (100,100) → token at flat index 50*112+50."""
        mask = torch.zeros(1, 1, 224, 224)
        mask[0, 0, 100:102, 100:102] = 1.0
        flags = mask_to_token_flags(mask, patch_stride=(2, 2))
        assert flags.shape == (1, 112 * 112)
        # Pixel (100,100) with stride 2 → token (50, 50) → flat 50*112+50
        expected = 50 * 112 + 50
        assert flags[0, expected].item() is True
        assert int(flags.sum().item()) >= 1

    def test_no_lesion_gives_all_false(self) -> None:
        """Empty mask → all False."""
        mask = torch.zeros(1, 1, 224, 224)
        flags = mask_to_token_flags(mask, patch_stride=(2, 2))
        assert not flags.any()

    def test_full_lesion_gives_all_true(self) -> None:
        """Full mask → all True."""
        mask = torch.ones(1, 1, 224, 224)
        flags = mask_to_token_flags(mask, patch_stride=(2, 2))
        assert flags.all()

    def test_batch_independent(self) -> None:
        """Batch elements are processed independently."""
        mask = torch.zeros(2, 1, 224, 224)
        mask[0, 0, 50:60, 50:60] = 1.0
        flags = mask_to_token_flags(mask, patch_stride=(2, 2))
        assert flags[0].any()
        assert not flags[1].any()

    def test_single_pixel_lesion(self) -> None:
        """A single pixel in a 2x2 patch is enough to flag the token."""
        mask = torch.zeros(1, 1, 224, 224)
        mask[0, 0, 0, 0] = 1.0
        flags = mask_to_token_flags(mask, patch_stride=(2, 2))
        assert flags[0, 0].item() is True
        assert int(flags.sum().item()) == 1


class TestWindowPartitionFlags:
    """AT4: Window partition shape and correctness."""

    def test_shape(self) -> None:
        """Output shape is (B*nW, W²)."""
        flags = torch.zeros(1, 112 * 112, dtype=torch.bool)
        wp = window_partition_flags(flags, img_hw_tok=(112, 112), window_size=7)
        # nW = (112/7)² = 16² = 256
        assert wp.shape == (256, 49)

    def test_wmsa_token_location(self) -> None:
        """Token at (50,50) in grid → correct window and position."""
        flags = torch.zeros(1, 112 * 112, dtype=torch.bool)
        flags[0, 50 * 112 + 50] = True
        wp = window_partition_flags(
            flags, img_hw_tok=(112, 112), window_size=7, shift_size=0
        )
        # Token (50,50): window_row=50//7=7, window_col=50//7=7
        # nW_cols = 112//7 = 16
        # window_idx = 7*16 + 7 = 119
        # within-window: row=50%7=1, col=50%7=1 → pos = 1*7+1 = 8
        assert wp[119, 8].item() is True
        assert int(wp.sum().item()) == 1

    def test_shift_changes_window_assignment(self) -> None:
        """SW-MSA shift moves the token to a different window."""
        flags = torch.zeros(1, 112 * 112, dtype=torch.bool)
        flags[0, 50 * 112 + 50] = True
        wp_no = window_partition_flags(flags, (112, 112), window_size=7, shift_size=0)
        wp_sh = window_partition_flags(flags, (112, 112), window_size=7, shift_size=3)
        assert not torch.equal(wp_no, wp_sh)
        assert int(wp_no.sum()) == 1
        assert int(wp_sh.sum()) == 1

    def test_total_tokens_preserved(self) -> None:
        """Window partition preserves total number of True tokens."""
        torch.manual_seed(0)
        flags = torch.rand(2, 112 * 112) > 0.95
        n_true = int(flags.sum().item())
        wp = window_partition_flags(flags, (112, 112), window_size=7)
        assert int(wp.sum().item()) == n_true

    def test_batch_size_two(self) -> None:
        """B=2 produces 2*256=512 windows."""
        flags = torch.zeros(2, 112 * 112, dtype=torch.bool)
        wp = window_partition_flags(flags, (112, 112), window_size=7)
        assert wp.shape == (512, 49)
