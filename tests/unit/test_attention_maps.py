"""AT8 — Attention-map reconstruction and rollout round-trip tests."""

from __future__ import annotations

import pytest
import torch

from lpqknorm.probes.attention_maps import (
    attention_rollout,
    reconstruct_query_heatmap,
)


class TestReconstructQueryHeatmap:
    """AT8.1: single-query heatmap reconstruction."""

    def test_identity_attention_single_peak(self) -> None:
        """Identity attention + query_idx=5 → one non-zero cell, sum = 1."""
        w = 7
        n_win = 4  # 2x2 grid of windows
        attn = (
            torch.eye(w * w)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(n_win, 1, w * w, w * w)
            .contiguous()
        )
        heatmap = reconstruct_query_heatmap(
            attention=attn,
            query_idx=5,
            shift=0,
            grid_hw=(2 * w, 2 * w),
            window_size=w,
        )
        assert heatmap.shape == (2 * w, 2 * w)
        assert heatmap.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert int((heatmap > 0).sum().item()) == 1

    def test_identity_attention_peak_at_query(self) -> None:
        """The single non-zero cell sits at the query's coordinate."""
        w = 7
        n_win = 4
        attn = (
            torch.eye(w * w)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(n_win, 1, w * w, w * w)
            .contiguous()
        )
        query_idx = 3 * (2 * w) + 5  # row 3, col 5
        heatmap = reconstruct_query_heatmap(
            attention=attn,
            query_idx=query_idx,
            shift=0,
            grid_hw=(2 * w, 2 * w),
            window_size=w,
        )
        assert heatmap[3, 5].item() == pytest.approx(1.0, abs=1e-5)


class TestAttentionRollout:
    """AT8.2: Abnar-Zuidema rollout residual form."""

    def test_identity_layers_give_identity_rollout(self) -> None:
        """A = [I, I] → rollout = I."""
        attn = [torch.eye(10), torch.eye(10)]
        r = attention_rollout(attn)
        torch.testing.assert_close(r, torch.eye(10), atol=1e-6, rtol=0.0)

    def test_row_sums_preserved(self) -> None:
        """Every row of the rollout sums to 1."""
        torch.manual_seed(0)
        a = [
            torch.softmax(torch.randn(8, 8), dim=-1),
            torch.softmax(torch.randn(8, 8), dim=-1),
        ]
        r = attention_rollout(a)
        sums = r.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=0.0)

    def test_single_layer_no_op(self) -> None:
        """One layer: rollout = 0.5*(A+I) renormalised."""
        a = torch.softmax(torch.randn(4, 4), dim=-1)
        r = attention_rollout([a])
        assert r.shape == (4, 4)
        torch.testing.assert_close(r.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0.0)
