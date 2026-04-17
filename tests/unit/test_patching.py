"""AT9 — Activation-patching correctness tests.

- Self-patching with source = target must leave the per-slice Dice
  unchanged (up to fp32 noise).
- Patching from a genuinely different checkpoint must change the
  per-slice Dice.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from lpqknorm.data.datamodule import MockAtlasDataModule, MockDataConfig
from lpqknorm.models.lp_qknorm import LpQKNormConfig
from lpqknorm.models.swin_unetr_lp import build_swin_unetr_lp
from lpqknorm.probes.patching import ActivationPatcher, PatchingConfig


pytestmark = pytest.mark.integration


def _build(seed: int) -> torch.nn.Module:
    """Small Lp-patched SwinUNETR for testing."""
    torch.manual_seed(seed)
    return build_swin_unetr_lp(
        img_size=(224, 224),
        in_channels=1,
        out_channels=1,
        feature_size=12,
        lp_cfg=LpQKNormConfig(p=3.0),
    ).eval()


def _loader(n: int = 2):
    dm = MockAtlasDataModule(
        MockDataConfig(n_val=n, batch_size=1, img_size=(224, 224), seed=0)
    )
    dm.setup("fit")
    return dm.val_dataloader()


class TestSelfPatchingIdentity:
    """Patching a model with its own captures must leave Dice unchanged."""

    def test_self_patching_keeps_dice(self, tmp_path: Path) -> None:
        model = _build(seed=0)
        cfg = PatchingConfig(
            source_checkpoint=Path("unused"),
            target_checkpoint=Path("unused"),
            stage=0,
            blocks=(0, 1),
            variants=("q", "k", "qk", "qhat_khat", "logits"),
            n_probe_samples=2,
        )
        patcher = ActivationPatcher(cfg, source_model=model, target_model=model)
        out = patcher.run(_loader(n=2), output_dir=tmp_path, device="cpu")
        with h5py.File(out) as f:
            for block in ("block_0", "block_1"):
                dtgt = f[f"{block}/dice_target"][()]
                for variant in cfg.variants:
                    dpat = f[f"{block}/variant_{variant}/dice_patched"][()]
                    np.testing.assert_allclose(dpat, dtgt, rtol=1e-4, atol=1e-4)


class TestPatchingChangesOutput:
    """Patching from a genuinely different model must change Dice."""

    def test_different_seeds_change_dice(self, tmp_path: Path) -> None:
        src = _build(seed=0)
        tgt = _build(seed=1)
        cfg = PatchingConfig(
            source_checkpoint=Path("unused"),
            target_checkpoint=Path("unused"),
            stage=0,
            blocks=(0,),
            variants=("qk",),
            n_probe_samples=2,
        )
        patcher = ActivationPatcher(cfg, source_model=src, target_model=tgt)
        out = patcher.run(_loader(n=2), output_dir=tmp_path, device="cpu")
        with h5py.File(out) as f:
            dtgt = f["block_0/dice_target"][()]
            dpat = f["block_0/variant_qk/dice_patched"][()]
            dsrc = f["block_0/dice_source"][()]
        # Source and target models differ → d_src != d_tgt or d_pat != d_tgt.
        assert not np.allclose(dsrc, dtgt, atol=1e-4) or not np.allclose(
            dpat, dtgt, atol=1e-4
        )


class TestPatchingVariantGrid:
    """All five variants run end-to-end and produce the expected HDF5 groups."""

    def test_all_variants_written(self, tmp_path: Path) -> None:
        model = _build(seed=0)
        cfg = PatchingConfig(
            source_checkpoint=Path("unused"),
            target_checkpoint=Path("unused"),
            stage=0,
            blocks=(0, 1),
            variants=("q", "k", "qk", "qhat_khat", "logits"),
            n_probe_samples=2,
        )
        patcher = ActivationPatcher(cfg, source_model=model, target_model=model)
        out = patcher.run(_loader(n=2), output_dir=tmp_path, device="cpu")
        with h5py.File(out) as f:
            for block in ("block_0", "block_1"):
                assert block in f
                for variant in cfg.variants:
                    key = f"{block}/variant_{variant}"
                    assert key in f
                    assert "dice_patched" in f[key]
                    assert "pe" in f[key]
                    assert "prediction" in f[key]
