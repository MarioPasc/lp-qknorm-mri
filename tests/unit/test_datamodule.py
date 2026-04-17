"""Unit tests for data/datamodule.py — DataModule and Mock classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path
import torch

from lpqknorm.data.converter import PreprocessConfig, write_standardized_h5
from lpqknorm.data.datamodule import (
    MockAtlasDataModule,
    MockDataConfig,
    SegmentationDataModule,
)
from tests.fixtures.synthetic_dataset import (
    make_synthetic_info,
    make_synthetic_subjects,
)


@pytest.fixture
def synthetic_h5(tmp_path: Path) -> Path:
    """Write a synthetic HDF5 with some lesion-free slices."""
    subjects = make_synthetic_subjects(
        n_subjects=12,
        depth=12,
        img_size=(32, 32),
        seed=42,
    )
    info = make_synthetic_info()
    cfg = PreprocessConfig(in_plane_size=(32, 32), min_lesion_voxels_per_slice=10)
    out = tmp_path / "test.h5"
    write_standardized_h5(subjects, info, cfg, out)
    return out


class TestMockAtlasDataModule:
    """Verify MockAtlasDataModule produces the expected API."""

    def test_basic_api(self) -> None:
        cfg = MockDataConfig(n_train=8, n_val=4, n_test=2, batch_size=2)
        dm = MockAtlasDataModule(cfg)
        dm.setup("fit")

        assert dm.n_train == 8
        assert dm.n_val == 4
        assert dm.n_test == 2
        assert isinstance(dm.split_hash, str)
        assert len(dm.split_hash) == 16
        assert isinstance(dm.pos_weight, torch.Tensor)

    def test_train_batch_shape(self) -> None:
        cfg = MockDataConfig(
            n_train=4, n_val=2, n_test=2, batch_size=2, img_size=(64, 64)
        )
        dm = MockAtlasDataModule(cfg)
        dm.setup("fit")

        batch = next(iter(dm.train_dataloader()))
        assert "image" in batch
        assert "mask" in batch
        assert batch["image"].shape == (2, 1, 64, 64)
        assert batch["mask"].shape == (2, 1, 64, 64)

    def test_val_batch_has_metadata(self) -> None:
        cfg = MockDataConfig(n_val=4, batch_size=2)
        dm = MockAtlasDataModule(cfg)
        dm.setup("fit")

        batch = next(iter(dm.val_dataloader()))
        assert "subject_id" in batch
        assert "volume_stratum" in batch
        assert isinstance(batch["subject_id"], list)
        assert len(batch["subject_id"]) == 2

    def test_deterministic_split_hash(self) -> None:
        cfg1 = MockDataConfig(seed=42)
        cfg2 = MockDataConfig(seed=42)
        dm1 = MockAtlasDataModule(cfg1)
        dm2 = MockAtlasDataModule(cfg2)
        assert dm1.split_hash == dm2.split_hash

    def test_masks_have_lesion(self) -> None:
        cfg = MockDataConfig(n_train=8, batch_size=4)
        dm = MockAtlasDataModule(cfg)
        dm.setup("fit")

        for batch in dm.train_dataloader():
            # Each mock sample has a lesion blob
            assert batch["mask"].sum() > 0


class TestSegmentationDataModule2D:
    """AT12-AT15: 2D mode DataModule tests on synthetic HDF5."""

    def test_lesion_only_true(self, synthetic_h5: Path) -> None:
        dm = SegmentationDataModule(
            synthetic_h5,
            fold=0,
            spatial_mode="2d",
            batch_size=4,
            lesion_only=True,
            num_workers=0,
            augment=False,
        )
        dm.setup("fit")

        assert dm.n_train > 0
        for batch in dm.train_dataloader():
            # Every sample should have mask > 0
            for i in range(batch["mask"].shape[0]):
                assert batch["mask"][i].sum() > 0

    def test_lesion_only_false_includes_empty(self, synthetic_h5: Path) -> None:
        dm = SegmentationDataModule(
            synthetic_h5,
            fold=0,
            spatial_mode="2d",
            batch_size=4,
            lesion_only=False,
            num_workers=0,
            augment=False,
        )
        dm.setup("fit")

        has_empty = False
        for batch in dm.train_dataloader():
            for i in range(batch["mask"].shape[0]):
                if batch["mask"][i].sum() == 0:
                    has_empty = True
                    break
            if has_empty:
                break
        assert has_empty, "lesion_only=False should include empty-mask slices"

    def test_val_has_metadata(self, synthetic_h5: Path) -> None:
        dm = SegmentationDataModule(
            synthetic_h5,
            fold=0,
            spatial_mode="2d",
            batch_size=4,
            num_workers=0,
            augment=False,
        )
        dm.setup("fit")

        batch = next(iter(dm.val_dataloader()))
        assert "subject_id" in batch
        assert "volume_stratum" in batch

    def test_header_access(self, synthetic_h5: Path) -> None:
        dm = SegmentationDataModule(
            synthetic_h5,
            fold=0,
            spatial_mode="2d",
            batch_size=4,
            num_workers=0,
        )
        header = dm.header
        assert header.n_subjects == 12
        assert header.n_modalities == 1

    def test_split_hash(self, synthetic_h5: Path) -> None:
        dm = SegmentationDataModule(
            synthetic_h5,
            fold=0,
            spatial_mode="2d",
            batch_size=4,
            num_workers=0,
        )
        assert len(dm.split_hash) == 16

    def test_pos_weight_shape(self, synthetic_h5: Path) -> None:
        dm = SegmentationDataModule(
            synthetic_h5,
            fold=0,
            spatial_mode="2d",
            batch_size=4,
            num_workers=0,
        )
        dm.setup("fit")
        assert dm.pos_weight is not None
        assert dm.pos_weight.shape == (1,)  # n_label_classes = 1
