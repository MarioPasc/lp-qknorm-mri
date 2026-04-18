"""Segmentation DataModule with dual-mode (2D/3D) HDF5 loading.

Provides :class:`SegmentationDataModule` for production use with
standardized HDF5 files, and :class:`MockAtlasDataModule` for fast
integration testing without real data.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lpqknorm.data.schema import DatasetHeader
from lpqknorm.data.transforms import get_train_transforms_2d, get_val_transforms_2d
from lpqknorm.utils.seeding import seed_worker


logger = logging.getLogger(__name__)


def _pin_memory_available() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        torch.zeros(1).pin_memory()
        return True
    except Exception:
        return False


def _safe_mp_context(num_workers: int) -> str | None:
    """Pick a fork-safe multiprocessing context for DataLoader workers.

    When CUDA is initialized in the parent process (which Lightning does
    before iterating the train DataLoader, e.g. by moving the module and
    its buffers to GPU), forking a worker inherits a corrupted CUDA state
    and the first destructor of a CUDA-backed tensor aborts with
    ``CUDA error: initialization error``.  ``spawn`` re-imports the module
    in the worker and avoids the issue.  On CPU-only runs the default
    context is fine.
    """
    if num_workers <= 0:
        return None
    if torch.cuda.is_available():
        return "spawn"
    return None


# ---------------------------------------------------------------------------
# Internal HDF5-backed dataset
# ---------------------------------------------------------------------------


class _SliceDataset(Dataset):  # type: ignore[type-arg]
    """HDF5-backed 2D slice dataset with lazy loading."""

    def __init__(
        self,
        h5_path: Path,
        row_indices: np.ndarray,
        transform: object | None = None,
        with_metadata: bool = False,
        subject_ids: np.ndarray | None = None,
        strata: np.ndarray | None = None,
    ) -> None:
        self.h5_path = h5_path
        self.row_indices = row_indices
        self.transform = transform
        self.with_metadata = with_metadata
        self.subject_ids = subject_ids
        self.strata = strata
        self._h5_file: h5py.File | None = None

    def _get_h5(self) -> h5py.File:
        # Per-worker file handle (set None on fork so each worker opens its own)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = int(self.row_indices[idx])
        f = self._get_h5()

        image = f["data/images"][row].astype(np.float32)  # (C, H, W)
        mask = f["data/masks"][row].astype(np.float32)  # (K, H, W)

        sample: dict[str, Any] = {"image": image, "mask": mask}

        if self.transform is not None:
            sample = self.transform(sample)  # type: ignore[operator]

        # Ensure tensors
        if not isinstance(sample["image"], Tensor):
            sample["image"] = torch.as_tensor(sample["image"])
        if not isinstance(sample["mask"], Tensor):
            sample["mask"] = torch.as_tensor(sample["mask"])

        if (
            self.with_metadata
            and self.subject_ids is not None
            and self.strata is not None
        ):
            sample["subject_id"] = str(self.subject_ids[idx])
            sample["volume_stratum"] = str(self.strata[idx])

        return sample

    def __del__(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Production DataModule
# ---------------------------------------------------------------------------


class SegmentationDataModule(pl.LightningDataModule):
    """Dual-mode (2D/3D) DataModule reading from a standardized HDF5 file.

    Parameters
    ----------
    h5_path : Path
        Path to the standardized HDF5 file.
    fold : int
        Fold index to use for train/val/test splits.
    spatial_mode : {"2d", "3d"}
        Loading mode. ``"2d"`` loads individual slices; ``"3d"`` loads
        full volumes.
    batch_size : int
        Batch size for DataLoaders.
    lesion_only : bool
        If ``True`` (default), only slices with ``has_lesion=True`` are
        included in 2D mode training.
    num_workers : int
        Number of DataLoader workers.
    augment : bool
        Whether to apply training augmentations.
    """

    def __init__(
        self,
        h5_path: Path | str,
        fold: int = 0,
        spatial_mode: Literal["2d", "3d"] = "2d",
        batch_size: int = 16,
        lesion_only: bool = True,
        num_workers: int = 8,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.fold = fold
        self.spatial_mode = spatial_mode
        self.batch_size = batch_size
        self.lesion_only = lesion_only
        self.num_workers = num_workers
        self.augment = augment

        self._header: DatasetHeader | None = None
        self._split_hash: str | None = None
        self._train_indices: np.ndarray | None = None
        self._val_indices: np.ndarray | None = None
        self._test_indices: np.ndarray | None = None
        self._train_sids: np.ndarray | None = None
        self._val_sids: np.ndarray | None = None
        self._test_sids: np.ndarray | None = None
        self._train_strata: np.ndarray | None = None
        self._val_strata: np.ndarray | None = None
        self._test_strata: np.ndarray | None = None
        self._pos_weight: Tensor | None = None

    @property
    def header(self) -> DatasetHeader:
        if self._header is None:
            self._header = DatasetHeader.from_h5(self.h5_path)
        return self._header

    @property
    def split_hash(self) -> str:
        if self._split_hash is None:
            with h5py.File(self.h5_path, "r") as f:
                self._split_hash = str(f["splits"].attrs["split_hash"])
        return self._split_hash

    @property
    def n_train(self) -> int:
        assert self._train_indices is not None, "Call setup() first"
        return len(self._train_indices)

    @property
    def n_val(self) -> int:
        assert self._val_indices is not None, "Call setup() first"
        return len(self._val_indices)

    @property
    def n_test(self) -> int:
        assert self._test_indices is not None, "Call setup() first"
        return len(self._test_indices)

    @property
    def pos_weight(self) -> Tensor | None:
        return self._pos_weight

    def setup(self, stage: str | None = None) -> None:
        """Read metadata and build partition indices."""
        with h5py.File(self.h5_path, "r") as f:
            self._header = DatasetHeader.from_h5(self.h5_path)
            self._split_hash = str(f["splits"].attrs["split_hash"])

            # Read split subject IDs
            fold_grp = f[f"splits/fold_{self.fold}"]
            train_subjects = set(
                s.decode() if isinstance(s, bytes) else s
                for s in fold_grp["train_subjects"][()]
            )
            val_subjects = set(
                s.decode() if isinstance(s, bytes) else s
                for s in fold_grp["val_subjects"][()]
            )
            test_subjects = set(
                s.decode() if isinstance(s, bytes) else s
                for s in fold_grp["test_subjects"][()]
            )

            # Read volume index and slice metadata
            vi_sids = np.array(
                [
                    s.decode() if isinstance(s, bytes) else s
                    for s in f["volume_index/subject_id"][()]
                ]
            )
            vi_starts = f["volume_index/start_row"][()]
            vi_ends = f["volume_index/end_row"][()]

            slice_has_lesion = f["slices/has_lesion"][()]
            slice_sids = np.array(
                [
                    s.decode() if isinstance(s, bytes) else s
                    for s in f["slices/subject_id"][()]
                ]
            )

            # Subject strata lookup
            subj_sids = np.array(
                [
                    s.decode() if isinstance(s, bytes) else s
                    for s in f["subjects/subject_id"][()]
                ]
            )
            subj_strata = np.array(
                [
                    s.decode() if isinstance(s, bytes) else s
                    for s in f["subjects/volume_stratum"][()]
                ]
            )
            sid_to_stratum = dict(zip(subj_sids.tolist(), subj_strata.tolist()))

            # Compute pos_weight from training partition masks
            if stage in ("fit", None):
                train_mask_counts = np.zeros(
                    self._header.n_label_classes, dtype=np.float64
                )
                train_total_voxels = 0.0
                for i, sid in enumerate(vi_sids):
                    if sid in train_subjects:
                        s, e = int(vi_starts[i]), int(vi_ends[i])
                        class_counts = f["slices/class_voxel_counts"][s:e]  # (D, K)
                        train_mask_counts += class_counts.sum(axis=0).astype(np.float64)
                        h, w = self._header.in_plane_size
                        train_total_voxels += (e - s) * h * w

                neg_counts = train_total_voxels - train_mask_counts
                pos_counts = np.maximum(train_mask_counts, 1.0)
                pw = np.clip(neg_counts / pos_counts, 0.1, 50.0)
                self._pos_weight = torch.tensor(pw, dtype=torch.float32)

        # Build row indices for each partition (2D mode)
        self._build_indices(
            vi_sids,
            vi_starts,
            vi_ends,
            slice_has_lesion,
            slice_sids,
            train_subjects,
            val_subjects,
            test_subjects,
            sid_to_stratum,
        )

    def _build_indices(
        self,
        vi_sids: np.ndarray,
        vi_starts: np.ndarray,
        vi_ends: np.ndarray,
        slice_has_lesion: np.ndarray,
        slice_sids: np.ndarray,
        train_subjects: set[str],
        val_subjects: set[str],
        test_subjects: set[str],
        sid_to_stratum: dict[str, str],
    ) -> None:
        """Build row indices and per-sample metadata arrays."""
        train_rows, val_rows, test_rows = [], [], []
        train_sids_list, val_sids_list, test_sids_list = [], [], []
        train_strata_list, val_strata_list, test_strata_list = [], [], []

        for i, sid in enumerate(vi_sids):
            s, e = int(vi_starts[i]), int(vi_ends[i])
            stratum = sid_to_stratum.get(sid, "unknown")

            for row in range(s, e):
                if sid in train_subjects:
                    if self.lesion_only and not slice_has_lesion[row]:
                        continue
                    train_rows.append(row)
                    train_sids_list.append(sid)
                    train_strata_list.append(stratum)
                elif sid in val_subjects:
                    val_rows.append(row)
                    val_sids_list.append(sid)
                    val_strata_list.append(stratum)
                elif sid in test_subjects:
                    test_rows.append(row)
                    test_sids_list.append(sid)
                    test_strata_list.append(stratum)

        self._train_indices = np.array(train_rows, dtype=np.int64)
        self._val_indices = np.array(val_rows, dtype=np.int64)
        self._test_indices = np.array(test_rows, dtype=np.int64)
        self._train_sids = np.array(train_sids_list, dtype=object)
        self._val_sids = np.array(val_sids_list, dtype=object)
        self._test_sids = np.array(test_sids_list, dtype=object)
        self._train_strata = np.array(train_strata_list, dtype=object)
        self._val_strata = np.array(val_strata_list, dtype=object)
        self._test_strata = np.array(test_strata_list, dtype=object)

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        assert self._train_indices is not None
        transform = (
            get_train_transforms_2d() if self.augment else get_val_transforms_2d()
        )
        ds = _SliceDataset(
            self.h5_path,
            self._train_indices,
            transform=transform,
            with_metadata=False,
        )
        g = torch.Generator()
        g.manual_seed(42)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=_pin_memory_available(),
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=g,
            multiprocessing_context=_safe_mp_context(self.num_workers),
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        assert self._val_indices is not None
        ds = _SliceDataset(
            self.h5_path,
            self._val_indices,
            transform=get_val_transforms_2d(),
            with_metadata=True,
            subject_ids=self._val_sids,
            strata=self._val_strata,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=_pin_memory_available(),
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker,
            multiprocessing_context=_safe_mp_context(self.num_workers),
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        assert self._test_indices is not None
        ds = _SliceDataset(
            self.h5_path,
            self._test_indices,
            transform=get_val_transforms_2d(),
            with_metadata=True,
            subject_ids=self._test_sids,
            strata=self._test_strata,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=_pin_memory_available(),
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker,
            multiprocessing_context=_safe_mp_context(self.num_workers),
        )


# ---------------------------------------------------------------------------
# Mock DataModule for integration tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockDataConfig:
    """Configuration for :class:`MockAtlasDataModule`.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    n_test : int
        Number of test samples.
    img_size : tuple[int, int]
        Spatial dimensions ``(H, W)``.
    n_subjects : int
        Number of mock subjects.
    seed : int
        Random seed.
    batch_size : int
        Batch size.
    """

    n_train: int = 16
    n_val: int = 8
    n_test: int = 4
    img_size: tuple[int, int] = (224, 224)
    n_subjects: int = 4
    seed: int = 0
    batch_size: int = 2


class _MockDataset(Dataset):  # type: ignore[type-arg]
    """In-memory random dataset for testing."""

    def __init__(
        self,
        n_samples: int,
        img_size: tuple[int, int],
        seed: int,
        with_metadata: bool = False,
        n_subjects: int = 4,
    ) -> None:
        self.n_samples = n_samples
        self.with_metadata = with_metadata
        rng = torch.Generator().manual_seed(seed)
        h, w = img_size

        self.images = torch.randn(n_samples, 1, h, w, generator=rng)
        self.masks = torch.zeros(n_samples, 1, h, w)

        # Place small lesion-like blobs in each sample
        for i in range(n_samples):
            cy = int(torch.randint(h // 4, 3 * h // 4, (1,), generator=rng).item())
            cx = int(torch.randint(w // 4, 3 * w // 4, (1,), generator=rng).item())
            r = int(torch.randint(3, 10, (1,), generator=rng).item())
            y0 = max(0, cy - r)
            y1 = min(h, cy + r)
            x0 = max(0, cx - r)
            x1 = min(w, cx + r)
            self.masks[i, 0, y0:y1, x0:x1] = 1.0

        self.subject_ids = [f"mock_{i % n_subjects:03d}" for i in range(n_samples)]
        strata = ["small", "medium", "large"]
        self.strata = [strata[i % len(strata)] for i in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "image": self.images[idx],
            "mask": self.masks[idx],
        }
        if self.with_metadata:
            sample["subject_id"] = self.subject_ids[idx]
            sample["volume_stratum"] = self.strata[idx]
        return sample


class MockAtlasDataModule(pl.LightningDataModule):
    """Synthetic in-memory DataModule for integration testing.

    Generates random float32 images ``(1, H, W)`` and binary masks
    ``(1, H, W)`` with small nonzero regions.  Uses 1 modality channel
    and 1 label class to match existing test fixtures.

    Parameters
    ----------
    cfg : MockDataConfig
        Mock configuration.
    """

    def __init__(self, cfg: MockDataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._train_ds: _MockDataset | None = None
        self._val_ds: _MockDataset | None = None
        self._test_ds: _MockDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        c = self.cfg
        self._train_ds = _MockDataset(
            c.n_train, c.img_size, c.seed, with_metadata=False, n_subjects=c.n_subjects
        )
        self._val_ds = _MockDataset(
            c.n_val, c.img_size, c.seed + 1, with_metadata=True, n_subjects=c.n_subjects
        )
        self._test_ds = _MockDataset(
            c.n_test,
            c.img_size,
            c.seed + 2,
            with_metadata=True,
            n_subjects=c.n_subjects,
        )

    @property
    def split_hash(self) -> str:
        return hashlib.sha256(f"mock-seed={self.cfg.seed}".encode()).hexdigest()[:16]

    @property
    def n_train(self) -> int:
        return self.cfg.n_train

    @property
    def n_val(self) -> int:
        return self.cfg.n_val

    @property
    def n_test(self) -> int:
        return self.cfg.n_test

    @property
    def pos_weight(self) -> Tensor:
        return torch.tensor(10.0)

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        assert self._test_ds is not None
        return DataLoader(
            self._test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )
