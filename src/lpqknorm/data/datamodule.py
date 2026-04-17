"""DataModule for ATLAS v2.0 lesion-bearing 2D slices (stub + mock).

Phase 1 is not yet implemented.  This file provides:

1. :class:`AtlasSliceDataModule` — stub with the Phase 1 API contract.
   Raises ``NotImplementedError`` unless a real ``atlas_2d.h5`` and
   ``manifest.parquet`` are available.
2. :class:`MockAtlasDataModule` — fully functional mock for testing.
   Generates synthetic ``(image, mask, metadata)`` batches without any
   real data file.

Both modules expose the same public attributes after ``setup("fit")``:
``pos_weight``, ``split_hash``, ``n_train``, ``n_val``, ``n_test``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lpqknorm.utils.seeding import seed_worker


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockDataConfig:
    """Configuration for :class:`MockAtlasDataModule`.

    Parameters
    ----------
    n_train : int
        Number of synthetic training samples.
    n_val : int
        Number of synthetic validation samples.
    n_test : int
        Number of synthetic test samples.
    img_size : tuple[int, int]
        Spatial size of images ``(H, W)``.
    n_subjects : int
        Number of distinct subject IDs (for cardinality tests).
    seed : int
        RNG seed for deterministic synthetic data.
    batch_size : int
        DataLoader batch size.
    """

    n_train: int = 16
    n_val: int = 8
    n_test: int = 8
    img_size: tuple[int, int] = (224, 224)
    n_subjects: int = 4
    seed: int = 42
    batch_size: int = 4


# ---------------------------------------------------------------------------
# Synthetic dataset for tests
# ---------------------------------------------------------------------------

_STRATA = ("small", "medium", "large")


class _SyntheticSliceDataset(Dataset[dict[str, Tensor | str | int]]):
    """Generate synthetic ``(image, mask, metadata)`` triples in memory.

    Each item returns a dict with keys:

    - ``"image"``: ``Tensor (1, H, W)`` float32
    - ``"mask"``:  ``Tensor (1, H, W)`` float32 binary, with at least one
      positive pixel per slice (lesion-only constraint).
    - ``"subject_id"``: ``str``
    - ``"volume_stratum"``: ``str`` — one of ``{"small", "medium", "large"}``
    - ``"slice_idx"``: ``int``
    """

    def __init__(
        self,
        n: int,
        img_size: tuple[int, int],
        n_subjects: int,
        seed: int,
    ) -> None:
        self._n = n
        h, w = img_size
        rng = np.random.default_rng(seed)

        # Pre-generate all data for determinism
        self._images = rng.standard_normal((n, 1, h, w)).astype(np.float32)

        # Create binary masks with at least one lesion pixel
        self._masks = np.zeros((n, 1, h, w), dtype=np.float32)
        for i in range(n):
            r0 = int(rng.integers(h // 4, 3 * h // 4))
            c0 = int(rng.integers(w // 4, 3 * w // 4))
            r1 = min(r0 + int(rng.integers(4, 20)), h)
            c1 = min(c0 + int(rng.integers(4, 20)), w)
            self._masks[i, 0, r0:r1, c0:c1] = 1.0

        self._subject_ids = [f"sub-{(i % n_subjects):04d}" for i in range(n)]
        self._strata = [_STRATA[i % len(_STRATA)] for i in range(n)]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Tensor | str | int]:
        return {
            "image": torch.from_numpy(self._images[idx]),
            "mask": torch.from_numpy(self._masks[idx]),
            "subject_id": self._subject_ids[idx],
            "volume_stratum": self._strata[idx],
            "slice_idx": idx,
        }


# ---------------------------------------------------------------------------
# Mock DataModule
# ---------------------------------------------------------------------------


class MockAtlasDataModule(pl.LightningDataModule):
    """Fully synthetic DataModule for testing Phase 3 without real ATLAS data.

    Produces ``(image, mask, metadata)`` batches that satisfy the Phase 3
    batch contract: ``image (B, 1, 224, 224)``, ``mask (B, 1, 224, 224)``,
    plus metadata fields ``subject_id``, ``volume_stratum``, ``slice_idx``.

    Parameters
    ----------
    cfg : MockDataConfig
        Configuration for the synthetic data.

    Attributes
    ----------
    pos_weight : Tensor
        Computed from synthetic masks (ratio of negative to positive
        pixels).  Available after ``setup("fit")`` is called.
    split_hash : str
        SHA256 hash of a synthetic splits descriptor (stable across calls
        with the same config).
    n_train : int
        Number of training samples.
    n_val : int
        Number of validation samples.
    n_test : int
        Number of test samples.
    """

    def __init__(self, cfg: MockDataConfig | None = None) -> None:
        super().__init__()
        self._cfg = cfg if cfg is not None else MockDataConfig()
        self.pos_weight: Tensor | None = None
        self.split_hash: str = ""
        self.n_train: int = self._cfg.n_train
        self.n_val: int = self._cfg.n_val
        self.n_test: int = self._cfg.n_test
        self._train_ds: _SyntheticSliceDataset | None = None
        self._val_ds: _SyntheticSliceDataset | None = None
        self._test_ds: _SyntheticSliceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create synthetic datasets and compute ``pos_weight``."""
        cfg = self._cfg
        self._train_ds = _SyntheticSliceDataset(
            cfg.n_train, cfg.img_size, cfg.n_subjects, cfg.seed
        )
        self._val_ds = _SyntheticSliceDataset(
            cfg.n_val, cfg.img_size, cfg.n_subjects, cfg.seed + 1
        )
        self._test_ds = _SyntheticSliceDataset(
            cfg.n_test, cfg.img_size, cfg.n_subjects, cfg.seed + 2
        )

        # Compute pos_weight from training masks
        total_pos = float(self._train_ds._masks.sum())
        total_pix = float(np.prod(self._train_ds._masks.shape))
        total_neg = total_pix - total_pos
        self.pos_weight = torch.tensor(
            total_neg / max(total_pos, 1.0), dtype=torch.float32
        )

        # Stable hash for manifest test
        splits_repr = json.dumps(
            {
                "n_train": cfg.n_train,
                "n_val": cfg.n_val,
                "n_test": cfg.n_test,
                "seed": cfg.seed,
                "n_subjects": cfg.n_subjects,
            },
            sort_keys=True,
        )
        self.split_hash = hashlib.sha256(splits_repr.encode()).hexdigest()
        logger.info(
            "MockAtlasDataModule setup: pos_weight=%.3f, split_hash=%s",
            self.pos_weight.item(),
            self.split_hash[:8],
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor | str | int]]:
        """Return shuffled training DataLoader."""
        assert self._train_ds is not None, "Call setup() first."
        return DataLoader(
            self._train_ds,
            batch_size=self._cfg.batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            persistent_workers=False,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Tensor | str | int]]:
        """Return validation DataLoader (no shuffle)."""
        assert self._val_ds is not None, "Call setup() first."
        return DataLoader(
            self._val_ds,
            batch_size=self._cfg.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )

    def test_dataloader(self) -> DataLoader[dict[str, Tensor | str | int]]:
        """Return test DataLoader (no shuffle)."""
        assert self._test_ds is not None, "Call setup() first."
        return DataLoader(
            self._test_ds,
            batch_size=self._cfg.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
        )


# ---------------------------------------------------------------------------
# Real DataModule stub (Phase 1 placeholder)
# ---------------------------------------------------------------------------


class AtlasSliceDataModule(pl.LightningDataModule):
    """DataModule for ``atlas_2d.h5`` via ``manifest.parquet`` and split JSON.

    This is a **stub**.  Full implementation is in Phase 1.

    Parameters
    ----------
    cache_root : Path
        Directory containing ``atlas_2d.h5``, ``manifest.parquet``,
        ``strata.parquet``.
    fold : int
        Fold index (0, 1, or 2).
    batch_size : int
        Batch size for DataLoaders.
    num_workers : int
        DataLoader workers.
    augment : bool
        Whether to apply training augmentations.
    seed : int
        Seed for the DataLoader worker init.
    """

    def __init__(
        self,
        cache_root: Path,
        fold: int,
        batch_size: int = 16,
        num_workers: int = 8,
        augment: bool = True,
        seed: int = 20260216,
    ) -> None:
        super().__init__()
        self._cache_root = cache_root
        self._fold = fold
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._augment = augment
        self._seed = seed
        self.pos_weight: Tensor | None = None
        self.split_hash: str = ""
        self.n_train: int = 0
        self.n_val: int = 0
        self.n_test: int = 0

    def setup(self, stage: str | None = None) -> None:
        """Phase 1 placeholder — raises ``NotImplementedError``."""
        raise NotImplementedError(
            "AtlasSliceDataModule.setup() requires Phase 1 to be complete. "
            "Use MockAtlasDataModule for testing."
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor | str | int]]:
        """Phase 1 placeholder."""
        raise NotImplementedError("Phase 1 not implemented.")

    def val_dataloader(self) -> DataLoader[dict[str, Tensor | str | int]]:
        """Phase 1 placeholder."""
        raise NotImplementedError("Phase 1 not implemented.")

    def test_dataloader(self) -> DataLoader[dict[str, Tensor | str | int]]:
        """Phase 1 placeholder."""
        raise NotImplementedError("Phase 1 not implemented.")
