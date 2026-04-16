"""Global deterministic seeding utilities.

Provides :func:`set_global_seed` to ensure reproducible training across
Python, NumPy, and PyTorch RNG states, and :func:`seed_worker` as a
``worker_init_fn`` for ``DataLoader`` to set per-worker deterministic seeds.

References
----------
- PyTorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- Lightning ``seed_everything`` supplements these but does not set
  ``PYTHONHASHSEED`` or per-worker NumPy/random seeds.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch


logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set deterministic seed across Python, NumPy, Torch, and CUDA.

    Parameters
    ----------
    seed : int
        The integer seed value.

    Notes
    -----
    Sets ``os.environ["PYTHONHASHSEED"]``, ``random.seed``,
    ``np.random.seed``, ``torch.manual_seed``, and
    ``torch.cuda.manual_seed_all``.

    Does **not** call ``torch.use_deterministic_algorithms`` — that is the
    ``Trainer``'s responsibility (via ``deterministic=True``).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Global seed set to %d", seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that sets per-worker deterministic seeds.

    Parameters
    ----------
    worker_id : int
        Worker index provided by ``DataLoader``.

    Notes
    -----
    Uses ``torch.initial_seed()`` which is already set per-worker by
    Lightning / ``DataLoader`` based on the global seed + worker_id offset.
    This function propagates that seed to Python ``random`` and NumPy so
    that any augmentation code using those RNGs is also deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)  # noqa: NPY002
