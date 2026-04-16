"""Unit tests for seeding utilities."""

from __future__ import annotations

import os

import torch

from lpqknorm.utils.seeding import set_global_seed


class TestSetGlobalSeed:
    """Tests for set_global_seed."""

    def test_reproducible_torch(self) -> None:
        """Two calls with the same seed produce identical torch tensors."""
        set_global_seed(42)
        t1 = torch.randn(5)
        set_global_seed(42)
        t2 = torch.randn(5)
        torch.testing.assert_close(t1, t2, atol=0.0, rtol=0.0)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different tensors."""
        set_global_seed(42)
        t1 = torch.randn(100)
        set_global_seed(99)
        t2 = torch.randn(100)
        assert not torch.allclose(t1, t2)

    def test_pythonhashseed_set(self) -> None:
        """PYTHONHASHSEED environment variable is set after seeding."""
        set_global_seed(123)
        assert os.environ["PYTHONHASHSEED"] == "123"

    def test_numpy_reproducible(self) -> None:
        """NumPy RNG is also deterministic after set_global_seed."""
        import numpy as np

        set_global_seed(7)
        a1 = np.random.rand(10)  # noqa: NPY002
        set_global_seed(7)
        a2 = np.random.rand(10)  # noqa: NPY002
        assert (a1 == a2).all()
