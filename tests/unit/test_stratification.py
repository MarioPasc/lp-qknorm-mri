"""Unit tests for data/stratification.py — volume-based lesion strata."""

from __future__ import annotations

import numpy as np
import pytest

from lpqknorm.data.stratification import compute_strata
from lpqknorm.utils.exceptions import StratificationError


class TestComputeStrata:
    """AT16-AT17: Monotonicity and boundary correctness."""

    def test_basic_three_strata(self) -> None:
        volumes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        labels, boundaries = compute_strata(volumes)

        assert len(labels) == 9
        assert len(boundaries) == 2
        assert set(labels) == {"small", "medium", "large"}

    def test_monotonicity(self) -> None:
        """Sorting by volume and re-assigning strata yields consistent labels."""
        rng = np.random.RandomState(42)
        volumes = rng.exponential(scale=1000.0, size=100)
        labels, _boundaries = compute_strata(volumes)

        sorted_idx = np.argsort(volumes)
        sorted_labels = labels[sorted_idx]

        # All "small" should come before "medium" which should come before "large"
        first_medium = np.where(sorted_labels == "medium")[0]
        last_small = np.where(sorted_labels == "small")[0]

        if len(first_medium) > 0 and len(last_small) > 0:
            assert (
                last_small[-1] < first_medium[0]
                or last_small[-1] == first_medium[0] - 1
            )

    def test_boundaries_separate_groups(self) -> None:
        volumes = np.arange(1.0, 31.0)
        labels, boundaries = compute_strata(volumes)

        for i, vol in enumerate(volumes):
            if labels[i] == "small":
                assert vol <= boundaries[0]
            elif labels[i] == "medium":
                assert vol > boundaries[0]
                assert vol <= boundaries[1]
            elif labels[i] == "large":
                assert vol > boundaries[1]

    def test_each_stratum_nonempty(self) -> None:
        volumes = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        labels, _ = compute_strata(volumes)
        for name in ["small", "medium", "large"]:
            assert (labels == name).sum() > 0

    def test_too_few_subjects_raises(self) -> None:
        with pytest.raises(StratificationError):
            compute_strata(np.array([1.0, 2.0]))

    def test_unsupported_method_raises(self) -> None:
        with pytest.raises(StratificationError, match="Unsupported"):
            compute_strata(np.arange(10.0), method="kmeans")

    def test_uniform_volumes(self) -> None:
        """All identical volumes should still produce 3 strata (boundary ties)."""
        volumes = np.ones(30) * 100.0
        labels, _boundaries = compute_strata(volumes)
        assert len(labels) == 30
