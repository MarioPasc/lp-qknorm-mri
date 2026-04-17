"""Volume-based lesion stratification into small / medium / large strata."""

from __future__ import annotations

import logging

import numpy as np

from lpqknorm.utils.exceptions import StratificationError


logger = logging.getLogger(__name__)

STRATUM_NAMES = ("small", "medium", "large")


def compute_strata(
    lesion_volumes: np.ndarray,
    n_strata: int = 3,
    method: str = "percentile",
) -> tuple[np.ndarray, np.ndarray]:
    """Assign subjects to lesion-volume strata using percentile boundaries.

    Parameters
    ----------
    lesion_volumes : np.ndarray
        Shape ``(S,)`` array of per-subject total lesion volumes in mm^3.
    n_strata : int
        Number of strata.  Default ``3``.
    method : str
        Stratification method.  Only ``"percentile"`` is supported.

    Returns
    -------
    strata_labels : np.ndarray
        Shape ``(S,)`` object array of stratum name strings.
    boundaries : np.ndarray
        Shape ``(n_strata - 1,)`` float64 array of boundary values in mm^3.

    Raises
    ------
    StratificationError
        If any stratum would be empty, or inputs are invalid.
    """
    if method != "percentile":
        raise StratificationError(f"Unsupported stratification method: {method}")

    if len(lesion_volumes) < n_strata:
        raise StratificationError(
            f"Need at least {n_strata} subjects for {n_strata} strata, "
            f"got {len(lesion_volumes)}"
        )

    vols = np.asarray(lesion_volumes, dtype=np.float64)

    percentiles = np.linspace(100.0 / n_strata, 100.0, n_strata + 1)[:-1]
    boundaries = np.percentile(vols, percentiles[:-1])

    names = (
        STRATUM_NAMES[:n_strata]
        if n_strata == 3
        else [f"stratum_{i}" for i in range(n_strata)]
    )

    # Check for degenerate boundaries (all identical volumes)
    unique_boundaries = np.unique(boundaries)
    if len(unique_boundaries) < len(boundaries):
        # Degenerate case: distribute subjects evenly across strata by rank
        sorted_idx = np.argsort(vols, kind="stable")
        labels = np.empty(len(vols), dtype=object)
        chunk_size = len(vols) // n_strata
        for i in range(n_strata):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_strata - 1 else len(vols)
            labels[sorted_idx[start:end]] = names[i]
    else:
        labels = np.empty(len(vols), dtype=object)
        for i in range(n_strata):
            if i == 0:
                mask = vols <= boundaries[0]
            elif i == n_strata - 1:
                mask = vols > boundaries[-1]
            else:
                mask = (vols > boundaries[i - 1]) & (vols <= boundaries[i])
            labels[mask] = names[i]

    # Validate no empty strata
    for name in names:
        count = int((labels == name).sum())
        if count == 0:
            raise StratificationError(
                f"Stratum '{name}' is empty",
                {"boundaries": boundaries.tolist()},
            )

    logger.info(
        "Stratification: %s",
        {name: int((labels == name).sum()) for name in names},
    )

    return labels, boundaries
