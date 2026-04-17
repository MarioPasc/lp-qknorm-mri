"""Patient-level stratified k-fold cross-validation splits.

Two split schemes are supported:

1. **Rotating test set** (``fixed_test_patients=None``, default): the test
   partition rotates across folds via :class:`StratifiedGroupKFold`.  Kept for
   backwards compatibility with older experiments.
2. **Common test holdout** (``fixed_test_patients > 0``): a single
   stratified patient-level test set of ``fixed_test_patients`` patients is
   extracted first and shared by every fold.  The remaining patients are
   distributed into ``n_folds`` train/val partitions by
   :class:`StratifiedGroupKFold`.  This scheme matches the Lp-QKNorm
   primary-sweep design, where all ``(p, fold)`` runs are evaluated on an
   identical held-out test set.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from lpqknorm.utils.exceptions import SplitLeakageError, StratificationError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldSpec:
    """One fold of a k-fold cross-validation split.

    Parameters
    ----------
    fold_idx : int
        Zero-based fold index.
    train_subjects : list[str]
        Subject IDs in the training partition.
    val_subjects : list[str]
        Subject IDs in the validation partition.
    test_subjects : list[str]
        Subject IDs in the test (held-out) partition.  When
        ``fixed_test_patients`` is used, this list is identical across folds.
    """

    fold_idx: int
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def make_patient_kfold(
    subject_ids: np.ndarray,
    patient_ids: np.ndarray,
    strata: np.ndarray,
    n_folds: int = 3,
    val_fraction: float = 0.15,
    fixed_test_patients: int | None = None,
    seed: int = 20260216,
) -> tuple[list[FoldSpec], str]:
    """Create patient-level stratified k-fold splits.

    Parameters
    ----------
    subject_ids : np.ndarray
        Shape ``(S,)`` array of unique subject/session identifiers.
    patient_ids : np.ndarray
        Shape ``(S,)`` array of patient identifiers used for grouping.
        Multiple sessions from the same patient share the same patient ID.
    strata : np.ndarray
        Shape ``(S,)`` array of stratum labels (e.g. ``"small"``).
    n_folds : int
        Number of cross-validation folds.
    val_fraction : float
        Fraction of the trainval partition reserved for validation.  Only
        used when ``fixed_test_patients`` is ``None`` (rotating test set
        scheme).
    fixed_test_patients : int or None
        If set, number of **patients** to hold out as a common test set.
        The test set is then identical across all folds.  If ``None``, the
        test partition rotates across folds via
        :class:`StratifiedGroupKFold`.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    folds : list[FoldSpec]
        One ``FoldSpec`` per fold.
    split_hash : str
        First 16 hex characters of the SHA-256 hash of the canonical split
        assignment JSON.

    Raises
    ------
    SplitLeakageError
        If any patient appears in more than one partition within a fold.
    StratificationError
        If the stratified holdout cannot satisfy ``fixed_test_patients``.
    """
    subject_ids = np.asarray(subject_ids)
    patient_ids = np.asarray(patient_ids)
    strata = np.asarray(strata)

    if not (len(subject_ids) == len(patient_ids) == len(strata)):
        raise ValueError(
            "subject_ids, patient_ids, strata must share the same length: "
            f"got {len(subject_ids)}, {len(patient_ids)}, {len(strata)}"
        )

    # Encode strata as integers for sklearn
    unique_strata = np.unique(strata)
    strata_int = np.searchsorted(unique_strata, strata)

    if fixed_test_patients is not None:
        return _make_common_test_kfold(
            subject_ids=subject_ids,
            patient_ids=patient_ids,
            strata_int=strata_int,
            n_folds=n_folds,
            n_test_patients=int(fixed_test_patients),
            seed=seed,
        )

    return _make_rotating_test_kfold(
        subject_ids=subject_ids,
        patient_ids=patient_ids,
        strata_int=strata_int,
        n_folds=n_folds,
        val_fraction=val_fraction,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Rotating test set (legacy scheme)
# ---------------------------------------------------------------------------


def _make_rotating_test_kfold(
    subject_ids: np.ndarray,
    patient_ids: np.ndarray,
    strata_int: np.ndarray,
    n_folds: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[FoldSpec], str]:
    """K-fold with a rotating test partition per fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds: list[FoldSpec] = []
    dummy_x = np.zeros(len(subject_ids))

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        sgkf.split(dummy_x, strata_int, groups=patient_ids)
    ):
        test_subs = sorted(subject_ids[test_idx].tolist())
        trainval_subs = subject_ids[trainval_idx]
        trainval_patients = patient_ids[trainval_idx]
        trainval_strata = strata_int[trainval_idx]

        train_idx_local, val_idx_local = _stratified_patient_split(
            trainval_patients,
            trainval_strata,
            val_fraction=val_fraction,
            seed=seed + fold_idx,
        )

        train_subs = sorted(trainval_subs[train_idx_local].tolist())
        val_subs = sorted(trainval_subs[val_idx_local].tolist())

        fold = FoldSpec(
            fold_idx=fold_idx,
            train_subjects=train_subs,
            val_subjects=val_subs,
            test_subjects=test_subs,
        )
        _check_no_leakage(fold, patient_ids, subject_ids)
        folds.append(fold)

    split_hash = _compute_split_hash(folds)

    logger.info(
        "Rotating-test scheme: %d folds, train=%d val=%d test=%d (fold 0)",
        n_folds,
        len(folds[0].train_subjects),
        len(folds[0].val_subjects),
        len(folds[0].test_subjects),
    )

    return folds, split_hash


# ---------------------------------------------------------------------------
# Common test holdout scheme
# ---------------------------------------------------------------------------


def _make_common_test_kfold(
    subject_ids: np.ndarray,
    patient_ids: np.ndarray,
    strata_int: np.ndarray,
    n_folds: int,
    n_test_patients: int,
    seed: int,
) -> tuple[list[FoldSpec], str]:
    """Extract a common stratified test holdout, then k-fold train/val."""
    unique_patients = np.unique(patient_ids)

    if n_test_patients <= 0:
        raise ValueError(f"fixed_test_patients must be > 0, got {n_test_patients}")
    if n_test_patients >= len(unique_patients):
        raise StratificationError(
            "fixed_test_patients leaves no patients for train/val",
            {
                "fixed_test_patients": n_test_patients,
                "n_total_patients": len(unique_patients),
            },
        )

    # --- Stratified patient-level holdout ---
    test_patients = _stratified_patient_holdout(
        patient_ids=patient_ids,
        strata_int=strata_int,
        n_test_patients=n_test_patients,
        seed=seed,
    )
    test_patient_set = set(test_patients.tolist())
    test_mask = np.array([pid in test_patient_set for pid in patient_ids])
    test_subs = sorted(subject_ids[test_mask].tolist())

    # --- K-fold train/val on remaining patients ---
    rem_idx = np.where(~test_mask)[0]
    rem_subs = subject_ids[rem_idx]
    rem_pids = patient_ids[rem_idx]
    rem_strata_int = strata_int[rem_idx]

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if len(np.unique(rem_pids)) < n_folds:
        raise StratificationError(
            "Fewer remaining patients than folds after test holdout",
            {
                "n_remaining_patients": len(np.unique(rem_pids)),
                "n_folds": n_folds,
            },
        )

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds: list[FoldSpec] = []
    dummy_x = np.zeros(len(rem_subs))

    for fold_idx, (train_idx, val_idx) in enumerate(
        sgkf.split(dummy_x, rem_strata_int, groups=rem_pids)
    ):
        train_subs = sorted(rem_subs[train_idx].tolist())
        val_subs = sorted(rem_subs[val_idx].tolist())

        fold = FoldSpec(
            fold_idx=fold_idx,
            train_subjects=train_subs,
            val_subjects=val_subs,
            test_subjects=test_subs,
        )
        _check_no_leakage(fold, patient_ids, subject_ids)
        folds.append(fold)

    # Consistency: test set must be byte-identical across folds
    for f in folds[1:]:
        if f.test_subjects != folds[0].test_subjects:
            raise SplitLeakageError(
                "Common test holdout violated: per-fold test sets differ",
                {"fold_idx": f.fold_idx},
            )

    split_hash = _compute_split_hash(folds)

    logger.info(
        "Common-test scheme: %d folds, %d test patients (%d test sessions), "
        "train=%d val=%d (fold 0)",
        n_folds,
        n_test_patients,
        len(test_subs),
        len(folds[0].train_subjects),
        len(folds[0].val_subjects),
    )

    return folds, split_hash


def _stratified_patient_holdout(
    patient_ids: np.ndarray,
    strata_int: np.ndarray,
    n_test_patients: int,
    seed: int,
) -> np.ndarray:
    """Sample ``n_test_patients`` distinct patients, balanced per stratum.

    The per-patient stratum is aggregated as the most common stratum across
    that patient's sessions.  Target counts per stratum are allocated
    proportionally to each stratum's patient count; residuals are assigned to
    the strata with the largest fractional remainder.  A deterministic
    per-stratum shuffle then picks the target number of patients.

    Returns
    -------
    np.ndarray
        Array of patient IDs held out as the common test set.
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)

    # Aggregate stratum per patient (majority stratum)
    patient_strata: list[int] = []
    for pid in unique_patients:
        mask = patient_ids == pid
        vals, counts = np.unique(strata_int[mask], return_counts=True)
        patient_strata.append(int(vals[np.argmax(counts)]))
    patient_strata_arr = np.array(patient_strata, dtype=np.int64)

    # --- Proportional allocation per stratum ---
    stratum_values, stratum_counts = np.unique(patient_strata_arr, return_counts=True)
    n_patients = len(unique_patients)
    raw = stratum_counts.astype(np.float64) * (n_test_patients / n_patients)
    base = np.floor(raw).astype(np.int64)
    residual = raw - base
    deficit = n_test_patients - int(base.sum())
    if deficit > 0:
        order = np.argsort(-residual, kind="stable")
        for i in order[:deficit]:
            base[i] += 1
    # Guard: allocations cannot exceed stratum size.
    allocation = np.minimum(base, stratum_counts)
    leftover = n_test_patients - int(allocation.sum())
    while leftover > 0:
        slack = stratum_counts - allocation
        if slack.sum() == 0:
            raise StratificationError(
                "Cannot allocate fixed_test_patients across strata",
                {
                    "requested": n_test_patients,
                    "available_per_stratum": slack.tolist(),
                },
            )
        # Assign one extra to the stratum with the largest slack.
        j = int(np.argmax(slack))
        allocation[j] += 1
        leftover -= 1

    # --- Sample per stratum ---
    test_patients: list[str] = []
    for s_val, n_take in zip(stratum_values, allocation, strict=True):
        stratum_mask = patient_strata_arr == s_val
        pool = unique_patients[stratum_mask]
        pool_perm = pool.copy()
        rng.shuffle(pool_perm)
        test_patients.extend(pool_perm[: int(n_take)].tolist())

    result = np.array(sorted(test_patients))
    if len(result) != n_test_patients:
        raise StratificationError(
            "Stratified holdout produced wrong patient count",
            {"expected": n_test_patients, "actual": len(result)},
        )
    return result


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _stratified_patient_split(
    patient_ids: np.ndarray,
    strata_int: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sessions into train/val at the patient level."""
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)

    patient_strata: dict[str, int] = {}
    for pid in unique_patients:
        mask = patient_ids == pid
        vals, counts = np.unique(strata_int[mask], return_counts=True)
        patient_strata[pid] = int(vals[np.argmax(counts)])

    patient_list = list(unique_patients)
    patient_strata_arr = np.array([patient_strata[p] for p in patient_list])

    val_patients: set[str] = set()
    for stratum_val in np.unique(patient_strata_arr):
        stratum_mask = patient_strata_arr == stratum_val
        stratum_patients = np.array(patient_list)[stratum_mask]
        rng.shuffle(stratum_patients)
        n_take = max(1, int(len(stratum_patients) * val_fraction))
        val_patients.update(stratum_patients[:n_take].tolist())

    val_mask = np.array([pid in val_patients for pid in patient_ids])
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    return train_idx, val_idx


def _check_no_leakage(
    fold: FoldSpec,
    patient_ids: np.ndarray,
    subject_ids: np.ndarray,
) -> None:
    """Assert no patient ID appears in more than one partition."""
    sid_to_pid = dict(zip(subject_ids.tolist(), patient_ids.tolist(), strict=True))

    train_pids = {sid_to_pid[s] for s in fold.train_subjects}
    val_pids = {sid_to_pid[s] for s in fold.val_subjects}
    test_pids = {sid_to_pid[s] for s in fold.test_subjects}

    if train_pids & val_pids:
        raise SplitLeakageError(
            f"Fold {fold.fold_idx}: patients in both train and val",
            {"leaked": sorted(train_pids & val_pids)},
        )
    if train_pids & test_pids:
        raise SplitLeakageError(
            f"Fold {fold.fold_idx}: patients in both train and test",
            {"leaked": sorted(train_pids & test_pids)},
        )
    if val_pids & test_pids:
        raise SplitLeakageError(
            f"Fold {fold.fold_idx}: patients in both val and test",
            {"leaked": sorted(val_pids & test_pids)},
        )


def _compute_split_hash(folds: list[FoldSpec]) -> str:
    """Compute a deterministic hash of the split assignment."""
    canonical = json.dumps(
        [
            {
                "fold": f.fold_idx,
                "train": sorted(f.train_subjects),
                "val": sorted(f.val_subjects),
                "test": sorted(f.test_subjects),
            }
            for f in folds
        ],
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
