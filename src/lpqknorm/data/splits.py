"""Patient-level stratified k-fold cross-validation splits."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from lpqknorm.utils.exceptions import SplitLeakageError


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
        Subject IDs in the test (held-out) partition.
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
        Fraction of the trainval partition reserved for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    folds : list[FoldSpec]
        One ``FoldSpec`` per fold.
    split_hash : str
        First 16 hex characters of the SHA-256 hash of the canonical
        split assignment JSON.

    Raises
    ------
    SplitLeakageError
        If any patient appears in more than one partition within a fold.
    """
    subject_ids = np.asarray(subject_ids)
    patient_ids = np.asarray(patient_ids)
    strata = np.asarray(strata)

    # Encode strata as integers for sklearn
    unique_strata = np.unique(strata)
    strata_int = np.searchsorted(unique_strata, strata)

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

        # Split trainval into train and val (patient-level)
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
        "Created %d folds: train=%d, val=%d, test=%d (fold 0)",
        n_folds,
        len(folds[0].train_subjects),
        len(folds[0].val_subjects),
        len(folds[0].test_subjects),
    )

    return folds, split_hash


def _stratified_patient_split(
    patient_ids: np.ndarray,
    strata_int: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sessions into train/val at the patient level."""
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)

    # Aggregate stratum per patient (take the most common stratum)
    patient_strata: dict[str, int] = {}
    for pid in unique_patients:
        mask = patient_ids == pid
        vals, counts = np.unique(strata_int[mask], return_counts=True)
        patient_strata[pid] = int(vals[np.argmax(counts)])

    # Stratified split at patient level
    patient_list = list(unique_patients)
    patient_strata_arr = np.array([patient_strata[p] for p in patient_list])

    n_val_patients = max(1, int(len(patient_list) * val_fraction))

    # Shuffle within each stratum, then take proportional val from each
    val_patients: set[str] = set()
    for stratum_val in np.unique(patient_strata_arr):
        stratum_mask = patient_strata_arr == stratum_val
        stratum_patients = np.array(patient_list)[stratum_mask]
        rng.shuffle(stratum_patients)
        n_take = max(1, int(len(stratum_patients) * val_fraction))
        val_patients.update(stratum_patients[:n_take].tolist())

    # Map back to session indices
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
    sid_to_pid = dict(zip(subject_ids.tolist(), patient_ids.tolist()))

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
