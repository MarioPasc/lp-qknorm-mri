"""Unit tests for data/splits.py — patient-level stratified k-fold."""

from __future__ import annotations

import numpy as np
import pytest

from lpqknorm.data.splits import FoldSpec, make_patient_kfold
from lpqknorm.utils.exceptions import StratificationError


def _make_test_data(
    n_subjects: int = 20,
    n_multi_session_patients: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic subject/patient/strata arrays."""
    subject_ids = []
    patient_ids = []
    strata_list = []
    strata_options = ["small", "medium", "large"]

    pid_counter = 0
    i = 0

    # Single-session patients
    n_single = n_subjects - n_multi_session_patients * 2
    for _ in range(n_single):
        sid = f"subj_{i:03d}"
        pid = f"pat_{pid_counter:03d}"
        subject_ids.append(sid)
        patient_ids.append(pid)
        strata_list.append(strata_options[pid_counter % 3])
        pid_counter += 1
        i += 1

    # Multi-session patients (2 sessions each)
    for _ in range(n_multi_session_patients):
        pid = f"pat_{pid_counter:03d}"
        for _sess in range(2):
            sid = f"subj_{i:03d}"
            subject_ids.append(sid)
            patient_ids.append(pid)
            strata_list.append(strata_options[pid_counter % 3])
            i += 1
        pid_counter += 1

    return (
        np.array(subject_ids),
        np.array(patient_ids),
        np.array(strata_list),
    )


class TestNoPatientLeakage:
    """AT7: No patient ID appears in more than one partition."""

    def test_no_leakage_basic(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=30)
        folds, _ = make_patient_kfold(sids, pids, strata, n_folds=3, seed=42)

        for fold in folds:
            all_sids = fold.train_subjects + fold.val_subjects + fold.test_subjects
            # All subjects accounted for
            assert set(all_sids) == set(sids.tolist())
            # No overlap between partitions
            assert set(fold.train_subjects) & set(fold.val_subjects) == set()
            assert set(fold.train_subjects) & set(fold.test_subjects) == set()
            assert set(fold.val_subjects) & set(fold.test_subjects) == set()

    def test_no_leakage_patient_level(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=30)
        folds, _ = make_patient_kfold(sids, pids, strata, n_folds=3, seed=42)

        sid_to_pid = dict(zip(sids.tolist(), pids.tolist(), strict=True))

        for fold in folds:
            train_pids = {sid_to_pid[s] for s in fold.train_subjects}
            val_pids = {sid_to_pid[s] for s in fold.val_subjects}
            test_pids = {sid_to_pid[s] for s in fold.test_subjects}
            assert train_pids & val_pids == set()
            assert train_pids & test_pids == set()
            assert val_pids & test_pids == set()

    def test_multi_session_no_leakage(self) -> None:
        """Multi-session patients: all sessions in the same fold."""
        sids, pids, strata = _make_test_data(
            n_subjects=24,
            n_multi_session_patients=3,
        )
        folds, _ = make_patient_kfold(sids, pids, strata, n_folds=3, seed=42)

        sid_to_pid = dict(zip(sids.tolist(), pids.tolist(), strict=True))

        for fold in folds:
            train_pids = {sid_to_pid[s] for s in fold.train_subjects}
            val_pids = {sid_to_pid[s] for s in fold.val_subjects}
            test_pids = {sid_to_pid[s] for s in fold.test_subjects}
            assert train_pids & test_pids == set()
            assert train_pids & val_pids == set()
            assert val_pids & test_pids == set()


class TestDeterminism:
    """AT8: Same seed → identical fold assignments."""

    def test_determinism(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=30)
        folds1, hash1 = make_patient_kfold(sids, pids, strata, seed=123)
        folds2, hash2 = make_patient_kfold(sids, pids, strata, seed=123)

        assert hash1 == hash2
        for f1, f2 in zip(folds1, folds2, strict=True):
            assert f1.train_subjects == f2.train_subjects
            assert f1.val_subjects == f2.val_subjects
            assert f1.test_subjects == f2.test_subjects

    def test_different_seeds_differ(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=30)
        _, hash1 = make_patient_kfold(sids, pids, strata, seed=1)
        _, hash2 = make_patient_kfold(sids, pids, strata, seed=2)
        assert hash1 != hash2


class TestStratumBalance:
    """AT9: Each fold's train partition has at least one patient per stratum."""

    def test_stratum_balance(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=30)
        folds, _ = make_patient_kfold(sids, pids, strata, n_folds=3, seed=42)

        sid_to_stratum = dict(zip(sids.tolist(), strata.tolist(), strict=True))

        for fold in folds:
            train_strata = {sid_to_stratum[s] for s in fold.train_subjects}
            assert "small" in train_strata
            assert "medium" in train_strata
            assert "large" in train_strata


class TestFoldSpecDataclass:
    """Basic FoldSpec tests."""

    def test_fold_spec_frozen(self) -> None:
        fold = FoldSpec(
            fold_idx=0, train_subjects=["a"], val_subjects=["b"], test_subjects=["c"]
        )
        with pytest.raises(AttributeError):
            fold.fold_idx = 1  # type: ignore[misc]

    def test_split_hash_format(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=15)
        _, split_hash = make_patient_kfold(sids, pids, strata, n_folds=3, seed=42)
        assert len(split_hash) == 16
        assert all(c in "0123456789abcdef" for c in split_hash)


class TestCommonTestHoldout:
    """Fixed-test-patient scheme: common test set across all folds."""

    def test_common_test_across_folds(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=60)
        folds, _ = make_patient_kfold(
            sids,
            pids,
            strata,
            n_folds=3,
            fixed_test_patients=15,
            seed=42,
        )
        assert len(folds) == 3
        # Test set byte-identical across every fold.
        ref_test = folds[0].test_subjects
        for f in folds[1:]:
            assert f.test_subjects == ref_test

    def test_exact_test_patient_count(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=60)
        folds, _ = make_patient_kfold(
            sids,
            pids,
            strata,
            n_folds=3,
            fixed_test_patients=15,
            seed=42,
        )
        sid_to_pid = dict(zip(sids.tolist(), pids.tolist(), strict=True))
        test_pids = {sid_to_pid[s] for s in folds[0].test_subjects}
        assert len(test_pids) == 15

    def test_no_leakage_common_test(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=60, n_multi_session_patients=5)
        folds, _ = make_patient_kfold(
            sids,
            pids,
            strata,
            n_folds=3,
            fixed_test_patients=10,
            seed=42,
        )
        sid_to_pid = dict(zip(sids.tolist(), pids.tolist(), strict=True))
        for fold in folds:
            train_pids = {sid_to_pid[s] for s in fold.train_subjects}
            val_pids = {sid_to_pid[s] for s in fold.val_subjects}
            test_pids = {sid_to_pid[s] for s in fold.test_subjects}
            assert train_pids & val_pids == set()
            assert train_pids & test_pids == set()
            assert val_pids & test_pids == set()

    def test_train_val_rotates(self) -> None:
        """Across folds, train/val partitions must differ (rotation)."""
        sids, pids, strata = _make_test_data(n_subjects=60)
        folds, _ = make_patient_kfold(
            sids,
            pids,
            strata,
            n_folds=3,
            fixed_test_patients=12,
            seed=42,
        )
        val_sets = [frozenset(f.val_subjects) for f in folds]
        # At least two folds must differ in their val set.
        assert len(set(val_sets)) > 1

    def test_determinism_common_test(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=60)
        folds1, h1 = make_patient_kfold(
            sids,
            pids,
            strata,
            n_folds=3,
            fixed_test_patients=12,
            seed=7,
        )
        folds2, h2 = make_patient_kfold(
            sids,
            pids,
            strata,
            n_folds=3,
            fixed_test_patients=12,
            seed=7,
        )
        assert h1 == h2
        for f1, f2 in zip(folds1, folds2, strict=True):
            assert f1.train_subjects == f2.train_subjects
            assert f1.val_subjects == f2.val_subjects
            assert f1.test_subjects == f2.test_subjects

    def test_stratified_allocation(self) -> None:
        """Common test holdout should draw from every stratum."""
        sids, pids, strata = _make_test_data(n_subjects=60)
        folds, _ = make_patient_kfold(
            sids,
            pids,
            strata,
            n_folds=3,
            fixed_test_patients=15,
            seed=42,
        )
        sid_to_stratum = dict(zip(sids.tolist(), strata.tolist(), strict=True))
        test_strata = {sid_to_stratum[s] for s in folds[0].test_subjects}
        assert test_strata == {"small", "medium", "large"}

    def test_rejects_oversized_holdout(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=30)
        with pytest.raises(StratificationError):
            make_patient_kfold(
                sids,
                pids,
                strata,
                n_folds=3,
                fixed_test_patients=999,
                seed=42,
            )

    def test_rejects_nonpositive_holdout(self) -> None:
        sids, pids, strata = _make_test_data(n_subjects=30)
        with pytest.raises(ValueError, match="fixed_test_patients"):
            make_patient_kfold(
                sids,
                pids,
                strata,
                n_folds=3,
                fixed_test_patients=-5,
                seed=42,
            )
