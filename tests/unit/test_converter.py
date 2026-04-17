"""Unit tests for the converter protocol and BraTS-MEN converter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py


if TYPE_CHECKING:
    from pathlib import Path
import numpy as np
import pytest

from lpqknorm.data.converter import (
    DatasetConverter,
    PreprocessConfig,
    write_standardized_h5,
)
from lpqknorm.data.converters.brats_men import BraTSMenConverter, extract_patient_id
from lpqknorm.data.schema import DatasetHeader, validate_h5
from lpqknorm.utils.exceptions import DataIntegrityError
from tests.fixtures.synthetic_dataset import (
    make_synthetic_info,
    make_synthetic_subjects,
)


class TestBraTSMenConverterProtocol:
    """Verify BraTSMenConverter satisfies the DatasetConverter protocol."""

    def test_implements_protocol(self) -> None:
        converter = BraTSMenConverter()
        assert isinstance(converter, DatasetConverter)

    def test_info_fields(self) -> None:
        info = BraTSMenConverter().info
        assert info.name == "brats_men"
        assert info.task == "multi_class_segmentation"
        assert len(info.label_names) == 3
        assert len(info.modalities) == 4
        assert info.pathology == "meningioma"


class TestExtractPatientId:
    """Test patient ID extraction from BraTS-MEN subject IDs."""

    @pytest.mark.parametrize(
        ("subject_id", "expected"),
        [
            ("BraTS-MEN-00004-000", "00004"),
            ("BraTS-MEN-01435-001", "01435"),
            ("BraTS-MEN-00717-013", "00717"),
        ],
    )
    def test_extraction(self, subject_id: str, expected: str) -> None:
        assert extract_patient_id(subject_id) == expected


class TestDiscoverSubjects:
    """AT10: discover_subjects on a synthetic directory tree."""

    def _create_fake_brats_tree(self, root: Path, n_subjects: int = 5) -> list[str]:
        """Create fake BraTS-MEN directory structure with empty files."""
        sids = []
        for i in range(n_subjects):
            sid = f"BraTS-MEN-{i:05d}-000"
            sids.append(sid)
            subdir = root / sid
            subdir.mkdir()
            for mod in ("t1n", "t1c", "t2w", "t2f"):
                (subdir / f"{sid}-{mod}.nii.gz").touch()
            (subdir / f"{sid}-seg.nii.gz").touch()
        return sids

    def test_discover_on_synthetic_tree(self, tmp_path: Path) -> None:
        """Create 950 fake sessions and verify discovery."""
        expected_sids = self._create_fake_brats_tree(tmp_path, n_subjects=950)

        converter = BraTSMenConverter()
        records = converter.discover_subjects(tmp_path)

        assert len(records) == 950
        discovered_sids = [r.subject_id for r in records]
        assert set(discovered_sids) == set(expected_sids)

    def test_skip_non_matching_dirs(self, tmp_path: Path) -> None:
        """Non-matching entries are skipped without error."""
        self._create_fake_brats_tree(tmp_path, n_subjects=950)
        # Add non-matching entries
        (tmp_path / ".semantic_cache").mkdir()
        (tmp_path / "supplementary.xlsx").touch()
        (tmp_path / "README.md").touch()

        converter = BraTSMenConverter()
        records = converter.discover_subjects(tmp_path)
        assert len(records) == 950

    def test_too_few_subjects_raises(self, tmp_path: Path) -> None:
        self._create_fake_brats_tree(tmp_path, n_subjects=5)
        converter = BraTSMenConverter()
        with pytest.raises(DataIntegrityError, match="900"):
            converter.discover_subjects(tmp_path)

    def test_missing_modality_raises(self, tmp_path: Path) -> None:
        sid = "BraTS-MEN-00000-000"
        subdir = tmp_path / sid
        subdir.mkdir()
        # Only create 3 of 4 modalities
        for mod in ("t1n", "t1c", "t2w"):
            (subdir / f"{sid}-{mod}.nii.gz").touch()
        (subdir / f"{sid}-seg.nii.gz").touch()
        # Need >= 900 for the count check, so add 999 more valid ones
        for i in range(1, 1000):
            s = f"BraTS-MEN-{i:05d}-000"
            d = tmp_path / s
            d.mkdir()
            for m in ("t1n", "t1c", "t2w", "t2f"):
                (d / f"{s}-{m}.nii.gz").touch()
            (d / f"{s}-seg.nii.gz").touch()

        converter = BraTSMenConverter()
        with pytest.raises(DataIntegrityError, match="Missing modality"):
            converter.discover_subjects(tmp_path)


class TestMultiModalHDF5:
    """AT11: Multi-modal converter writes C=4 correctly."""

    def test_multi_modal_write(self, tmp_path: Path) -> None:
        subjects = make_synthetic_subjects(
            n_subjects=12,
            n_modalities=4,
            n_classes=3,
            depth=6,
            img_size=(32, 32),
        )
        info = make_synthetic_info(n_modalities=4, n_classes=3)
        cfg = PreprocessConfig(in_plane_size=(32, 32))
        out = tmp_path / "multi.h5"
        write_standardized_h5(subjects, info, cfg, out)

        header = DatasetHeader.from_h5(out)
        assert header.n_modalities == 4
        assert header.n_label_classes == 3

        with h5py.File(out, "r") as f:
            assert f["data/images"].shape[1] == 4
            assert f["data/masks"].shape[1] == 3

        errors = validate_h5(out)
        assert errors == []


class TestVolumeIndexing:
    """AT3-AT4: Volume reconstruction and slice access from flat layout."""

    def test_volume_reconstruction(self, tmp_path: Path) -> None:
        subjects = make_synthetic_subjects(
            n_subjects=12,
            depth=10,
            img_size=(32, 32),
            variable_depth=True,
        )

        info = make_synthetic_info()
        cfg = PreprocessConfig(in_plane_size=(32, 32))
        out = tmp_path / "test.h5"
        write_standardized_h5(subjects, info, cfg, out)

        with h5py.File(out, "r") as f:
            vi_starts = f["volume_index/start_row"][()]
            vi_ends = f["volume_index/end_row"][()]
            vi_depths = f["volume_index/depth"][()]

            for i, subj in enumerate(subjects):
                s, e = int(vi_starts[i]), int(vi_ends[i])
                assert e - s == vi_depths[i]
                assert e - s == subj.images.shape[0]

                reconstructed = f["data/images"][s:e]
                np.testing.assert_array_equal(reconstructed, subj.images)

    def test_slice_access_cross_check(self, tmp_path: Path) -> None:
        subjects = make_synthetic_subjects(n_subjects=12, depth=8, img_size=(32, 32))
        info = make_synthetic_info()
        cfg = PreprocessConfig(in_plane_size=(32, 32))
        out = tmp_path / "test.h5"
        write_standardized_h5(subjects, info, cfg, out)

        with h5py.File(out, "r") as f:
            slice_sids = [
                s.decode() if isinstance(s, bytes) else s
                for s in f["slices/subject_id"][()]
            ]
            slice_depth_idxs = f["slices/depth_idx"][()]
            vi_sids = [
                s.decode() if isinstance(s, bytes) else s
                for s in f["volume_index/subject_id"][()]
            ]
            vi_starts = f["volume_index/start_row"][()]

            # Check a few random rows
            for row in [0, 5, 15]:
                if row >= len(slice_sids):
                    break
                sid = slice_sids[row]
                z = slice_depth_idxs[row]
                subj_idx = vi_sids.index(sid)
                expected_row = int(vi_starts[subj_idx]) + z
                assert row == expected_row


class TestSliceManifestConsistency:
    """AT5: Slice manifest length matches data row count."""

    def test_consistency(self, tmp_path: Path) -> None:
        subjects = make_synthetic_subjects(n_subjects=12, depth=10, img_size=(32, 32))
        info = make_synthetic_info()
        cfg = PreprocessConfig(in_plane_size=(32, 32))
        out = tmp_path / "test.h5"
        write_standardized_h5(subjects, info, cfg, out)

        with h5py.File(out, "r") as f:
            n_data = f["data/images"].shape[0]
            n_slices = f["slices/subject_id"].shape[0]
            assert n_data == n_slices

            has_lesion = f["slices/has_lesion"][()]
            voxel_counts = f["slices/lesion_voxel_count"][()]

            for i in range(n_data):
                if has_lesion[i]:
                    assert voxel_counts[i] >= cfg.min_lesion_voxels_per_slice


class TestSortedIndices:
    """AT6: rank_by_lesion_volume is a valid permutation."""

    def test_rank_is_valid_permutation(self, tmp_path: Path) -> None:
        subjects = make_synthetic_subjects(n_subjects=12, depth=10, img_size=(32, 32))
        info = make_synthetic_info()
        cfg = PreprocessConfig(in_plane_size=(32, 32))
        out = tmp_path / "test.h5"
        write_standardized_h5(subjects, info, cfg, out)

        with h5py.File(out, "r") as f:
            rank = f["subjects/rank_by_lesion_volume"][()]
            volumes = f["subjects/total_lesion_volume_mm3"][()]

        assert sorted(rank.tolist()) == list(range(len(rank)))
        # Verify monotonicity: volumes[argsort(rank)] should be non-decreasing
        # rank[i] = position of subject i in sorted order
        sorted_volumes = volumes[np.argsort(rank)]
        assert np.all(sorted_volumes[:-1] <= sorted_volumes[1:])
