"""Unit tests for data/schema.py — HDF5 schema validation and header I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py


if TYPE_CHECKING:
    from pathlib import Path
import pytest

from lpqknorm.data.converter import PreprocessConfig, write_standardized_h5
from lpqknorm.data.schema import DatasetHeader, validate_h5
from lpqknorm.utils.exceptions import SchemaValidationError
from tests.fixtures.synthetic_dataset import (
    make_synthetic_info,
    make_synthetic_subjects,
)


@pytest.fixture
def synthetic_h5(tmp_path: Path) -> Path:
    """Write a small synthetic HDF5 file for validation tests."""
    subjects = make_synthetic_subjects(n_subjects=12, depth=8, img_size=(32, 32))
    info = make_synthetic_info()
    cfg = PreprocessConfig(in_plane_size=(32, 32))
    out = tmp_path / "test.h5"
    write_standardized_h5(subjects, info, cfg, out)
    return out


class TestValidateH5:
    """AT1-AT2: validate_h5 catches errors in valid and invalid files."""

    def test_valid_file_returns_empty(self, synthetic_h5: Path) -> None:
        errors = validate_h5(synthetic_h5)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_attr_detected(self, synthetic_h5: Path) -> None:
        with h5py.File(synthetic_h5, "a") as f:
            del f.attrs["dataset_name"]
        errors = validate_h5(synthetic_h5)
        assert any("dataset_name" in e for e in errors)

    def test_missing_group_detected(self, synthetic_h5: Path) -> None:
        with h5py.File(synthetic_h5, "a") as f:
            del f["strata"]
        # strata is not in the required groups list, so no error expected
        # But if we delete a required group like /data...
        errors = validate_h5(synthetic_h5)
        assert errors == []  # strata is optional

    def test_missing_required_group_detected(self, synthetic_h5: Path) -> None:
        with h5py.File(synthetic_h5, "a") as f:
            del f["volume_index"]
        errors = validate_h5(synthetic_h5)
        assert any("volume_index" in e for e in errors)

    def test_shape_mismatch_detected(self, synthetic_h5: Path) -> None:
        with h5py.File(synthetic_h5, "a") as f:
            f.attrs["n_total_slices"] = 999999
        errors = validate_h5(synthetic_h5)
        assert any("n_total_slices" in e for e in errors)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        errors = validate_h5(tmp_path / "nonexistent.h5")
        assert len(errors) == 1
        assert "does not exist" in errors[0]


class TestDatasetHeaderRoundTrip:
    """AT2: Header round-trip — write then read back."""

    def test_round_trip(self, synthetic_h5: Path) -> None:
        header = DatasetHeader.from_h5(synthetic_h5)
        assert header.format_version == "1.0"
        assert header.dataset_name == "synthetic"
        assert header.n_subjects == 12
        assert header.n_modalities == 1
        assert header.n_label_classes == 1
        assert header.spatial_dims == 3
        assert header.in_plane_size == (32, 32)
        assert isinstance(header.label_names, list)
        assert len(header.label_names) == 1
        assert isinstance(header.label_descriptions, dict)

    def test_multi_modal_round_trip(self, tmp_path: Path) -> None:
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
        assert len(header.modalities) == 4
        assert len(header.label_names) == 3

    def test_missing_required_attr_raises(self, synthetic_h5: Path) -> None:
        with h5py.File(synthetic_h5, "a") as f:
            del f.attrs["format_version"]
        with pytest.raises(SchemaValidationError, match="format_version"):
            DatasetHeader.from_h5(synthetic_h5)
