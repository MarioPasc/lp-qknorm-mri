"""HDF5 v1.0 schema definition and validation for standardized datasets.

Provides :class:`DatasetHeader` (frozen dataclass mirroring all root-level
HDF5 attributes) and :func:`validate_h5` (structural validator that returns
a list of error strings).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lpqknorm.utils.exceptions import SchemaValidationError


logger = logging.getLogger(__name__)

FORMAT_VERSION = "1.0"

_REQUIRED_GROUPS = ["/data", "/volume_index", "/subjects", "/slices", "/splits"]

_REQUIRED_DATASETS = {
    "/data": ["images", "masks"],
    "/volume_index": ["subject_id", "start_row", "end_row", "depth", "original_shape", "original_spacing_mm"],
    "/subjects": [
        "subject_id",
        "cohort",
        "site",
        "total_lesion_volume_mm3",
        "class_volumes_mm3",
        "volume_stratum",
        "n_lesion_slices",
        "total_lesion_voxels",
        "rank_by_lesion_volume",
    ],
    "/slices": [
        "subject_id",
        "subject_idx",
        "depth_idx",
        "has_lesion",
        "lesion_voxel_count",
        "class_voxel_counts",
        "lesion_area_mm2",
    ],
}


@dataclass(frozen=True)
class DatasetHeader:
    """Mirrors all root-level attributes of a standardized HDF5 file.

    Parameters
    ----------
    format_version : str
        Schema version (currently ``"1.0"``).
    dataset_name : str
        Machine-readable dataset identifier (e.g. ``"brats_men"``).
    dataset_display_name : str
        Human-readable name (e.g. ``"BraTS 2024 Meningioma"``).
    dataset_version : str
        Dataset release version.
    dataset_reference : str
        Citation string for the dataset.
    task : str
        ``"binary_segmentation"`` or ``"multi_class_segmentation"``.
    pathology : str
        Target pathology (e.g. ``"meningioma"``).
    anatomy : str
        Anatomical region (e.g. ``"brain"``).
    n_subjects : int
        Number of subjects (sessions) in the file.
    n_total_slices : int
        Total axial slices across all subjects.
    n_lesion_slices : int
        Slices with ``has_lesion=True``.
    spatial_dims : int
        Always ``3`` (volumes stored as stacked slices).
    n_label_classes : int
        Number of foreground label channels (``K``).
    label_names : list[str]
        Names of each label channel.
    label_descriptions : dict[str, str]
        Description per label name.
    n_modalities : int
        Number of imaging modalities (``C``).
    modalities : list[str]
        Names of each modality channel.
    target_spacing_mm : tuple[float, float, float]
        Target voxel spacing ``(D, H, W)`` in mm.
    in_plane_size : tuple[int, int]
        Target in-plane spatial size ``(H, W)``.
    depth_handling : str
        ``"native"`` (variable depth) or ``"fixed"``.
    intensity_normalization : str
        Normalization method applied at conversion time.
    skull_stripped : bool
        Whether the data is skull-stripped.
    min_lesion_voxels_per_slice : int
        Minimum foreground voxels for ``has_lesion=True``.
    created_utc : str
        ISO 8601 creation timestamp.
    creator_version : str
        Package version that created the file.
    preprocessing_config_sha : str
        SHA-256 of the preprocessing configuration.
    source_format : str
        Original data format (e.g. ``"nifti"``).
    """

    format_version: str
    dataset_name: str
    dataset_display_name: str
    dataset_version: str
    dataset_reference: str
    task: str
    pathology: str
    anatomy: str
    n_subjects: int
    n_total_slices: int
    n_lesion_slices: int
    spatial_dims: int
    n_label_classes: int
    label_names: list[str]
    label_descriptions: dict[str, str]
    n_modalities: int
    modalities: list[str]
    target_spacing_mm: tuple[float, float, float]
    in_plane_size: tuple[int, int]
    depth_handling: str
    intensity_normalization: str
    skull_stripped: bool
    min_lesion_voxels_per_slice: int
    created_utc: str
    creator_version: str
    preprocessing_config_sha: str
    source_format: str

    @classmethod
    def from_h5(cls, path: Path) -> DatasetHeader:
        """Read and validate root attributes from an HDF5 file.

        Parameters
        ----------
        path : Path
            Path to the HDF5 file.

        Returns
        -------
        DatasetHeader
            Populated header dataclass.

        Raises
        ------
        SchemaValidationError
            If a required attribute is missing or has the wrong type.
        """
        with h5py.File(path, "r") as f:
            attrs = dict(f.attrs)

        def _get(key: str) -> Any:
            if key not in attrs:
                raise SchemaValidationError(f"Missing root attribute: {key}")
            return attrs[key]

        def _decode_json(key: str) -> Any:
            raw = _get(key)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw)

        return cls(
            format_version=str(_get("format_version")),
            dataset_name=str(_get("dataset_name")),
            dataset_display_name=str(_get("dataset_display_name")),
            dataset_version=str(_get("dataset_version")),
            dataset_reference=str(_get("dataset_reference")),
            task=str(_get("task")),
            pathology=str(_get("pathology")),
            anatomy=str(_get("anatomy")),
            n_subjects=int(_get("n_subjects")),
            n_total_slices=int(_get("n_total_slices")),
            n_lesion_slices=int(_get("n_lesion_slices")),
            spatial_dims=int(_get("spatial_dims")),
            n_label_classes=int(_get("n_label_classes")),
            label_names=_decode_json("label_names"),
            label_descriptions=_decode_json("label_descriptions"),
            n_modalities=int(_get("n_modalities")),
            modalities=_decode_json("modalities"),
            target_spacing_mm=tuple(_decode_json("target_spacing_mm")),
            in_plane_size=tuple(_decode_json("in_plane_size")),
            depth_handling=str(_get("depth_handling")),
            intensity_normalization=str(_get("intensity_normalization")),
            skull_stripped=bool(_get("skull_stripped")),
            min_lesion_voxels_per_slice=int(_get("min_lesion_voxels_per_slice")),
            created_utc=str(_get("created_utc")),
            creator_version=str(_get("creator_version")),
            preprocessing_config_sha=str(_get("preprocessing_config_sha")),
            source_format=str(_get("source_format")),
        )

    def write_to_h5(self, f: h5py.File) -> None:
        """Write all header fields as root-level HDF5 attributes.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file in write mode.
        """
        for fld in fields(self):
            val = getattr(self, fld.name)
            if isinstance(val, (list, dict, tuple)):
                f.attrs[fld.name] = json.dumps(val)
            elif isinstance(val, bool):
                f.attrs[fld.name] = np.bool_(val)
            else:
                f.attrs[fld.name] = val


def validate_h5(path: Path) -> list[str]:
    """Validate an HDF5 file against the v1.0 schema.

    Parameters
    ----------
    path : Path
        Path to the HDF5 file.

    Returns
    -------
    list[str]
        List of error descriptions.  Empty list means the file is valid.
    """
    errors: list[str] = []

    if not path.exists():
        return [f"File does not exist: {path}"]

    try:
        f = h5py.File(path, "r")
    except Exception as exc:
        return [f"Cannot open HDF5 file: {exc}"]

    with f:
        # Check required root attributes
        required_attrs = [fld.name for fld in fields(DatasetHeader)]
        for attr in required_attrs:
            if attr not in f.attrs:
                errors.append(f"Missing root attribute: {attr}")

        # Check required groups
        for group in _REQUIRED_GROUPS:
            if group not in f:
                errors.append(f"Missing group: {group}")

        # Check required datasets within groups
        for group, datasets in _REQUIRED_DATASETS.items():
            if group not in f:
                continue
            for ds_name in datasets:
                if ds_name not in f[group]:
                    errors.append(f"Missing dataset: {group}/{ds_name}")

        # Shape consistency checks
        if "/data/images" in f and "/slices/subject_id" in f:
            n_rows = f["/data/images"].shape[0]
            n_slice_rows = f["/slices/subject_id"].shape[0]
            if n_rows != n_slice_rows:
                errors.append(
                    f"Shape mismatch: /data/images has {n_rows} rows "
                    f"but /slices/subject_id has {n_slice_rows}"
                )

        if "/data/images" in f and "/data/masks" in f:
            if f["/data/images"].shape[0] != f["/data/masks"].shape[0]:
                errors.append(
                    "Shape mismatch: /data/images and /data/masks have different row counts"
                )

        if "n_subjects" in f.attrs and "/volume_index/subject_id" in f:
            n_subj_attr = int(f.attrs["n_subjects"])
            n_subj_data = f["/volume_index/subject_id"].shape[0]
            if n_subj_attr != n_subj_data:
                errors.append(
                    f"n_subjects attr ({n_subj_attr}) != "
                    f"/volume_index/subject_id length ({n_subj_data})"
                )

        if "n_total_slices" in f.attrs and "/data/images" in f:
            n_slices_attr = int(f.attrs["n_total_slices"])
            n_slices_data = f["/data/images"].shape[0]
            if n_slices_attr != n_slices_data:
                errors.append(
                    f"n_total_slices attr ({n_slices_attr}) != "
                    f"/data/images row count ({n_slices_data})"
                )

    return errors
