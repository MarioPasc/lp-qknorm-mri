"""Dataset converter protocol and standardized HDF5 writer.

Defines the :class:`DatasetConverter` protocol that all dataset-specific
converters must implement, plus the generic :func:`write_standardized_h5`
writer that produces a self-describing HDF5 file from any converter's output.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import h5py
import numpy as np
from tqdm import tqdm  # type: ignore[import-untyped]

import lpqknorm
from lpqknorm.data.schema import FORMAT_VERSION, DatasetHeader
from lpqknorm.data.splits import make_patient_kfold
from lpqknorm.data.stratification import compute_strata


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectRecord:
    """Raw discovery result for one subject/session.

    Parameters
    ----------
    subject_id : str
        Unique identifier (e.g. ``"BraTS-MEN-00004-000"``).
    image_paths : dict[str, Path]
        Mapping from modality name to NIfTI file path.
    mask_path : Path
        Path to the segmentation mask NIfTI.
    cohort : str
        Cohort or dataset partition name.
    site : str | None
        Acquisition site, if known.
    """

    subject_id: str
    image_paths: dict[str, Path]
    mask_path: Path
    cohort: str
    site: str | None


@dataclass(frozen=True)
class SubjectVolume:
    """Preprocessed volume ready for HDF5 writing.

    Parameters
    ----------
    subject_id : str
        Unique identifier.
    images : np.ndarray
        Shape ``(D, C, H, W)`` float32 normalized images.
    masks : np.ndarray
        Shape ``(D, K, H, W)`` uint8 binary masks per class.
    original_shape : tuple[int, int, int]
        ``(D, H, W)`` before any preprocessing.
    original_spacing_mm : tuple[float, float, float]
        Voxel spacing in mm before resampling.
    cohort : str
        Cohort name.
    site : str
        Acquisition site (``"unknown"`` if unavailable).
    """

    subject_id: str
    images: np.ndarray
    masks: np.ndarray
    original_shape: tuple[int, int, int]
    original_spacing_mm: tuple[float, float, float]
    cohort: str
    site: str


@dataclass(frozen=True)
class DatasetInfo:
    """Static metadata describing a dataset.

    Parameters
    ----------
    name : str
        Machine-readable identifier (e.g. ``"brats_men"``).
    display_name : str
        Human-readable name.
    version : str
        Dataset release version.
    reference : str
        Citation string.
    pathology : str
        Target pathology.
    anatomy : str
        Anatomical region.
    task : str
        ``"binary_segmentation"`` or ``"multi_class_segmentation"``.
    label_names : list[str]
        Names of foreground label channels.
    label_descriptions : dict[str, str]
        Descriptions per label name.
    modalities : list[str]
        Imaging modality names.
    source_format : str
        Original data format.
    """

    name: str
    display_name: str
    version: str
    reference: str
    pathology: str
    anatomy: str
    task: str
    label_names: list[str]
    label_descriptions: dict[str, str]
    modalities: list[str]
    source_format: str


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing configuration for the converter pipeline.

    Parameters
    ----------
    target_spacing_mm : tuple[float, float, float]
        Target isotropic spacing in mm.
    in_plane_size : tuple[int, int]
        Target ``(H, W)`` after crop/resize.
    intensity_normalization : str
        Normalization method (e.g. ``"z_score_nonzero"``).
    min_lesion_voxels_per_slice : int
        Minimum foreground voxels for a slice to be ``has_lesion=True``.
    skull_stripped : bool
        Whether the input data is skull-stripped.
    seed : int
        Random seed for reproducibility.
    """

    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    in_plane_size: tuple[int, int] = (224, 224)
    intensity_normalization: str = "z_score_nonzero"
    min_lesion_voxels_per_slice: int = 10
    skull_stripped: bool = True
    seed: int = 20260216


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DatasetConverter(Protocol):
    """Protocol for dataset-specific converters."""

    @property
    def info(self) -> DatasetInfo: ...

    def discover_subjects(self, raw_root: Path) -> list[SubjectRecord]: ...

    def load_subject(
        self, record: SubjectRecord, cfg: PreprocessConfig
    ) -> SubjectVolume | None: ...


# ---------------------------------------------------------------------------
# Generic HDF5 writer
# ---------------------------------------------------------------------------


def write_standardized_h5(
    subjects: Iterable[SubjectVolume],
    info: DatasetInfo,
    cfg: PreprocessConfig,
    out_path: Path,
    n_folds: int = 3,
    seed: int = 20260216,
    expected_n_subjects: int | None = None,
    patient_id_extractor: Callable[[str], str] | None = None,
) -> Path:
    """Write a standardized HDF5 file from an iterable of preprocessed volumes.

    Parameters
    ----------
    subjects : Iterable[SubjectVolume]
        Preprocessed volumes, streamed one at a time.
    info : DatasetInfo
        Static dataset metadata.
    cfg : PreprocessConfig
        Preprocessing configuration.
    out_path : Path
        Output HDF5 file path.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed for splits.
    expected_n_subjects : int | None
        Expected subject count (for progress bar).
    patient_id_extractor : callable or None
        Function ``(subject_id: str) -> str`` extracting patient ID from
        subject ID.  Defaults to identity (subject_id == patient_id).

    Returns
    -------
    Path
        Path to the written HDF5 file.
    """
    if patient_id_extractor is None:
        patient_id_extractor = lambda sid: sid  # noqa: E731

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Accumulators for metadata (lightweight, kept in RAM)
    subject_ids: list[str] = []
    patient_ids_list: list[str] = []
    cohorts: list[str] = []
    sites: list[str] = []
    start_rows: list[int] = []
    end_rows: list[int] = []
    depths: list[int] = []
    original_shapes: list[tuple[int, int, int]] = []
    original_spacings: list[tuple[float, float, float]] = []

    # Per-slice metadata accumulators
    slice_subject_ids: list[str] = []
    slice_subject_idxs: list[int] = []
    slice_depth_idxs: list[int] = []
    slice_has_lesion: list[bool] = []
    slice_lesion_voxel_counts: list[int] = []
    slice_class_voxel_counts: list[np.ndarray] = []
    slice_lesion_area_mm2: list[float] = []

    current_row = 0
    n_modalities: int | None = None
    n_classes: int | None = None

    progress = tqdm(
        subjects,
        desc="Writing subjects",
        total=expected_n_subjects,
        unit="subj",
    )

    with h5py.File(out_path, "w") as f:
        images_ds = None
        masks_ds = None

        for subj_idx, sv in enumerate(progress):
            d, c, h, w = sv.images.shape

            if n_modalities is None:
                n_modalities = c
                n_classes = sv.masks.shape[1]
                # Create resizable datasets
                images_ds = f.create_dataset(
                    "data/images",
                    shape=(0, c, h, w),
                    maxshape=(None, c, h, w),
                    dtype=np.float32,
                    chunks=(1, c, h, w),
                    compression="gzip",
                    compression_opts=4,
                )
                masks_ds = f.create_dataset(
                    "data/masks",
                    shape=(0, n_classes, h, w),
                    maxshape=(None, n_classes, h, w),
                    dtype=np.uint8,
                    chunks=(1, n_classes, h, w),
                    compression="gzip",
                    compression_opts=4,
                )

            assert images_ds is not None and masks_ds is not None

            # Resize and write data
            new_size = current_row + d
            images_ds.resize(new_size, axis=0)
            masks_ds.resize(new_size, axis=0)
            images_ds[current_row:new_size] = sv.images
            masks_ds[current_row:new_size] = sv.masks

            # Accumulate subject metadata
            subject_ids.append(sv.subject_id)
            patient_ids_list.append(patient_id_extractor(sv.subject_id))
            cohorts.append(sv.cohort)
            sites.append(sv.site)
            start_rows.append(current_row)
            end_rows.append(new_size)
            depths.append(d)
            original_shapes.append(sv.original_shape)
            original_spacings.append(sv.original_spacing_mm)

            # Per-slice metadata
            voxel_size_mm2 = cfg.target_spacing_mm[1] * cfg.target_spacing_mm[2]
            for z in range(d):
                mask_slice = sv.masks[z]  # (K, H, W)
                union_mask = mask_slice.any(axis=0)  # (H, W)
                total_fg = int(union_mask.sum())
                has_lesion = total_fg >= cfg.min_lesion_voxels_per_slice
                class_counts = np.array(
                    [int(mask_slice[k].sum()) for k in range(mask_slice.shape[0])]
                )

                slice_subject_ids.append(sv.subject_id)
                slice_subject_idxs.append(subj_idx)
                slice_depth_idxs.append(z)
                slice_has_lesion.append(has_lesion)
                slice_lesion_voxel_counts.append(total_fg)
                slice_class_voxel_counts.append(class_counts)
                slice_lesion_area_mm2.append(float(total_fg) * voxel_size_mm2)

            current_row = new_size
            progress.set_postfix(slices=current_row)

        if n_modalities is None:
            logger.warning("No subjects provided — writing empty file")
            n_modalities = len(info.modalities)
            n_classes = len(info.label_names)

        assert n_classes is not None  # guaranteed after first subject or fallback

        # Convert accumulators to arrays
        n_subjects = len(subject_ids)
        n_total_slices = current_row

        # --- Compute per-subject aggregated metadata ---
        total_lesion_volumes = np.zeros(n_subjects, dtype=np.float64)
        class_volumes = np.zeros((n_subjects, n_classes), dtype=np.float64)
        n_lesion_slices_per_subj = np.zeros(n_subjects, dtype=np.int32)
        total_lesion_voxels_per_subj = np.zeros(n_subjects, dtype=np.int64)

        voxel_vol_mm3 = float(np.prod(cfg.target_spacing_mm))
        for i in range(n_subjects):
            s, e = start_rows[i], end_rows[i]
            for j in range(s, e):
                if slice_has_lesion[j]:
                    n_lesion_slices_per_subj[i] += 1
                total_lesion_voxels_per_subj[i] += slice_lesion_voxel_counts[j]
                class_volumes[i] += slice_class_voxel_counts[j] * voxel_vol_mm3

            total_lesion_volumes[i] = (
                float(total_lesion_voxels_per_subj[i]) * voxel_vol_mm3
            )

        # Stratification
        strata_labels, strata_boundaries = compute_strata(total_lesion_volumes)

        # Rank by lesion volume (ascending)
        rank_by_volume = np.argsort(np.argsort(total_lesion_volumes)).astype(np.int32)

        # Patient-level splits
        subject_ids_arr = np.array(subject_ids)
        patient_ids_arr = np.array(patient_ids_list)
        folds, split_hash = make_patient_kfold(
            subject_ids_arr,
            patient_ids_arr,
            strata_labels,
            n_folds=n_folds,
            seed=seed,
        )

        # Lesion slice sorted indices
        lesion_indices = [i for i in range(n_total_slices) if slice_has_lesion[i]]
        lesion_areas = np.array([slice_lesion_area_mm2[i] for i in lesion_indices])
        sorted_order = np.argsort(lesion_areas)
        lesion_slice_indices_by_area = np.array(lesion_indices, dtype=np.int32)[
            sorted_order
        ]

        n_lesion_total = len(lesion_indices)

        # --- Write metadata groups ---
        str_dt = h5py.string_dtype()

        # /volume_index/
        vi = f.create_group("volume_index")
        vi.create_dataset(
            "subject_id", data=np.array(subject_ids, dtype=object), dtype=str_dt
        )
        vi.create_dataset("start_row", data=np.array(start_rows, dtype=np.int64))
        vi.create_dataset("end_row", data=np.array(end_rows, dtype=np.int64))
        vi.create_dataset("depth", data=np.array(depths, dtype=np.int32))
        vi.create_dataset(
            "original_shape", data=np.array(original_shapes, dtype=np.int32)
        )
        vi.create_dataset(
            "original_spacing_mm", data=np.array(original_spacings, dtype=np.float32)
        )

        # /subjects/
        sg = f.create_group("subjects")
        sg.create_dataset(
            "subject_id", data=np.array(subject_ids, dtype=object), dtype=str_dt
        )
        sg.create_dataset("cohort", data=np.array(cohorts, dtype=object), dtype=str_dt)
        sg.create_dataset("site", data=np.array(sites, dtype=object), dtype=str_dt)
        sg.create_dataset("total_lesion_volume_mm3", data=total_lesion_volumes)
        sg.create_dataset("class_volumes_mm3", data=class_volumes)
        sg.create_dataset(
            "volume_stratum", data=np.array(strata_labels, dtype=object), dtype=str_dt
        )
        sg.create_dataset("n_lesion_slices", data=n_lesion_slices_per_subj)
        sg.create_dataset("total_lesion_voxels", data=total_lesion_voxels_per_subj)
        sg.create_dataset("rank_by_lesion_volume", data=rank_by_volume)

        # /slices/
        sl = f.create_group("slices")
        sl.create_dataset(
            "subject_id", data=np.array(slice_subject_ids, dtype=object), dtype=str_dt
        )
        sl.create_dataset(
            "subject_idx", data=np.array(slice_subject_idxs, dtype=np.int32)
        )
        sl.create_dataset("depth_idx", data=np.array(slice_depth_idxs, dtype=np.int32))
        sl.create_dataset("has_lesion", data=np.array(slice_has_lesion, dtype=bool))
        sl.create_dataset(
            "lesion_voxel_count",
            data=np.array(slice_lesion_voxel_counts, dtype=np.int32),
        )
        sl.create_dataset(
            "class_voxel_counts",
            data=np.stack(slice_class_voxel_counts).astype(np.int32)
            if slice_class_voxel_counts
            else np.zeros((0, n_classes), dtype=np.int32),
        )
        sl.create_dataset(
            "lesion_area_mm2", data=np.array(slice_lesion_area_mm2, dtype=np.float32)
        )
        sl.create_dataset(
            "lesion_slice_indices_by_area", data=lesion_slice_indices_by_area
        )

        # /splits/
        sp = f.create_group("splits")
        sp.attrs["n_folds"] = n_folds
        sp.attrs["seed"] = seed
        sp.attrs["stratified_by"] = "volume_stratum"
        sp.attrs["split_hash"] = split_hash

        for fold in folds:
            fg = sp.create_group(f"fold_{fold.fold_idx}")
            fg.create_dataset(
                "train_subjects",
                data=np.array(fold.train_subjects, dtype=object),
                dtype=str_dt,
            )
            fg.create_dataset(
                "val_subjects",
                data=np.array(fold.val_subjects, dtype=object),
                dtype=str_dt,
            )
            fg.create_dataset(
                "test_subjects",
                data=np.array(fold.test_subjects, dtype=object),
                dtype=str_dt,
            )

        # /strata/
        st = f.create_group("strata")
        st.attrs["n_strata"] = 3
        st.attrs["method"] = "percentile"
        st.attrs["boundaries_mm3"] = json.dumps(strata_boundaries.tolist())
        st.attrs["boundary_percentiles"] = json.dumps([33.33, 66.67])

        # Root attributes (header)
        cfg_sha = hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True).encode()
        ).hexdigest()[:16]

        header = DatasetHeader(
            format_version=FORMAT_VERSION,
            dataset_name=info.name,
            dataset_display_name=info.display_name,
            dataset_version=info.version,
            dataset_reference=info.reference,
            task=info.task,
            pathology=info.pathology,
            anatomy=info.anatomy,
            n_subjects=n_subjects,
            n_total_slices=n_total_slices,
            n_lesion_slices=n_lesion_total,
            spatial_dims=3,
            n_label_classes=n_classes,
            label_names=info.label_names,
            label_descriptions=info.label_descriptions,
            n_modalities=n_modalities,
            modalities=info.modalities,
            target_spacing_mm=cfg.target_spacing_mm,
            in_plane_size=cfg.in_plane_size,
            depth_handling="native",
            intensity_normalization=cfg.intensity_normalization,
            skull_stripped=cfg.skull_stripped,
            min_lesion_voxels_per_slice=cfg.min_lesion_voxels_per_slice,
            created_utc=datetime.now(UTC).isoformat(),
            creator_version=lpqknorm.__version__,
            preprocessing_config_sha=cfg_sha,
            source_format=info.source_format,
        )
        header.write_to_h5(f)

    logger.info(
        "Wrote %s: %d subjects, %d slices (%d with lesions) to %s",
        info.name,
        n_subjects,
        n_total_slices,
        n_lesion_total,
        out_path,
    )

    return out_path
