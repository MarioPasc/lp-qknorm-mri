"""BraTS 2024 Meningioma dataset converter.

Converts raw BraTS-MEN NIfTI files to the standardized HDF5 format.
The dataset contains 4 MRI modalities (T1n, T1c, T2w, T2-FLAIR) and
3-class segmentation (NET/NCR, SNFH, ET).

Native grid: 240x240x155 at 1mm isotropic, skull-stripped, MNI-registered.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import nibabel as nib
import numpy as np

from lpqknorm.data.converter import (
    DatasetInfo,
    PreprocessConfig,
    SubjectRecord,
    SubjectVolume,
)
from lpqknorm.utils.exceptions import ConverterError, DataIntegrityError


logger = logging.getLogger(__name__)

_DIR_PATTERN = re.compile(r"^BraTS-MEN-(\d{5})-(\d{3})$")
_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
_MODALITY_DISPLAY = ("T1n", "T1c", "T2w", "T2f")
_SEG_LABEL_TO_CHANNEL = {1: 0, 2: 1, 3: 2}
_N_CLASSES = 3
_NATIVE_SHAPE_HW = (240, 240)
_NATIVE_DEPTH = 155
_SPACING_TOL = 0.05


def extract_patient_id(subject_id: str) -> str:
    """Extract the 5-digit patient ID from a BraTS-MEN subject ID.

    Parameters
    ----------
    subject_id : str
        Full session name (e.g. ``"BraTS-MEN-00004-000"``).

    Returns
    -------
    str
        5-digit patient ID (e.g. ``"00004"``).
    """
    parts = subject_id.split("-")
    return parts[2]


class BraTSMenConverter:
    """Converter for the BraTS 2024 Meningioma training dataset."""

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="brats_men",
            display_name="BraTS 2024 Meningioma",
            version="2024",
            reference=(
                "BraTS 2024 Meningioma Challenge. "
                "https://www.synapse.org/Synapse:syn51156910/wiki/622553"
            ),
            pathology="meningioma",
            anatomy="brain",
            task="multi_class_segmentation",
            label_names=["NET_NCR", "SNFH", "ET"],
            label_descriptions={
                "NET_NCR": "Non-enhancing tumor / necrotic core (label 1)",
                "SNFH": "Surrounding non-tumor FLAIR hyperintensity / edema (label 2)",
                "ET": "Enhancing tumor (label 3)",
            },
            modalities=list(_MODALITY_DISPLAY),
            source_format="nifti",
        )

    def discover_subjects(self, raw_root: Path) -> list[SubjectRecord]:
        """Scan the BraTS-MEN directory for valid sessions.

        Parameters
        ----------
        raw_root : Path
            Root directory containing per-session subdirectories.

        Returns
        -------
        list[SubjectRecord]
            Sorted list of discovered sessions.

        Raises
        ------
        DataIntegrityError
            If fewer than 900 valid sessions are found, or if expected
            files are missing for any session.
        """
        raw_root = Path(raw_root)
        if not raw_root.is_dir():
            raise DataIntegrityError(f"Raw root is not a directory: {raw_root}")

        records: list[SubjectRecord] = []

        for entry in sorted(raw_root.iterdir()):
            if not entry.is_dir():
                continue
            match = _DIR_PATTERN.match(entry.name)
            if match is None:
                continue

            subject_id = entry.name
            image_paths: dict[str, Path] = {}

            for mod in _MODALITIES:
                p = entry / f"{subject_id}-{mod}.nii.gz"
                if not p.exists():
                    raise DataIntegrityError(
                        f"Missing modality file for {subject_id}",
                        {"modality": mod, "expected_path": str(p)},
                    )
                image_paths[mod] = p

            mask_path = entry / f"{subject_id}-seg.nii.gz"
            if not mask_path.exists():
                raise DataIntegrityError(
                    f"Missing segmentation for {subject_id}",
                    {"expected_path": str(mask_path)},
                )

            records.append(
                SubjectRecord(
                    subject_id=subject_id,
                    image_paths=image_paths,
                    mask_path=mask_path,
                    cohort="BraTS-MEN-Train",
                    site=None,
                )
            )

        if len(records) < 900:
            raise DataIntegrityError(
                f"Expected >= 900 BraTS-MEN sessions, found {len(records)}",
                {"raw_root": str(raw_root)},
            )

        logger.info("Discovered %d BraTS-MEN sessions in %s", len(records), raw_root)
        return records

    def load_subject(
        self, record: SubjectRecord, cfg: PreprocessConfig
    ) -> SubjectVolume | None:
        """Load, verify, crop, and normalize one BraTS-MEN session.

        Parameters
        ----------
        record : SubjectRecord
            Discovery record with file paths.
        cfg : PreprocessConfig
            Preprocessing configuration.

        Returns
        -------
        SubjectVolume or None
            Preprocessed volume, or ``None`` if the volume should be
            excluded (e.g. entirely empty masks).

        Raises
        ------
        ConverterError
            If the NIfTI files have unexpected shapes or spacing.
        """
        sid = record.subject_id

        # Load and stack modalities
        volumes: list[np.ndarray] = []
        original_shape: tuple[int, int, int] | None = None
        original_spacing: tuple[float, float, float] | None = None

        for mod in _MODALITIES:
            nii: nib.Nifti1Image = nib.load(record.image_paths[mod])  # type: ignore[assignment]
            data: np.ndarray = np.asarray(nii.dataobj, dtype=np.float32)
            zooms = nii.header.get_zooms()  # type: ignore[no-untyped-call]
            spacing = tuple(float(s) for s in zooms[:3])

            if original_shape is None:
                original_shape = (data.shape[0], data.shape[1], data.shape[2])
                original_spacing = (spacing[0], spacing[1], spacing[2])

            # Verify shape
            if data.shape[:2] != _NATIVE_SHAPE_HW:
                raise ConverterError(
                    f"{sid}/{mod}: unexpected in-plane shape {data.shape[:2]}, "
                    f"expected {_NATIVE_SHAPE_HW}",
                )

            # Verify spacing (approximately 1mm isotropic)
            for dim, s in enumerate(spacing):
                if abs(s - cfg.target_spacing_mm[dim]) > _SPACING_TOL:
                    logger.warning(
                        "%s/%s: spacing dim %d = %.3f mm (expected ~%.1f)",
                        sid,
                        mod,
                        dim,
                        s,
                        cfg.target_spacing_mm[dim],
                    )

            volumes.append(data)

        assert original_shape is not None and original_spacing is not None

        # Stack: list of (240, 240, D) → (C, 240, 240, D) → (D, C, 240, 240)
        images = np.stack(volumes, axis=0)  # (C, 240, 240, D)
        images = np.transpose(images, (3, 0, 1, 2))  # (D, C, 240, 240)

        # Center-crop in-plane from 240 to target size
        th, tw = cfg.in_plane_size
        sh, sw = _NATIVE_SHAPE_HW
        h_start = (sh - th) // 2
        w_start = (sw - tw) // 2
        images = images[:, :, h_start : h_start + th, w_start : w_start + tw]

        # Z-score normalization per channel on nonzero voxels
        for c in range(images.shape[1]):
            channel = images[:, c, :, :]
            nonzero_mask = channel != 0.0
            if nonzero_mask.sum() > 0:
                mu = channel[nonzero_mask].mean()
                sigma = channel[nonzero_mask].std()
                if sigma > 1e-6:
                    channel[nonzero_mask] = (channel[nonzero_mask] - mu) / sigma
                else:
                    channel[nonzero_mask] = 0.0

        # Load and remap segmentation mask
        seg_nii: nib.Nifti1Image = nib.load(record.mask_path)  # type: ignore[assignment]
        seg_data = np.asarray(seg_nii.dataobj).astype(np.int32)  # (240, 240, D)
        seg_data = np.transpose(seg_data, (2, 0, 1))  # (D, 240, 240)
        seg_data = seg_data[:, h_start : h_start + th, w_start : w_start + tw]

        masks = np.zeros((seg_data.shape[0], _N_CLASSES, th, tw), dtype=np.uint8)
        for label_val, channel_idx in _SEG_LABEL_TO_CHANNEL.items():
            masks[:, channel_idx, :, :] = (seg_data == label_val).astype(np.uint8)

        # Check for completely empty masks
        if masks.sum() == 0:
            logger.warning("%s: all mask channels are empty, excluding", sid)
            return None

        return SubjectVolume(
            subject_id=sid,
            images=images,
            masks=masks,
            original_shape=original_shape,
            original_spacing_mm=original_spacing,
            cohort=record.cohort,
            site=record.site or "unknown",
        )
