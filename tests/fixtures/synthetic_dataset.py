"""Synthetic dataset fixtures for Phase 1 testing."""

from __future__ import annotations

import numpy as np

from lpqknorm.data.converter import DatasetInfo, SubjectVolume


def make_synthetic_info(
    n_modalities: int = 1,
    n_classes: int = 1,
    name: str = "synthetic",
) -> DatasetInfo:
    """Create a synthetic DatasetInfo for testing.

    Parameters
    ----------
    n_modalities : int
        Number of imaging modalities.
    n_classes : int
        Number of foreground label classes.
    name : str
        Dataset name.

    Returns
    -------
    DatasetInfo
    """
    label_names = [f"class_{i}" for i in range(n_classes)]
    return DatasetInfo(
        name=name,
        display_name=f"Synthetic {name}",
        version="test",
        reference="synthetic test data",
        pathology="synthetic",
        anatomy="brain",
        task="binary_segmentation" if n_classes == 1 else "multi_class_segmentation",
        label_names=label_names,
        label_descriptions={ln: f"synthetic {ln}" for ln in label_names},
        modalities=[f"mod_{i}" for i in range(n_modalities)],
        source_format="synthetic",
    )


def make_synthetic_subjects(
    n_subjects: int = 5,
    n_modalities: int = 1,
    n_classes: int = 1,
    depth: int = 10,
    img_size: tuple[int, int] = (32, 32),
    seed: int = 0,
    variable_depth: bool = False,
) -> list[SubjectVolume]:
    """Generate synthetic SubjectVolume objects with gaussian-blob lesions.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_modalities : int
        Number of channels in images.
    n_classes : int
        Number of foreground mask channels.
    depth : int
        Number of slices per volume (base depth if ``variable_depth``).
    img_size : tuple[int, int]
        ``(H, W)`` spatial size.
    seed : int
        Random seed.
    variable_depth : bool
        If ``True``, each subject has a different depth.

    Returns
    -------
    list[SubjectVolume]
    """
    rng = np.random.RandomState(seed)
    h, w = img_size
    subjects: list[SubjectVolume] = []

    for i in range(n_subjects):
        d = depth + i * 5 if variable_depth else depth

        images = rng.randn(d, n_modalities, h, w).astype(np.float32)

        masks = np.zeros((d, n_classes, h, w), dtype=np.uint8)

        # Place gaussian-blob lesions in a few slices
        n_lesion_slices = max(1, d // 3)
        lesion_slice_indices = rng.choice(d, size=n_lesion_slices, replace=False)

        for z in lesion_slice_indices:
            for k in range(n_classes):
                cy = rng.randint(h // 4, 3 * h // 4)
                cx = rng.randint(w // 4, 3 * w // 4)
                r = rng.randint(2, max(3, min(h, w) // 6))
                y0 = max(0, cy - r)
                y1 = min(h, cy + r)
                x0 = max(0, cx - r)
                x1 = min(w, cx + r)
                masks[z, k, y0:y1, x0:x1] = 1

        subjects.append(
            SubjectVolume(
                subject_id=f"subj_{i:03d}",
                images=images,
                masks=masks,
                original_shape=(d, h * 2, w * 2),
                original_spacing_mm=(1.0, 1.0, 1.0),
                cohort="synthetic",
                site="unknown",
            )
        )

    return subjects
