"""MONAI transform compositions for 2D and 3D training/validation."""

from __future__ import annotations

from monai.transforms import (  # type: ignore[attr-defined]
    Compose,
    EnsureTyped,
    RandAffined,
    RandBiasFieldd,
    RandFlipd,
    RandGaussianNoised,
)


_IMAGE_KEY = "image"
_MASK_KEY = "mask"
_KEYS = [_IMAGE_KEY, _MASK_KEY]


def get_train_transforms_2d() -> Compose:
    """Training augmentation pipeline for 2D slices.

    Returns
    -------
    Compose
        MONAI transform composition.
    """
    return Compose(
        [
            EnsureTyped(keys=_KEYS, dtype="float32"),
            RandAffined(
                keys=_KEYS,
                prob=0.5,
                rotate_range=(0.2,),
                scale_range=((-0.1, 0.1),) * 2,
                translate_range=(10,) * 2,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandGaussianNoised(keys=[_IMAGE_KEY], prob=0.3, std=0.01),
            RandBiasFieldd(keys=[_IMAGE_KEY], prob=0.2, coeff_range=(0.0, 0.1)),
        ]
    )


def get_val_transforms_2d() -> Compose:
    """Validation/test transform pipeline for 2D slices.

    Returns
    -------
    Compose
        MONAI transform composition (no augmentation).
    """
    return Compose([EnsureTyped(keys=_KEYS, dtype="float32")])


def get_train_transforms_3d() -> Compose:
    """Training augmentation pipeline for 3D volumes.

    Returns
    -------
    Compose
        MONAI transform composition with 3D-specific augmentations.
    """
    return Compose(
        [
            EnsureTyped(keys=_KEYS, dtype="float32"),
            RandAffined(
                keys=_KEYS,
                prob=0.5,
                rotate_range=(0.2, 0.2, 0.2),
                scale_range=((-0.1, 0.1),) * 3,
                translate_range=(10,) * 3,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandGaussianNoised(keys=[_IMAGE_KEY], prob=0.3, std=0.01),
            RandFlipd(keys=_KEYS, prob=0.5, spatial_axis=0),
        ]
    )


def get_val_transforms_3d() -> Compose:
    """Validation/test transform pipeline for 3D volumes.

    Returns
    -------
    Compose
        MONAI transform composition (no augmentation).
    """
    return Compose([EnsureTyped(keys=_KEYS, dtype="float32")])
