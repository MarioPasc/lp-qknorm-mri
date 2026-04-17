"""Data pipeline for standardized HDF5 datasets with multi-dataset converter support."""

from __future__ import annotations

from lpqknorm.data.converter import (
    DatasetConverter,
    DatasetInfo,
    PreprocessConfig,
    SubjectRecord,
    SubjectVolume,
    write_standardized_h5,
)
from lpqknorm.data.datamodule import (
    MockAtlasDataModule,
    MockDataConfig,
    SegmentationDataModule,
)
from lpqknorm.data.schema import DatasetHeader, validate_h5
from lpqknorm.data.splits import FoldSpec, make_patient_kfold
from lpqknorm.data.stratification import compute_strata


__all__ = [
    "DatasetConverter",
    "DatasetHeader",
    "DatasetInfo",
    "FoldSpec",
    "MockAtlasDataModule",
    "MockDataConfig",
    "PreprocessConfig",
    "SegmentationDataModule",
    "SubjectRecord",
    "SubjectVolume",
    "compute_strata",
    "make_patient_kfold",
    "validate_h5",
    "write_standardized_h5",
]
