# Phase 1 — Data Pipeline

## Goal

Build a **dataset-agnostic**, **3D-native** preprocessing pipeline that
converts raw medical imaging datasets into a standardized, self-describing
HDF5 format. The same file supports both 2D slice-level and 3D volume-level
training via a dual-mode DataModule.

The primary dataset is ATLAS v2.0 (*Liew et al., 2022,
doi:10.1038/s41597-022-01401-7*), but the architecture is designed so that
adding a new dataset (BraTS glioma, MELD epilepsy, meningioma, etc.) requires
only writing a new converter class — no changes to the DataModule, training
loop, probes, or analysis code.

Three design principles drive every decision:

1. **3D storage, flexible loading.** Volumes are stored complete (all slices,
   including non-lesion ones). The DataModule handles lesion-only filtering
   for 2D mode at load time, not at storage time. This means the same HDF5
   supports 2D experiments (current study) and 3D experiments (future work)
   without reprocessing.
2. **Self-describing files.** The HDF5 header contains everything needed to
   understand the file without external metadata: dataset name, label names,
   modalities, preprocessing parameters, lesion volumes, sorted indices, and
   cross-validation splits. Opening the file and reading root attributes gives
   a complete summary.
3. **One converter per dataset, one format for all.** Each dataset has
   idiosyncratic directory layouts, modalities, and label conventions. A
   converter translates these into the standardized schema. Downstream code
   never touches dataset-specific logic.

## Standardized HDF5 Format Specification (v1.0)

### Overview

Every preprocessed dataset lives in a single HDF5 file named
`{dataset_name}.h5` (e.g., `atlas_v2.h5`, `brats2024.h5`, `meld.h5`).
The file has five top-level groups plus root-level attributes that serve as
the file header.

```
{dataset_name}.h5
├── attrs: {file header — dataset identity, dimensions, provenance}
├── /data/
│   ├── images  (N_total, C, H, W) float32
│   └── masks   (N_total, K, H, W) uint8
├── /volume_index/
│   ├── subject_id      (S,) str
│   ├── start_row       (S,) int64
│   ├── end_row         (S,) int64
│   ├── depth           (S,) int32
│   ├── original_shape  (S, 3) int32
│   └── original_spacing_mm (S, 3) float32
├── /subjects/
│   ├── subject_id              (S,) str
│   ├── cohort                  (S,) str
│   ├── site                    (S,) str
│   ├── total_lesion_volume_mm3 (S,) float64
│   ├── class_volumes_mm3       (S, K) float64
│   ├── volume_stratum          (S,) str
│   ├── n_lesion_slices         (S,) int32
│   ├── total_lesion_voxels     (S,) int64
│   └── rank_by_lesion_volume   (S,) int32
├── /slices/
│   ├── subject_id              (N_total,) str
│   ├── subject_idx             (N_total,) int32
│   ├── depth_idx               (N_total,) int32
│   ├── has_lesion              (N_total,) bool
│   ├── lesion_voxel_count      (N_total,) int32
│   ├── class_voxel_counts      (N_total, K) int32
│   ├── lesion_area_mm2         (N_total,) float32
│   └── lesion_slice_indices_by_area (M,) int32
├── /splits/
│   ├── attrs: {n_folds, seed, stratified_by, split_hash}
│   └── fold_{i}/
│       ├── train_subjects  (N_train,) str
│       ├── val_subjects    (N_val,) str
│       └── test_subjects   (N_test,) str
└── /strata/
    └── attrs: {n_strata, method, boundaries_mm3}
```

Notation: `S = n_subjects`, `N_total = total slices across all subjects`,
`C = n_modalities`, `K = n_label_classes`, `M = n_lesion_bearing_slices`.

### File Header (Root Attributes)

Every attribute is written at the HDF5 root level. Complex types (lists,
tuples) are stored as JSON strings for portability.

| Attribute | Type | Example | Description |
|-----------|------|---------|-------------|
| `format_version` | str | `"1.0"` | Schema version for forward compatibility |
| `dataset_name` | str | `"atlas_v2"` | Machine-readable dataset identifier |
| `dataset_display_name` | str | `"ATLAS v2.0"` | Human-readable name |
| `dataset_version` | str | `"2.0"` | Dataset release version |
| `dataset_reference` | str | `"Liew et al., Sci Data 2022, doi:..."` | Citation |
| `task` | str | `"binary_segmentation"` | `"binary_segmentation"` or `"multi_class_segmentation"` |
| `pathology` | str | `"stroke"` | `"stroke"`, `"glioma"`, `"fcd"`, `"meningioma"`, ... |
| `anatomy` | str | `"brain"` | Target anatomy |
| `n_subjects` | int | `955` | Number of subjects in file |
| `n_total_slices` | int | `143250` | Sum of all subject depths |
| `n_lesion_slices` | int | `14841` | Slices with lesion voxels above threshold |
| `spatial_dims` | int | `3` | Always 3 (stored as volumes) |
| `n_label_classes` | int | `1` | Foreground classes (excluding background) |
| `label_names` | str (JSON) | `'["lesion"]'` | Ordered list, index 0 = first fg class |
| `label_description` | str (JSON) | `'{"lesion": "ischemic stroke lesion"}'` | Per-class descriptions |
| `n_modalities` | int | `1` | Number of input modalities |
| `modalities` | str (JSON) | `'["T1w"]'` | Ordered modality names |
| `target_spacing_mm` | str (JSON) | `'[1.0, 1.0, 1.0]'` | `(D, H, W)` after resampling |
| `in_plane_size` | str (JSON) | `'[224, 224]'` | `(H, W)` after crop/pad |
| `depth_handling` | str | `"native"` | `"native"` (variable) or `"fixed"` (padded) |
| `intensity_normalization` | str | `"z_score_nonzero"` | Normalization method applied |
| `skull_stripped` | bool | `true` | Whether volumes are skull-stripped |
| `min_lesion_voxels_per_slice` | int | `10` | Threshold for `has_lesion` flag |
| `created_utc` | str | `"2026-04-17T10:30:00Z"` | ISO 8601 creation timestamp |
| `creator_version` | str | `"0.1.0"` | `lpqknorm` package version |
| `preprocessing_config_sha` | str | `"a3f2..."` | SHA-256 of the converter config |
| `source_format` | str | `"nifti"` | Original file format |

### `/data/` — Volume Data (Flat Layout)

All slices from all subjects are concatenated along axis 0. Within each
subject, slices are **physically contiguous and ordered by ascending
z-index** (inferior-to-superior in RAS+ convention). The `/volume_index/`
group maps each subject to its contiguous row range.

```python
/data/images: (N_total, C, H, W)  float32
    # C = n_modalities (1 for T1w-only, 4 for BraTS multi-modal)
    # Intensity-normalized per the method in root attrs
    # Chunking: (1, C, H, W) — single-slice random access
    # Compression: gzip level 4

/data/masks:  (N_total, K, H, W)  uint8
    # K = n_label_classes (1 for binary, 3 for BraTS ET/TC/WT)
    # Each channel is a binary mask for one foreground class
    # Chunking: (1, K, H, W)
    # Compression: gzip level 4
```

**Why flat layout instead of per-subject groups.** A single contiguous
dataset avoids the overhead of traversing thousands of HDF5 groups. On
Lustre (Picasso), group-heavy files cause metadata-server saturation.
The flat layout with per-slice chunking gives O(1) random access for 2D
loading and sequential-read efficiency for 3D volume loading (contiguous
chunks within a subject).

**Why store all slices (not just lesion-bearing).** 3D models require
complete volumes. Storing the full volume once and filtering at load time
is simpler and more flexible than maintaining two copies of the data. The
per-slice metadata (`/slices/has_lesion`) enables efficient filtering for
2D lesion-only training.

### `/volume_index/` — Subject-to-Row Mapping

```python
/volume_index/subject_id:          (S,) variable-length UTF-8
/volume_index/start_row:           (S,) int64
/volume_index/end_row:             (S,) int64   # exclusive
/volume_index/depth:               (S,) int32   # = end_row - start_row
/volume_index/original_shape:      (S, 3) int32 # (D, H, W) before resampling
/volume_index/original_spacing_mm: (S, 3) float32
```

To load subject `i`'s full volume:
```python
start, end = volume_index["start_row"][i], volume_index["end_row"][i]
volume = images[start:end]  # shape (D_i, C, H, W)
```

To load a single 2D slice at row `j`:
```python
slice_2d = images[j]  # shape (C, H, W)
```

Both operations are efficient because slices within a subject are contiguous
and each slice is one HDF5 chunk.

### `/subjects/` — Subject-Level Metadata

One row per subject, ordered consistently with `/volume_index/`.

```python
/subjects/subject_id:              (S,) variable-length UTF-8
/subjects/cohort:                  (S,) variable-length UTF-8
/subjects/site:                    (S,) variable-length UTF-8  # "unknown" if unavailable
/subjects/total_lesion_volume_mm3: (S,) float64
/subjects/class_volumes_mm3:       (S, K) float64   # per-class volumes
/subjects/volume_stratum:          (S,) variable-length UTF-8  # "small"/"medium"/"large"
/subjects/n_lesion_slices:         (S,) int32   # slices with lesion > threshold
/subjects/total_lesion_voxels:     (S,) int64
/subjects/rank_by_lesion_volume:   (S,) int32   # argsort ascending (0 = smallest)
```

**`class_volumes_mm3`**: For binary segmentation (K=1), this is identical to
`total_lesion_volume_mm3`. For multi-class (e.g., BraTS with K=3), each
column gives the volume of one foreground class (ET, TC, WT). This enables
class-specific stratification and analysis.

**`rank_by_lesion_volume`**: Pre-computed argsort of `total_lesion_volume_mm3`
in ascending order. Enables quick access to the smallest/largest lesions
without re-sorting. Row `rank_by_lesion_volume[0]` is the subject with the
smallest lesion; row `rank_by_lesion_volume[-1]` the largest.

### `/slices/` — Per-Slice Metadata

One row per row of `/data/` (same length, same order). This is the slice
manifest.

```python
/slices/subject_id:          (N_total,) variable-length UTF-8
/slices/subject_idx:         (N_total,) int32   # index into /subjects/
/slices/depth_idx:           (N_total,) int32   # z-position within the volume
/slices/has_lesion:          (N_total,) bool     # lesion_voxel_count >= threshold
/slices/lesion_voxel_count:  (N_total,) int32    # total fg voxels (union of classes)
/slices/class_voxel_counts:  (N_total, K) int32  # per-class voxel counts
/slices/lesion_area_mm2:     (N_total,) float32  # in-plane lesion area

# Pre-computed sorted index of lesion-bearing slices
/slices/lesion_slice_indices_by_area: (M,) int32
    # M = n_lesion_slices (from root attrs)
    # Values are row indices into /data/ where has_lesion=True,
    # sorted by lesion_area_mm2 ascending (smallest lesion first).
    # Enables analysis binned by lesion size.
```

**Why per-slice metadata alongside per-subject.** The 2D DataLoader needs
to know which rows of `/data/` contain lesion and how large each lesion is,
without loading the masks. The pre-computed `lesion_slice_indices_by_area`
gives direct access to slices ordered by lesion size — useful for
curriculum-style training or size-stratified analysis.

### `/splits/` — Cross-Validation Splits

Stored inside the HDF5 for self-containedness, and also written as a
separate JSON file for easy inspection.

```python
/splits/ attrs:
    n_folds:        int     # 3
    seed:           int     # 20260216
    stratified_by:  str     # "volume_stratum"
    split_hash:     str     # SHA-256 of the split assignment

/splits/fold_0/train_subjects: (N_train,) variable-length UTF-8
/splits/fold_0/val_subjects:   (N_val,) variable-length UTF-8
/splits/fold_0/test_subjects:  (N_test,) variable-length UTF-8
/splits/fold_1/ ...
/splits/fold_2/ ...
```

Split assignment is at the **patient level** — no subject ID appears in
more than one partition within a fold. Stratified on `volume_stratum` so
each fold has roughly balanced small/medium/large representation.

### `/strata/` — Stratum Boundaries

```python
/strata/ attrs:
    n_strata:          int       # 3
    method:            str       # "percentile"
    boundaries_mm3:    str (JSON) # '[33rd_pct_value, 66th_pct_value]'
    boundary_percentiles: str (JSON)  # '[33.33, 66.67]'
```

Stored so that analysis can reproduce the exact same strata assignment
without recomputing percentiles.

## Dataset Converter Architecture

### Abstract Interface

Every dataset converter implements a `DatasetConverter` protocol. The
generic pipeline calls `discover_subjects` → `load_subject` (per subject)
→ `write_standardized_h5` (once, writing all subjects).

```python
# data/converter.py

from typing import Protocol, runtime_checkable

@dataclass(frozen=True)
class SubjectRecord:
    """Minimal metadata discovered from the raw directory tree."""
    subject_id: str
    image_paths: dict[str, Path]   # modality_name -> path
    mask_path: Path
    cohort: str
    site: str | None

@dataclass(frozen=True)
class SubjectVolume:
    """One subject's preprocessed 3D volume, ready for HDF5 writing."""
    subject_id: str
    images: np.ndarray         # (D, C, H, W) float32, normalized
    masks: np.ndarray          # (D, K, H, W) uint8
    original_shape: tuple[int, int, int]
    original_spacing_mm: tuple[float, float, float]
    cohort: str
    site: str

@dataclass(frozen=True)
class DatasetInfo:
    """Static metadata about the dataset, provided by the converter."""
    name: str                     # machine-readable: "atlas_v2"
    display_name: str             # human-readable: "ATLAS v2.0"
    version: str
    reference: str                # citation string
    pathology: str                # "stroke", "glioma", "fcd", ...
    anatomy: str                  # "brain"
    task: str                     # "binary_segmentation" | "multi_class_segmentation"
    label_names: list[str]        # ["lesion"] or ["ET", "TC", "WT"]
    label_descriptions: dict[str, str]
    modalities: list[str]         # ["T1w"] or ["T1w", "T1ce", "T2w", "FLAIR"]
    source_format: str            # "nifti"

@runtime_checkable
class DatasetConverter(Protocol):
    """Protocol for dataset-specific converters."""

    @property
    def info(self) -> DatasetInfo: ...

    def discover_subjects(self, raw_root: Path) -> list[SubjectRecord]:
        """Discover all subjects in the raw directory tree.

        Returns a list of SubjectRecord with paths to images and masks.
        Must validate the directory structure and raise DataIntegrityError
        on unexpected layouts.
        """
        ...

    def load_subject(
        self,
        record: SubjectRecord,
        cfg: PreprocessConfig,
    ) -> SubjectVolume | None:
        """Load and preprocess one subject's 3D volume.

        Returns None if the subject should be excluded (e.g., no lesion
        voxels at all, corrupt file, etc.). The caller logs the exclusion.

        Preprocessing steps (implemented by the converter):
        1. Load NIfTI (or DICOM, etc.)
        2. Resample to target spacing
        3. Crop/pad to in-plane size
        4. Intensity-normalize
        5. Extract and align mask channels
        """
        ...
```

### Preprocessing Configuration

```python
# data/converter.py

@dataclass(frozen=True)
class PreprocessConfig:
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    in_plane_size: tuple[int, int] = (224, 224)
    intensity_normalization: str = "z_score_nonzero"
    min_lesion_voxels_per_slice: int = 10
    skull_stripped: bool = True   # assumed; converter verifies
    seed: int = 20260216
```

### Generic Writer

The writer is dataset-agnostic. It takes a list of `SubjectVolume` objects
and the converter's `DatasetInfo`, then writes the standardized HDF5.

```python
# data/converter.py

def write_standardized_h5(
    subjects: list[SubjectVolume],
    info: DatasetInfo,
    cfg: PreprocessConfig,
    out_path: Path,
    n_folds: int = 3,
    seed: int = 20260216,
) -> Path:
    """Write all subjects to a standardized HDF5 file.

    Performs in order:
    1. Concatenate volumes into flat /data/ arrays
    2. Build /volume_index/
    3. Compute per-subject and per-slice metadata
    4. Compute lesion-volume strata
    5. Generate patient-level stratified k-fold splits
    6. Write everything to a single HDF5

    Returns the path to the written file.
    """
    ...
```

### ATLAS v2.0 Converter

```python
# data/converters/atlas.py

class AtlasConverter:
    """Converter for ATLAS v2.0 stroke lesion dataset.

    Expected raw layout (verify by inspection — see open questions):
        {raw_root}/Training/sub-{id}/ses-1/anat/
            sub-{id}_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
            sub-{id}_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz

    ATLAS v2.0 specifics:
    - Single modality: T1w
    - Binary segmentation: stroke lesion (fg=1)
    - MNI-registered, skull-stripped, defaced
    - ~955 training subjects with labels
    """

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="atlas_v2",
            display_name="ATLAS v2.0",
            version="2.0",
            reference="Liew et al., A large, curated, open-source stroke "
                      "neuroimaging dataset (ATLAS v2.0), Sci Data 2022, "
                      "doi:10.1038/s41597-022-01401-7",
            pathology="stroke",
            anatomy="brain",
            task="binary_segmentation",
            label_names=["lesion"],
            label_descriptions={"lesion": "ischemic stroke lesion"},
            modalities=["T1w"],
            source_format="nifti",
        )

    def discover_subjects(self, raw_root: Path) -> list[SubjectRecord]: ...
    def load_subject(self, record: SubjectRecord,
                     cfg: PreprocessConfig) -> SubjectVolume | None: ...
```

### Adding a New Dataset Converter

To add a new dataset (e.g., BraTS 2024 glioma):

1. Create `src/lpqknorm/data/converters/brats.py`.
2. Implement the `DatasetConverter` protocol:
   - `info` returns a `DatasetInfo` with `modalities=["T1w", "T1ce", "T2w", "FLAIR"]`,
     `label_names=["ET", "TC", "WT"]`, `task="multi_class_segmentation"`.
   - `discover_subjects` parses the BraTS directory tree.
   - `load_subject` loads and stacks the four modalities into `(D, 4, H, W)`,
     converts the segmentation mask `{0,1,2,4}` into three binary channels.
3. Register in `src/lpqknorm/data/converters/__init__.py`.
4. Add a Hydra config `configs/data/brats2024.yaml` pointing to the raw root
   and selecting `converter=brats`.

No changes needed to `datamodule.py`, `training/`, `probes/`, or `analysis/`.
The DataModule reads the HDF5 header to discover `n_modalities` and
`n_label_classes`, and adjusts the model's `in_channels` and `out_channels`
accordingly.

### Planned converters (implement as needed)

| Converter | Dataset | Pathology | Modalities | Labels | Reference |
|-----------|---------|-----------|------------|--------|-----------|
| `atlas` | ATLAS v2.0 | Stroke | T1w (1) | lesion (1) | Liew et al., 2022 |
| `brats` | BraTS 2024 | Glioma | T1, T1ce, T2, FLAIR (4) | ET, TC, WT (3) | Menze et al., 2015 |
| `meld` | MELD | FCD/Epilepsy | T1w (1) | FCD lesion (1) | Spitzer et al., 2022 |
| `meningioma` | (TBD) | Meningioma | T1ce (1) | tumor (1) | (TBD) |

## DataModule: Dual-Mode Loading (2D and 3D)

The DataModule is dataset-agnostic. It reads the HDF5 header to discover
dimensions and constructs the appropriate PyTorch datasets.

```python
# data/datamodule.py

class SegmentationDataModule(pl.LightningDataModule):
    """Dataset-agnostic DataModule with 2D and 3D loading modes.

    Reads the standardized HDF5 format. The same file supports both modes
    without reprocessing.

    Parameters
    ----------
    h5_path : Path
        Path to the standardized HDF5 file.
    fold : int
        Cross-validation fold index (0-indexed).
    spatial_mode : {"2d", "3d"}
        Loading mode.
        - "2d": each sample is a single slice (C, H, W). The DataLoader
          iterates over the slice index. Optionally filters to lesion-only
          slices via ``lesion_only``.
        - "3d": each sample is a complete volume (C, D, H, W). The
          DataLoader iterates over subjects. Volumes have variable depth,
          requiring a custom collate function or depth padding/cropping.
    lesion_only : bool
        (2D mode only) If True, only slices with
        lesion_voxel_count >= min_lesion_voxels are loaded. Ignored in 3D mode.
    min_lesion_voxels : int
        Threshold for the lesion_only filter. Read from the HDF5 header
        by default; override here if needed.
    batch_size : int
    num_workers : int
    augment : bool
        Whether to apply training augmentations.
    depth_range : tuple[int, int] | None
        (3D mode only) If set, crop/pad all volumes to this depth range.
        If None, use variable depth with a custom collate.
    """

    def __init__(
        self,
        h5_path: Path,
        fold: int,
        spatial_mode: Literal["2d", "3d"],
        batch_size: int,
        lesion_only: bool = True,
        min_lesion_voxels: int | None = None,
        num_workers: int = 8,
        augment: bool = True,
        depth_range: tuple[int, int] | None = None,
    ) -> None: ...

    def setup(self, stage: str | None = None) -> None:
        """Read HDF5 header, resolve fold splits, build index arrays."""
        # 1. Read root attrs → discover n_modalities, n_label_classes, etc.
        # 2. Read /splits/fold_{self.fold}/ → get train/val/test subject IDs
        # 3. Read /volume_index/ and /slices/ metadata into memory
        # 4. For 2D mode: build row indices per partition, optionally
        #    filtered by has_lesion
        # 5. For 3D mode: build subject indices per partition
        ...
```

### 2D Mode

Each `__getitem__` returns a single slice `(C, H, W)` and its mask
`(K, H, W)`, plus metadata (subject_id, depth_idx, lesion_voxel_count).

The dataset reads one HDF5 chunk per sample — efficient random access.
When `lesion_only=True`, the index array contains only rows where
`has_lesion=True`, matching the current study's design (every training
sample has a non-trivial segmentation target).

### 3D Mode

Each `__getitem__` returns a full volume `(C, D_i, H, W)` and mask
`(K, D_i, H, W)`, where `D_i` varies per subject. Two strategies for
batching:

1. **Variable depth + custom collate**: Pad volumes within each batch to
   the maximum depth in that batch. Most memory-efficient; standard in
   nnU-Net-style pipelines.
2. **Fixed depth crop/pad**: All volumes cropped or padded to a fixed
   depth (e.g., 128 or 160 slices). Simpler; trades some data for
   uniform tensor shapes.

The choice is configurable via `depth_range`. Default is None (variable
depth with collate), since MONAI's `SwinUNETR` handles variable spatial
dims via its patch embedding.

### Reading the HDF5 header from code

```python
# data/schema.py — dataclass mirroring root attrs

@dataclass(frozen=True)
class DatasetHeader:
    """Parsed HDF5 root attributes. Fully describes the dataset."""
    format_version: str
    dataset_name: str
    dataset_display_name: str
    pathology: str
    task: str
    n_subjects: int
    n_total_slices: int
    n_lesion_slices: int
    n_label_classes: int
    label_names: list[str]
    label_descriptions: dict[str, str]
    n_modalities: int
    modalities: list[str]
    target_spacing_mm: tuple[float, float, float]
    in_plane_size: tuple[int, int]
    min_lesion_voxels_per_slice: int
    # ... plus provenance fields

    @classmethod
    def from_h5(cls, path: Path) -> DatasetHeader:
        """Read and validate root attributes from an HDF5 file."""
        ...
```

This dataclass is used throughout downstream code. The training CLI reads
it to set `in_channels=header.n_modalities` and
`out_channels=header.n_label_classes` automatically — no manual config
needed when switching datasets.

## Prerequisites

- Conda environment `lpqknorm` installed, MONAI >= 1.3.0 available.
- Raw dataset downloaded and verified (see `scripts/download_atlas.sh` for
  ATLAS; each dataset has its own download script).
- No dependency on any other phase.

## Open Questions the Agent Must Resolve Before Coding

The agent must **inspect the actual environment** and confirm the following.
Do not assume — verify, and record resolved values as module-level constants
with citations.

### ATLAS v2.0 specifics (for `converters/atlas.py`)

1. The on-disk layout of ATLAS v2.0. Expected:
   `Training/sub-*/ses-*/anat/` containing `*_T1w.nii.gz` and
   `*_label-L_desc-T1lesion_mask.nii.gz`. Confirm against the INDI release.
2. Intensity range and skull-stripping status. ATLAS v2.0 ships defaced and
   registered to MNI-152, already skull-stripped. Re-verify by histogram
   inspection on 5 random subjects.
3. Lesion label polarity (foreground = 1 expected). Check a handful of masks.
4. Whether `participants.tsv` is present and usable for site/scanner
   stratification. If present, parse; if not, set site to `"unknown"`.
5. The "R" (hidden test) partition — default **no**: ships without labels.

### Format and architecture

6. HDF5 compression: benchmark gzip level 4 vs. blosc (lz4) on 10 subjects.
   Pick whichever gives better decode speed at comparable ratio. Document
   the choice with numbers.
7. Variable-depth 3D batching: test MONAI's `SwinUNETR(spatial_dims=3)` with
   variable-depth inputs. Confirm it handles non-cubic inputs via its
   patch embedding, or determine the minimum depth constraint.
8. Memory footprint of a full-volume ATLAS HDF5. Estimate: ~955 subjects ×
   ~180 slices × 1 × 224 × 224 × 4 bytes ≈ 35 GB uncompressed. Confirm
   compressed size with gzip-4 is manageable (target: < 15 GB).

## I/O Contract

### Inputs

- `data.converter: str` — converter name (`"atlas"`, `"brats"`, `"meld"`).
- `data.raw_root: Path` — raw dataset root after download.
- `data.cache_root: Path` — destination for the standardized HDF5 and
  auxiliary files.
- `data.preprocess: PreprocessConfig` — target spacing, in-plane size,
  normalization method, min lesion voxels threshold, seed.
- `data.n_folds: int = 3` — number of cross-validation folds.

### Outputs written to `cache_root`

- `{dataset_name}.h5` — Standardized HDF5 file as specified above. Contains
  all data, metadata, splits, and strata.
- `splits_k{n_folds}_seed{seed}.json` — External copy of the splits for
  easy inspection (same content as `/splits/` in the HDF5).
- `preprocessing_report.json` — Summary: n_subjects processed, n_excluded,
  total slices, lesion statistics, wall-clock time, warnings.
- `qc/` — Quality control artefacts:
  - `lesion_volume_histogram.png` — distribution with stratum boundaries
  - `slice_count_histogram.png` — depth distribution across subjects
  - `sample_slices.png` — 4×4 grid of random lesion-bearing slices with
    mask overlay for visual sanity check

### Exposed Python API (`src/lpqknorm/data/`)

```python
# schema.py
@dataclass(frozen=True)
class DatasetHeader:
    """Parsed root attributes of a standardized HDF5 file."""
    ...
    @classmethod
    def from_h5(cls, path: Path) -> DatasetHeader: ...

def validate_h5(path: Path) -> list[str]:
    """Validate an HDF5 file against the v1.0 schema.
    Returns a list of validation errors (empty = valid)."""
    ...

# converter.py
class DatasetConverter(Protocol): ...

@dataclass(frozen=True)
class PreprocessConfig: ...
@dataclass(frozen=True)
class SubjectRecord: ...
@dataclass(frozen=True)
class SubjectVolume: ...
@dataclass(frozen=True)
class DatasetInfo: ...

def write_standardized_h5(
    subjects: list[SubjectVolume],
    info: DatasetInfo,
    cfg: PreprocessConfig,
    out_path: Path,
    n_folds: int = 3,
    seed: int = 20260216,
) -> Path: ...

# converters/atlas.py
class AtlasConverter:
    """ATLAS v2.0 converter."""
    ...

# stratification.py
def compute_strata(
    lesion_volumes: np.ndarray,
    n_strata: int = 3,
    method: str = "percentile",
) -> tuple[np.ndarray, np.ndarray]:
    """Assign volume strata and return (labels, boundaries)."""
    ...

# splits.py
@dataclass(frozen=True)
class FoldSpec:
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]

def make_patient_kfold(
    subject_ids: np.ndarray,
    strata: np.ndarray,
    n_folds: int = 3,
    seed: int = 20260216,
) -> list[FoldSpec]: ...

# transforms.py
def get_train_transforms_2d(cfg: PreprocessConfig) -> Callable: ...
def get_val_transforms_2d() -> Callable: ...
def get_train_transforms_3d(cfg: PreprocessConfig) -> Callable: ...
def get_val_transforms_3d() -> Callable: ...

# datamodule.py
class SegmentationDataModule(pl.LightningDataModule): ...
```

## Design Notes

**Why patient-level splits.** Slices from the same patient share scanner,
intensity distribution, and lesion morphology; mixing them across splits
inflates metrics (*Roberts et al., Nat. Mach. Intell. 2021,
doi:10.1038/s42256-021-00307-0*). The split function must refuse to produce
a fold where any subject ID appears in more than one partition.

**Why 3D storage with 2D loading instead of 2D-only.** The current study
uses 2D slices for computational efficiency and because the Swin window
attention mechanism operates on 2D patches. However, the broader goal is to
study Lp normalization across pathologies and architectures, some of which
require 3D input (nnFormer, 3D Swin-UNETR). Storing complete volumes now
avoids reprocessing later. The 2D DataLoader filters to lesion-only slices
at load time, preserving the mechanistic focus of the current experiment.

**Why a single HDF5 instead of per-patient files.** Picasso's Lustre
filesystem penalises large numbers of small files (inode overhead,
metadata-server saturation under concurrent SLURM jobs). A single chunked
HDF5 reduces file count from O(n_patients) to O(1). Random access by slice
remains efficient because each slice is one chunk.

**Why pre-compute rather than extract on-the-fly.** Determinism (re-runs
read the same pixels), I/O efficiency (single HDF5 beats per-sample NIfTI
decode), and separability of preprocessing bugs from training bugs.

**Why store all slices but flag lesion-bearing ones.** The experiment targets
a mechanistic question about small lesions, so 2D training uses lesion-only
slices. But background slices are needed for (a) 3D training, (b) computing
false-positive rates in future work, and (c) analyzing how the model behaves
on lesion-free anatomy. Filtering at load time via the `has_lesion` flag
is flexible and cheap.

**Volume stratification.** Stratum boundaries are the 33rd and 66th
percentiles of per-patient lesion volume (mm^3, computed in original scanner
space before resampling). Stored in `/strata/` attrs so they can be reused
at analysis time. The **small stratum** is the headline evaluation cohort.

**Augmentations** (train-time only, applied in DataModule transforms):

- 2D: `RandAffined(rotate=0.2rad, scale=(0.9, 1.1), translate=10px)`,
  `RandGaussianNoised(std=0.01)`, `RandBiasFieldd(coeff_range=(0.0, 0.1))`.
- 3D: Same affine + noise, plus `RandFlipd(spatial_axes=[0])` (axial flip).
  No elastic deformation (lesion morphology matters).

Validation and test transforms are deterministic (resize + normalize only).

## Implementation Checklist

1. `utils/exceptions.py` — define `DataIntegrityError`, `SplitLeakageError`,
   `StratificationError`, `SchemaValidationError`, `ConverterError`.
2. `utils/seeding.py` — `set_global_seed(seed)` covering Python, NumPy,
   Torch, Torch CUDA, `PYTHONHASHSEED`, and dataloader worker init.
3. `data/schema.py` — `DatasetHeader` dataclass, `validate_h5()` function.
   This is the single source of truth for the HDF5 schema. All writing and
   reading code references this module.
4. `data/converter.py` — `DatasetConverter` protocol, `SubjectRecord`,
   `SubjectVolume`, `DatasetInfo`, `PreprocessConfig` dataclasses, and the
   generic `write_standardized_h5` function.
5. `data/converters/__init__.py` — converter registry mapping names to
   classes: `CONVERTERS = {"atlas": AtlasConverter, ...}`.
6. `data/converters/atlas.py` — `AtlasConverter` implementing
   `discover_subjects` (glob + regex, schema validation, fail if < 500
   subjects) and `load_subject` (MONAI `LoadImaged`, `Spacingd`,
   `NormalizeIntensityd(nonzero=True)`, `Resized`).
7. `data/stratification.py` — `compute_strata()` with percentile-based
   boundaries. Plot histogram as side-effect.
8. `data/splits.py` — `make_patient_kfold()` with patient-level stratified
   k-fold. Deterministic given seed.
9. `data/transforms.py` — train/val/test transform compositions for both 2D
   and 3D modes.
10. `data/datamodule.py` — `SegmentationDataModule` with dual-mode loading.
    Reads HDF5 header at setup, resolves fold partitions, builds index arrays.
11. `cli/preprocess.py` — Hydra entry point: select converter → discover →
    load per-subject → write standardized H5 → emit QC artifacts and
    `_SUCCESS` file.

## Acceptance Tests

All must pass under `pytest -q`. Tests use synthetic fixtures unless noted.

### Schema and format validation

1. **HDF5 schema compliance.** Write a small synthetic dataset via
   `write_standardized_h5`, then call `validate_h5()`. Assert zero errors.
   Separately, corrupt one attr and assert the validator catches it.

2. **Header round-trip.** Write a file, read `DatasetHeader.from_h5()`,
   assert all fields match the original `DatasetInfo` + `PreprocessConfig`.

### Volume storage and indexing

3. **Volume reconstruction from flat layout.** For a synthetic 5-subject
   dataset with known depths `[10, 15, 12, 8, 20]`, write the HDF5, then
   for each subject load `images[start:end]` via `/volume_index/` and assert
   shape `(D_i, C, H, W)` matches the original volume.

4. **2D slice access.** For any row `j` in `/data/images`, assert
   `images[j]` matches the corresponding slice from the correct subject's
   volume at the correct depth index (cross-check with `/slices/` metadata).

5. **Slice manifest consistency.** Assert `len(/slices/subject_id) ==
   len(/data/images)`. Assert every `has_lesion=True` slice has
   `lesion_voxel_count >= min_lesion_voxels_per_slice`.

6. **Sorted indices.** Assert `rank_by_lesion_volume` is a valid permutation
   and `total_lesion_volume_mm3[rank[i]] <= total_lesion_volume_mm3[rank[i+1]]`
   for all `i`. Similarly for `lesion_slice_indices_by_area`.

### Splits

7. **No patient leakage.**
   ```python
   def test_no_patient_leakage(synthetic_dataset):
       folds = make_patient_kfold(...)
       for f in folds:
           assert set(f.train_subjects) & set(f.val_subjects) == set()
           assert set(f.train_subjects) & set(f.test_subjects) == set()
           assert set(f.val_subjects) & set(f.test_subjects) == set()
       all_subjects = set().union(*(f.test_subjects for f in folds))
       assert all_subjects == set(synthetic_dataset.subject_ids)
   ```

8. **Determinism.** Two calls to `make_patient_kfold` with the same seed
   yield identical fold assignments.

9. **Stratum balance.** Each fold's train partition has at least one patient
   from each stratum; `|train ∩ small| / |train| ∈ [0.25, 0.40]`.

### Converter interface

10. **ATLAS converter discovery.** On a synthetic 5-patient directory tree
    mimicking ATLAS layout, `AtlasConverter.discover_subjects()` returns
    exactly 5 `SubjectRecord` objects with correct paths.

11. **Multi-modal converter.** A mock BraTS converter producing `(D, 4, H, W)`
    images writes correctly to `/data/images` with `C=4` and the header
    reports `n_modalities=4`.

### DataModule

12. **2D mode with lesion-only.** Load a DataModule in 2D mode with
    `lesion_only=True`. Assert every returned batch has
    `mask.sum() > 0` (no empty masks).

13. **2D mode without lesion filter.** `lesion_only=False` returns samples
    including slices where `mask.sum() == 0`.

14. **3D mode.** Load a DataModule in 3D mode. Assert returned volumes have
    shape `(C, D_i, H, W)` with `D_i > 1`.

15. **Fold partition consistency.** In 2D mode, assert every returned
    `subject_id` belongs to the correct fold partition (train/val/test).

### Stratification

16. **Monotonicity.** Sorting patients by lesion volume and re-assigning
    strata yields the same partition as `compute_strata`.

17. **Stratum boundaries stored.** Assert `/strata/` attrs contain valid
    boundary values that correctly separate the three groups.

### Real-data smoke test (marked `@pytest.mark.slow`)

18. After running `preprocess` on full ATLAS v2.0:
    - Total patient count within ±5% of documented release size.
    - Small/medium/large strata each contain >= 100 patients.
    - `validate_h5()` returns zero errors.
    - File size < 20 GB.

## Expected Runtime and Storage

### ATLAS v2.0 (single modality, ~955 subjects)

- **Preprocessing wall-clock:** 30–50 min on a single Picasso node
  (dominated by NIfTI decode + resampling).
- **HDF5 size (full volumes, gzip-4):** estimated 8–15 GB.
- **HDF5 size (lesion-only slices, if generated separately):** estimated
  2–5 GB.

### BraTS 2024 (4 modalities, ~1200 subjects)

- **HDF5 size (full volumes, gzip-4):** estimated 30–60 GB.
- **Note:** Multi-modal datasets are significantly larger. Consider storing
  on Picasso's scratch filesystem and using `--compression=blosc_lz4` for
  faster decode at training time.

## Migration from Previous Design

The previous design stored lesion-only 2D slices in `atlas_2d.h5`. The
new format stores complete 3D volumes. Key differences:

| Aspect | Previous (`atlas_2d.h5`) | New (`{dataset}.h5`) |
|--------|--------------------------|----------------------|
| Storage | Lesion-only 2D slices | Complete 3D volumes |
| Datasets | ATLAS only | Any (via converters) |
| Modalities | 1 (T1w) | Variable (1–4+) |
| Label classes | 1 (binary) | Variable (1–3+) |
| 2D loading | Direct indexing | Via `/slices/` + `has_lesion` filter |
| 3D loading | Not possible | Via `/volume_index/` |
| File size | ~3–8 GB | ~8–15 GB (ATLAS) |
| Splits | External JSON | In HDF5 + external JSON |

The DataModule API change:
```python
# Old
AtlasSliceDataModule(cache_root, fold, batch_size, ...)

# New
SegmentationDataModule(h5_path, fold, spatial_mode="2d", batch_size, ...)
```

## References

- Liew et al. *A large, curated, open-source stroke neuroimaging dataset
  (ATLAS v2.0)*. Scientific Data 2022. doi:10.1038/s41597-022-01401-7.
- Menze et al. *The Multimodal Brain Tumor Image Segmentation Benchmark
  (BRATS)*. IEEE TMI 2015. doi:10.1109/TMI.2014.2377694.
- Spitzer et al. *Interpretable surface-based detection of focal cortical
  dysplasias: a MELD study*. Brain 2022. doi:10.1093/brain/awac224.
- Roberts et al. *Common pitfalls and recommendations for using machine
  learning to detect and prognosticate for COVID-19 using chest radiographs
  and CT scans*. Nat. Mach. Intell. 2021. doi:10.1038/s42256-021-00307-0.
- MONAI Consortium. *MONAI: Medical Open Network for AI*.
  doi:10.5281/zenodo.6114127.
