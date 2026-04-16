# Phase 1 — Data Pipeline

## Goal

Turn the raw ATLAS v2.0 release (*Liew et al., 2022, doi:10.1038/s41597-022-01401-7*)
into a reproducible 2D-slice dataset with **patient-level** cross-validation
splits and a **volume-based stratification** of lesions into small / medium /
large cohorts. All downstream phases consume only the artefacts produced here.

Two design decisions distinguish this pipeline from a general-purpose
preprocessing stage:

1. **Lesion-only slices.** Only axial slices containing at least
   `min_lesion_voxels_per_slice` lesion voxels are retained. Background-only
   slices are discarded entirely. The experiment targets a mechanistic
   question — does Lp normalization with `p > 2` improve attention
   concentration on small lesions? — not deployment-ready segmentation.
   Including empty masks would dilute the training signal and exercise
   the attention mechanism on a regime irrelevant to the hypothesis.
2. **Single HDF5 file.** All retained slices are written to one
   `atlas_2d.h5` file instead of per-patient files. Picasso's parallel
   filesystem penalises large numbers of small files (inode overhead,
   metadata-server saturation under concurrent SLURM jobs). A single
   chunked, compressed HDF5 reduces the file count from O(n_patients) to
   O(1) and simplifies data transfers.

The deliverable is not a one-shot script — it is a set of composable,
typed modules under `src/lpqknorm/data/` plus a CLI entry point
`python -m lpqknorm.cli.preprocess` that materialises a deterministic slice
cache and split manifest from the raw NIfTI tree.

## Prerequisites

- ATLAS v2.0 downloaded and verified (`scripts/download_atlas.sh`).
- Conda environment installed, MONAI ≥ 1.3.0 available.
- No dependency on any other phase.

## Open questions the agent must resolve before coding

The agent should **inspect the downloaded tree** and confirm the following.
Do not assume — verify, and record the resolved values in
`src/lpqknorm/data/atlas.py` as module-level constants with citations.

1. The on-disk layout of ATLAS v2.0. The expected structure is
   `Training/sub-*/ses-*/anat/` containing `*_T1w.nii.gz` and
   `*_label-L_desc-T1lesion_mask.nii.gz`, but cohort suffixes have changed
   between releases. Confirm against the INDI release notes.
2. The intensity range and the skull-stripping status of the released T1w
   volumes. ATLAS v2.0 ships defaced and registered to MNI-152, already
   skull-stripped; re-verify by histogram inspection on 5 random subjects.
3. Lesion label polarity (foreground = 1 expected). Check a handful of masks.
4. Whether metadata CSV (`participants.tsv`) is present and usable for
   site/scanner stratification. If present, load it; if not, skip site
   stratification and document the decision.

## I/O contract

### Inputs

- `data.raw_root: Path` — ATLAS v2.0 root after unzip.
- `data.cache_root: Path` — destination for processed artefacts.
- `data.slice_axis: Literal["axial"]` — fixed to axial for this study.
- `data.min_lesion_voxels_per_slice: int = 10` — slice is kept only if its
  in-plane lesion voxel count ≥ threshold. Prevents pathological "1-voxel"
  slices from biasing training.
- `data.resample_spacing_mm: tuple[float, float] = (1.0, 1.0)` — in-plane
  resampling to 1 mm isotropic. ATLAS is MNI-registered so this is close to
  native spacing; still resample for safety.
- `data.patch_size: tuple[int, int] = (224, 224)` — MONAI SwinUNETR works
  well at this size; padded or cropped centrally.
- `data.seed: int = 20260216` — global seed used for splits and stratification.

### Outputs written to `cache_root`

- `atlas_2d.h5` — **single HDF5 file** containing every retained
  lesion-bearing slice across all patients:
  - `/images` `(N_total, 1, 224, 224)` float32, z-score normalised per
    volume using non-zero voxels of the full 3D volume.
  - `/masks` `(N_total, 1, 224, 224)` uint8, binary.
  - `/slice_indices` `(N_total,)` int32, original axial index into the
    3D volume.
  - `/subject_ids` `(N_total,)` variable-length UTF-8 string.
  - HDF5 dataset options:
    - Chunking: `(1, 1, 224, 224)` for images/masks (single-slice random
      access for DataLoader).
    - Compression: gzip level 4 (good ratio, fast decode; blosc is an
      acceptable alternative if h5py is compiled with it).
  - File-level attrs: `atlas_release`, `n_subjects`, `n_slices`,
    `patch_size`, `resample_spacing_mm`, `min_lesion_voxels_per_slice`,
    `lesion_only` (always `true`), `preprocessing_sha`.
- `manifest.parquet` — one row per kept slice, **row-order matching the
  H5 datasets** (row `i` in the parquet corresponds to index `i` in
  `/images`, `/masks`, etc.):
  `h5_index, subject_id, slice_index, in_slice_lesion_voxels,
   lesion_volume_mm3, volume_stratum, cohort, site`.
- `splits/kfold_patient_k3_seed{seed}.json` — list of 3 folds; each fold has
  `train_subjects`, `val_subjects`, `test_subjects` (disjoint patient IDs).
  Stratified on `volume_stratum` so each fold has roughly balanced small /
  medium / large patients.
- `strata.parquet` — per-patient stratum assignment with cutoff values.

### Exposed Python API (`src/lpqknorm/data/`)

All public functions typed, dataclass-configured, and raising
`lpqknorm.utils.exceptions.DataIntegrityError` on invariant violations.

```python
# atlas.py
@dataclass(frozen=True)
class AtlasSubject:
    subject_id: str
    t1w_path: Path
    mask_path: Path
    cohort: str
    site: str | None

def discover_subjects(raw_root: Path) -> list[AtlasSubject]: ...

# preprocessing.py
@dataclass(frozen=True)
class SliceExtractionConfig:
    min_lesion_voxels_per_slice: int
    resample_spacing_mm: tuple[float, float]
    patch_size: tuple[int, int]

@dataclass(frozen=True)
class SubjectSliceData:
    """In-memory slices for one subject (not yet written to disk)."""
    subject_id: str
    images: np.ndarray          # (n_slices, 1, H, W) float32
    masks: np.ndarray           # (n_slices, 1, H, W) uint8
    slice_indices: np.ndarray   # (n_slices,) int32
    lesion_voxels_per_slice: np.ndarray  # (n_slices,) int32
    lesion_volume_mm3: float
    cohort: str
    site: str | None

def extract_subject_slices(
    subject: AtlasSubject,
    cfg: SliceExtractionConfig,
) -> SubjectSliceData | None:
    """Extract lesion-bearing 2D slices from one subject.

    Returns in-memory arrays (not written to disk). Returns None if the
    subject has no slices meeting the threshold. The caller accumulates
    results across subjects and passes them to ``write_dataset_h5``.
    """
    ...

def write_dataset_h5(
    all_slices: list[SubjectSliceData],
    out_path: Path,
    cfg: SliceExtractionConfig,
) -> int:
    """Write all accumulated slices to a single chunked, compressed HDF5.

    Returns the total number of slices written.
    """
    ...

# stratification.py
def compute_lesion_volumes(subjects: list[AtlasSubject]) -> pd.DataFrame: ...
def assign_strata(volumes: pd.DataFrame, n_strata: int = 3) -> pd.DataFrame: ...

# splits.py
def make_patient_kfold(
    strata: pd.DataFrame, n_folds: int = 3, seed: int = 20260216
) -> list[FoldSpec]: ...

# datamodule.py
class AtlasSliceDataModule(pytorch_lightning.LightningDataModule):
    """Reads slices from ``atlas_2d.h5`` via the manifest + split files.

    Uses the manifest to resolve which H5 rows belong to each fold
    partition. Each ``__getitem__`` indexes into the H5 datasets by row.
    """
    def __init__(self, cache_root: Path, fold: int, batch_size: int,
                 num_workers: int, augment: bool) -> None: ...
```

## Design notes

**Why patient-level splits.** Slices from the same patient share scanner,
intensity distribution, and lesion morphology; mixing them across splits
inflates metrics (*cf. Roberts et al., Nat. Mach. Intell. 2021,
doi:10.1038/s42256-021-00307-0*). The split function must refuse to produce a
fold where any subject ID appears in more than one partition — enforce with
an explicit assertion and a dedicated test.

**Why pre-compute slices instead of extracting on-the-fly.** Three reasons:
determinism (training re-runs the same pixels), I/O efficiency on Picasso
(a single chunked HDF5 beats per-sample 3D NIfTI decode), and separability
of preprocessing bugs from training bugs.

**Why a single HDF5 file instead of per-patient files.** Picasso's parallel
filesystem (Lustre) penalises large numbers of small files: each file incurs
inode overhead and metadata operations that saturate the MDS under concurrent
SLURM jobs. A single chunked HDF5 with gzip compression reduces the file
count from O(n_patients) to O(1), eliminates per-file open/close overhead in
the DataLoader, and simplifies data transfers (`rsync` of one file instead of
thousands). Random access by slice index remains efficient because each slice
is one HDF5 chunk. The companion `manifest.parquet` serves as the index —
the DataModule filters it by fold/partition, then indexes into the H5 by row.

**Why lesion-only slices.** The experiment targets a mechanistic question —
does Lp normalisation with `p > 2` improve attention concentration on small
lesions? — not deployment-ready segmentation. Including background-only
slices (where there is no lesion to attend to) would dilute the training
signal and exercise the attention mechanism on a regime irrelevant to the
hypothesis. Every training sample must contain a non-trivial segmentation
target so that the Lp effect on lesion-attending attention is exercised at
every step. **Caveat:** this means we cannot report false-positive rates on
background-only slices. The paper must note this as a deliberate design
choice and scope limitation.

**Volume stratification.** Stratum boundaries are the 33rd and 66th
percentiles of per-patient lesion volume (`mm³`, computed in the original
scanner space before resampling). Record cutoffs in `strata.parquet` so they
can be re-used at analysis time without recomputation. The **small stratum**
is the headline evaluation cohort.

**Augmentations** (applied in `transforms.py`, train-time only):

- `RandAffined(rotate=0.2rad, scale=(0.9, 1.1), translate=10px)` — MONAI.
- `RandGaussianNoised(std=0.01)`.
- `RandBiasFieldd(coeff_range=(0.0, 0.1))`.
- No intensity inversion, no elastic deformation (stroke morphology matters).

Validation and test transforms are deterministic (resize + normalise only).

## Implementation checklist

1. `utils/exceptions.py` — define `DataIntegrityError`, `SplitLeakageError`,
   `StratificationError`.
2. `utils/seeding.py` — `set_global_seed(seed)` covering Python, NumPy, Torch,
   Torch CUDA, `PYTHONHASHSEED`, and dataloader worker init.
3. `data/atlas.py` — `discover_subjects` glob + regex based, with schema
   validation. Log total patient count; fail if < 500 (ATLAS v2.0 should have
   ≥ 900).
4. `data/preprocessing.py` — per-subject slice extraction returning
   in-memory `SubjectSliceData` (no disk write), plus `write_dataset_h5`
   that writes the single `atlas_2d.h5` from the accumulated list. Use
   `monai.transforms.LoadImaged`, `Spacingd`,
   `NormalizeIntensityd(nonzero=True)`, `Resized` to patch_size. Only
   retain slices with `in_slice_lesion_voxels >= min_lesion_voxels_per_slice`.
5. `data/stratification.py` — lesion-volume computation (count of non-zero
   voxels × voxel volume), percentile-based strata. Plot histogram as a
   side-effect (saved to `cache_root/qc/lesion_volume_hist.png`).
6. `data/splits.py` — 3-fold stratified k-fold at the **patient** level.
   Deterministic given the seed; write the split hash into the manifest.
7. `data/transforms.py` — train/val/test transform compositions.
8. `data/datamodule.py` — Lightning DataModule consuming `atlas_2d.h5` via
   `manifest.parquet` and a fold index. Resolves which H5 rows belong to
   each partition by filtering the manifest on the split's subject IDs.
9. `cli/preprocess.py` — Hydra entry point that runs the whole preprocessing
   pipeline: discover subjects → extract per-subject slices (in memory) →
   write single `atlas_2d.h5` → compute strata → generate splits → write
   `manifest.parquet` and `strata.parquet` → emit `_SUCCESS` file.

## Things to leave for agent inspection / decision

- Exact ATLAS v2.0 directory regex. Discover, then hard-code with a comment.
- Whether to include the "R" (hidden test) partition of ATLAS. Default **no**:
  it ships without labels and is reserved for challenge submissions.
- MNI template location — MONAI `SpatialResampled` handles this with the
  reference image, no external template required.

## Acceptance tests (`tests/unit/test_splits.py`, `test_stratification.py`)

All must pass under `pytest -q`. Tests use the synthetic fixture in
`tests/fixtures/synthetic_atlas.py` (5 patients, deterministic lesion
volumes) unless otherwise noted.

1. **No patient leakage.**

   ```python
   def test_no_patient_leakage(synthetic_strata):
       folds = make_patient_kfold(synthetic_strata, n_folds=3, seed=0)
       for f in folds:
           assert set(f.train) & set(f.val) == set()
           assert set(f.train) & set(f.test) == set()
           assert set(f.val) & set(f.test) == set()
       all_subjects = set().union(*(f.test for f in folds))
       assert all_subjects == set(synthetic_strata["subject_id"])
   ```

2. **Determinism.** Running `make_patient_kfold` twice with the same seed
   yields identical fold assignments (set equality on each partition).

3. **Stratum balance.** Each fold's train partition has at least one patient
   from each stratum; |train ∩ small| / |train| ∈ [0.25, 0.40].

4. **Slice extraction preserves lesion voxels.** For a synthetic volume with
   known lesion voxels, the sum of `in_slice_lesion_voxels` across kept
   slices equals the total lesion voxel count in slices that pass the
   threshold.

5. **Lesion-only and threshold respected.** Every slice in the output
   manifest has `in_slice_lesion_voxels >= min_lesion_voxels_per_slice > 0`.
   No background-only slices exist in the dataset.

6. **HDF5 integrity.** `atlas_2d.h5` exists, has datasets `/images`,
   `/masks`, `/slice_indices`, `/subject_ids` with shape `(N, ...)` where
   `N == len(manifest)`. Dtypes match the spec (float32, uint8, int32,
   UTF-8 string). For 10 random rows, the `subject_id` and `slice_index`
   in the H5 match the corresponding manifest row. Chunking is
   `(1, 1, 224, 224)` for image/mask datasets.

7. **Stratification monotonicity.** Sorting patients by lesion volume and
   re-assigning strata yields the same partition as the function.

8. **Real-data smoke test** (marked `@pytest.mark.slow`, run manually once):
   after running `preprocess` on the full ATLAS v2.0, assert that the total
   patient count is within ±5 % of the documented release size and that the
   small / medium / large cohorts each contain ≥ 100 patients.

## Expected runtime

On a single Picasso node: ≈ 25–40 min for the full ATLAS v2.0 preprocessing,
dominated by NIfTI decode + resampling. Estimated output size: with
lesion-only retention and gzip compression, `atlas_2d.h5` should be
≈ 3–8 GB (substantially smaller than retaining all slices, and a single
file on Picasso's filesystem instead of ~1000 per-patient files).

## References

- Liew et al. *A large, curated, open-source stroke neuroimaging dataset
  (ATLAS v2.0)*. Scientific Data 2022. doi:10.1038/s41597-022-01401-7.
- Roberts et al. *Common pitfalls and recommendations for using machine
  learning to detect and prognosticate for COVID-19 using chest radiographs
  and CT scans*. Nat. Mach. Intell. 2021. doi:10.1038/s42256-021-00307-0.
- MONAI Consortium. *MONAI: Medical Open Network for AI*.
  doi:10.5281/zenodo.6114127.
