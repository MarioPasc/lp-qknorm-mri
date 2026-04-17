# Lp-QKNorm for Small-Lesion Segmentation — Agent Instructions

## Project Identity

This repository implements a **mechanistic study** of generalized Lp query-key
normalization inside windowed self-attention for medical image segmentation,
with a focus on small-lesion detection. It is **not** a new segmentation
method — it is a controlled experiment isolating the effect of the Lp norm
parameter `p` on attention concentration for small lesions.

The pipeline supports **multiple datasets** (ATLAS stroke, BraTS glioma,
MELD epilepsy, meningioma) via a standardized HDF5 format with
dataset-specific converters, and **both 2D and 3D** training from the same
preprocessed file.

- **Author:** Mario Pascual González (Health Engineering / Bioinformatics, UMA)
- **Research group:** Grupo de Inteligencia Computacional y Análisis de Imagen (GIC-AIA), UMA
- **Clinical collaborator:** IBIMA-BIONAND
- **License:** MIT
- **Primary dataset:** ATLAS v2.0 (Liew et al., Scientific Data 2022)
- **Architecture:** Swin-UNETR (MONAI, 2D or 3D) with patched windowed attention

## Scientific Hypothesis

For lesions whose query/key representations develop peaky coordinate
distributions, higher `p` increases the logit gap between lesion-aligned and
background keys. The predicted mechanistic chain is:

> higher feature peakiness (Probe 1) → lower per-query attention entropy
> (Probe 2) → more attention mass on lesion tokens (Probe 3) → larger logit
> gap (Probe 4) → tighter attention–mask spatial alignment (Probe 5) → higher
> small-lesion recall (headline metric).

The toy-model prediction is `Δ(p) = s^{1 − 2/p}[1 − (s/d_k)^{1/p}]` with an
interior maximum at `p* > 2`.

### Critical Constraint

This is a **representation/normalization study**, not a new architecture. The
only modification to stock Swin-UNETR is the Q-K normalization inside
`WindowAttention`. Everything else (patch embedding, relative position bias,
skip connections, decoder) must remain architecturally identical. Any code or
text that frames this as "a novel segmentation architecture" violates the
experimental design.

## Key References

- Henry et al. *Query-Key Normalization for Transformers*. EMNLP 2020. arXiv:2010.04245.
- López-Rubio et al. *Enhanced QKNorm with the Lp Norm*. 2026. arXiv:2602.05006.
- Liu et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
- Hatamizadeh et al. *Swin UNETR*. BrainLes 2021. arXiv:2201.01266.
- Cao et al. *Swin-Unet*. ECCVW 2022. arXiv:2105.05537.
- Liew et al. *ATLAS v2.0*. Scientific Data 2022. doi:10.1038/s41597-022-01401-7.

## Environment

- **Conda env:** `lpqknorm` (`~/.conda/envs/lpqknorm`)
- **Python:** 3.11
- **Project root:** `/home/mpascual/research/code/lp-qknorm-mri`
- **Results directory:** `/media/mpascual/Sandisk2TB/research/lpqknorm_mri/results/`
- **HPC:** Picasso (SLURM + Singularity, no Docker)
- **Local GPU:** RTX 4060

### Running commands

Always prefix commands with the conda environment python:

```bash
~/.conda/envs/lpqknorm/bin/python -m pytest tests/ -v --tb=short
~/.conda/envs/lpqknorm/bin/python -m ruff check src/ tests/
~/.conda/envs/lpqknorm/bin/python -m ruff format src/ tests/
~/.conda/envs/lpqknorm/bin/python -m mypy src/lpqknorm/
```

## Repository Layout

```
src/lpqknorm/           # Main package (editable install)
  data/                 # Phase 1: Standardized HDF5 format, converters, DataModule
    schema.py           #   HDF5 format spec (DatasetHeader, validate_h5)
    converter.py        #   Abstract converter protocol + generic writer
    converters/         #   Dataset-specific converters (atlas, brats, meld, ...)
    splits.py           #   Patient-level stratified k-fold
    stratification.py   #   Volume-based lesion strata
    transforms.py       #   MONAI transforms (2D + 3D)
    datamodule.py       #   Generic dual-mode DataModule (2D/3D)
  models/               # Phase 2: LpQKNorm, LpWindowAttention, patched SwinUNETR
  training/             # Phase 3: LightningModule, losses, metrics, callbacks
  probes/               # Phase 4: Five mechanistic probes + recorder
  analysis/             # Phase 5: Bootstrap, effect sizes, figures
  utils/                # Exceptions, seeding, I/O, git, logging
  cli/                  # Hydra entry points (train, probe, analyze, preprocess)

configs/                # Hydra configuration hierarchy
scripts/                # SLURM submission, download, environment verification
tests/unit/             # Isolated module tests
tests/integration/      # End-to-end pipeline tests
tests/fixtures/         # Synthetic dataset (5 patients)
docs/                   # Phase specifications (read-only reference)
```

## Development Phases (Sequential)

Work through phases in order. **Do not start phase N+1 until phase N's
acceptance tests pass.**

1. **Phase 1 — Data Pipeline** (`docs/phase_01_data.md`)
   Deliverables: `data/schema.py`, `data/converter.py`, `data/converters/atlas.py`,
   `data/datamodule.py`, `cli/preprocess.py`, `utils/exceptions.py`, `utils/seeding.py`
   Output: single `{dataset}.h5` (standardized HDF5, complete 3D volumes,
   self-describing header) + patient-level splits + strata inside the H5.
   **Multi-dataset:** converter architecture; ATLAS first, others added later.
   **Dual-mode:** same file supports 2D slice and 3D volume loading.
   Tests: `test_splits.py`, `test_stratification.py`, `test_schema.py`

2. **Phase 2 — Model** (`docs/phase_02_model.md`)
   Deliverables: `models/lp_qknorm.py`, `models/attention.py`,
   `models/swin_unetr_lp.py`, `models/hooks.py`, `models/init.py`
   Supports `spatial_dims=2` and `spatial_dims=3`; reads `in_channels` and
   `out_channels` from the HDF5 header for dataset-agnostic configuration.
   **From-scratch weight init** (`init_scheme="scratch_trunc_normal"`,
   default) applies `trunc_normal_(std=0.02)` to every `nn.Linear`,
   `nn.Conv{2,3}d`, and `relative_position_bias_table`, sets `nn.LayerNorm`
   to `(ones, zeros)`, and seeds `LpQKNorm.alpha_raw = softplus_inverse(log d_k)`
   per stage (Henry et al., 2020).  Pretrained Swin-UNETR weights are
   deferred to a single ablation row (`init_scheme="pretrained_ssl"`) to
   avoid biasing Q/K geometry toward the `p=2` (ℓ₂) regime.
   Tests: `test_lp_qknorm.py`, `test_attention_equivalence.py`,
   `test_forward_pass.py`, `test_init.py`.
   **Critical test:** `p = 2` must recover original QKNorm exactly.

3. **Phase 3 — Training** (`docs/phase_03_training.md`)
   Deliverables: `training/`, `cli/train.py`, `scripts/submit_sweep.sbatch`
   Dataset-agnostic: reads HDF5 header to configure model and loss.
   Tests: `test_training_step.py`, `test_resume.py`

4. **Phase 4 — Probes** (`docs/phase_04_probes.md`)  ✅ **complete**
   Deliverables: `probes/` (Probes 1–8 + `attention_maps.py` + `patching.py` +
   `tokenization.py` with `window_boundary_distance`), `cli/probe.py`,
   `cli/patching.py`, `training/callbacks.py::AlphaLogger`,
   `training/callbacks.py::PatchingCallback`.
   Output schema: one `probes/epoch_{N}.h5` with `/metadata`, `/inputs`,
   `/block_0_wmsa`, `/block_1_swmsa` per checkpoint, plus
   `probes/patching_best_dice.h5` (five variants × two blocks, denoising +
   noising) and `probes/alpha_trajectory.jsonl` (per-step α).
   Tests (all green): `test_probes_synthetic.py`, `test_tokenization.py`,
   `test_attention_maps.py`, `test_patching.py`, `test_probe_pipeline.py`
   (AT1–AT11).

5. **Phase 5 — Analysis** (`docs/phase_05_analysis.md`)
   Deliverables: `analysis/`, `cli/analyze.py`
   Tests: `test_bootstrap.py`

## Mathematical Specifications (Quick Reference)

### Lp-QKNorm normalization

```
q̂_i^(p) = q_i / (||q_i||_p + ε)
k̂_j^(p) = k_j / (||k_j||_p + ε)
s_ij^(p) = α · ⟨q̂_i^(p), k̂_j^(p)⟩
A = softmax(S^(p) + B_rel)
```

- `α = softplus(α_raw)` (learnable positive scalar)
- `ε = 1e-6`
- Numerically stable form for `p ≥ 2`: `||v||_p = (Σ_h |v_h|^p + ε)^(1/p)`
- Numerically stable form for `p < 2`: `||v||_p = (Σ_h (|v_h| + ε)^p)^(1/p)`
- `p` is a buffer (not a learnable parameter)

### Five mechanistic probes (on stage-1 attention)

| # | Probe | Formula | Range | Predicted p-trend |
|---|-------|---------|-------|-------------------|
| 1 | Peakiness | `ρ = \|\|v\|\|_∞ / \|\|v\|\|_2` | `[1/√d_k, 1]` | ↑ for lesion tokens |
| 2 | Entropy | `H = -Σ A_ij log A_ij` | `[0, log W²]` | ↓ on lesion queries |
| 3 | Lesion mass | `M = Σ_{j∈L} A_ij` | `[0, 1]` | ↑ on small stratum |
| 4 | Logit gap | `Δ = max_L s_ij - med_B s_ij` | `ℝ` | interior max at p* |
| 5 | Attn-mask IoU | top-k binarized attn vs mask | `[0, 1]` | ↑ with p |

### Experimental design

- **Sweep:** `p ∈ {vanilla, 2.0, 2.5, 3.0, 3.5, 4.0}` × `fold ∈ {0, 1, 2}` = 18 runs
- **Vanilla baseline:** stock MONAI `WindowAttention` (no QKNorm) — lower-bound control
- **Primary baseline:** `p = 2.0` (original QKNorm of Henry et al.)
- **Headline metric:** lesion-wise recall on the small-lesion stratum (volume < 33rd percentile)
- **Statistical test:** paired patient-level bootstrap (B = 10000), Cohen's d, Holm-Bonferroni (5 comparisons vs p=2)
- **Probes:** stage-1 windowed attention only, on a fixed 32-slice validation batch
- **Figure 1 (theory):** toy-model `Δ(p)` curves — can be generated before any experiments

## Invariants (Never Violate)

1. **Patient-level splits.** No patient ID may appear in more than one partition
   (train/val/test) within a fold. Enforce with assertions and a dedicated test.
2. **p=2 equivalence.** `LpQKNorm(p=2)` must be numerically identical to
   Henry et al.'s original QKNorm. This is the single most important test.
3. **Architecture preservation.** Only `WindowAttention.forward` is modified.
   All other SwinUNETR components are stock MONAI.
4. **Deterministic probes.** Probe batch is fixed across epochs and runs. No
   augmentation during probing. Two identical calls produce identical results.
5. **Log everything.** If it is computed during training/validation, it is
   logged. Re-training to collect a missing metric is unacceptable.
6. **Relative position bias preserved.** The bias is added after the QK dot
   product, not absorbed into Q/K. Removing it conflates two effects.
7. **Self-describing HDF5.** Each `{dataset}.h5` stores complete 3D volumes
   with a root-level header containing dataset name, label names, modalities,
   and all preprocessing parameters. Lesion-only filtering for 2D training
   happens at DataModule load time via the `/slices/has_lesion` flag — not
   at storage time.
8. **Single H5 per dataset.** All preprocessed volumes for one dataset live
   in one `{dataset_name}.h5`. No per-patient files. The `/slices/` manifest
   row order matches `/data/` row order.
9. **Dataset-agnostic downstream code.** Training, probes, and analysis code
   must never reference a specific dataset (e.g., "atlas") — they read the
   HDF5 header (`DatasetHeader.from_h5()`) to discover `n_modalities`,
   `n_label_classes`, `dataset_name`, etc. Only converter code is
   dataset-specific.
10. **Init spec identity across the sweep.** `init_scheme`, `linear_init_std`,
    `alpha_init_scheme`, and `alpha_init_fixed` must be identical across all
    `p` values and all folds of a primary sweep. Drift in any of these
    confounds the `p` effect. The training CLI hashes these fields into
    `manifest.json::init_spec_hash`; the sweep aggregator must assert
    equality across runs grouped by `experiment`.
11. **Controlled init RNG consumption.** `build_swin_unetr_lp` calls
    `initialize_model` after patching, which re-initializes every `nn.Linear`,
    `nn.Conv{2,3}d`, `nn.LayerNorm`, and `relative_position_bias_table` via a
    single `model.apply`.  At a fixed seed, all non-`alpha_raw` tensors must
    be byte-identical across `p` values; `alpha_raw` may differ only if the
    scheme makes it depend on `d_k` (it does not under `log_dk` with
    `feature_size=24`, since `d_k=8` at every stage).

## Code Conventions

- **Python 3.11+** syntax: `X | None`, `list[int]`, `tuple[str, ...]`.
- **Type hints** on every function signature and return type.
- **Docstrings:** NumPy-style with `Parameters`, `Returns`, `Raises`.
- **Config objects:** `@dataclass(frozen=True)` or Pydantic `BaseModel`.
- **Exceptions:** Custom hierarchy per module under `utils/exceptions.py`.
  At minimum: `LpQKNormError` (base), `DataIntegrityError`, `SplitLeakageError`,
  `StratificationError`.
- **Logging:** `logging` module with structured output. Never bare `print()` in
  library code (`print` is acceptable only in scripts and notebooks).
- **GPU memory:** Explicit `.detach()`, `torch.no_grad()`, `torch.cuda.empty_cache()`.
  Use context managers for autocast control.
- **Testing:** `pytest`, fixtures in `conftest.py`, parametrize edge cases.
  Scientific assertions: `np.testing.assert_allclose(rtol=...)` or
  `torch.testing.assert_close`. Separate unit/integration via
  `@pytest.mark.integration`.
- **Test file naming:** `test_<module>.py`, co-located in `tests/` mirroring `src/`.
- **No bare dicts** for configuration that crosses function boundaries.

## Verification Commands

After any code change, run the full pipeline:

```bash
# 1. Unit tests
~/.conda/envs/lpqknorm/bin/python -m pytest tests/unit/ -v --tb=short

# 2. Integration tests (optional, slower)
~/.conda/envs/lpqknorm/bin/python -m pytest tests/integration/ -v --tb=short

# 3. Linter (auto-fix)
~/.conda/envs/lpqknorm/bin/python -m ruff check --fix src/ tests/
~/.conda/envs/lpqknorm/bin/python -m ruff format src/ tests/

# 4. Type checker
~/.conda/envs/lpqknorm/bin/python -m mypy src/lpqknorm/
```

## Git Conventions

- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- Branch naming: `feature/<short-description>`, `fix/<issue-ref>`.
- Never commit large binary files (models, datasets, HDF5 caches).
- Imperative subject line, max 72 chars.

## Open Questions (Resolve Before Coding Each Phase)

Each phase doc lists open questions the agent must resolve by inspecting the
actual environment (MONAI source, ATLAS v2.0 directory structure, etc.) before
writing code. Do not assume — verify, and record resolved values as module-level
constants with citations. See `docs/phase_01_data.md` through
`docs/phase_05_analysis.md` for the full list.

## What Success Looks Like

> Small-stratum lesion-wise recall at `p = p*` exceeds that at `p = 2` with a
> 95% bootstrap CI excluding zero, a paired Cohen's d >= 0.3, and per-patient
> correlation with at least two of the five probes in the theoretically
> predicted direction (positive for peakiness/lesion mass/attention IoU,
> negative for entropy) at |r| >= 0.2.

If both the headline and probe chain hold, we have a mechanistically
interpretable result. If only one holds, the paper is weaker but still
publishable. Either outcome is informative.
