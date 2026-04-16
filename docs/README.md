# Lp-QKNorm for Small-Lesion Segmentation (ATLAS v2.0)

Mechanistic study of generalized Lp query–key normalization inside windowed
self-attention for 2D stroke-lesion segmentation. Extends the QKNorm scheme
of Henry et al. (2020, arXiv:2010.04245) from `p = 2` (Euclidean) to a sweep
`p ∈ {2.0, 2.5, 3.0, 3.5, 4.0}`, following the generalization proposed by
López-Rubio et al. (2026, arXiv:2602.05006), and tests the hypothesis that
`p > 2` improves attention concentration on small lesions. A vanilla softmax
baseline (no QKNorm) is included to contextualise the Lp improvement.

## Scientific hypothesis

For lesions whose query/key representations develop peaky coordinate
distributions, higher `p` increases the logit gap between lesion-aligned and
background keys (the toy-model prediction `Δ(p) = s^{1 − 2/p}[1 − (s/d_k)^{1/p}]`
has an interior maximum for `p* > 2`). The mechanistic consequence is a chain
of five observable quantities: higher feature peakiness → lower per-query
attention entropy → more attention mass on lesion tokens → larger logit gap →
tighter spatial alignment between attention and lesion mask. The downstream
consequence is higher lesion-wise recall on the small-lesion stratum.

## Repository layout

```
lp-qknorm-mri/
├── README.md                         # this file
├── LICENSE                           # MIT
├── pyproject.toml                    # project metadata, deps, ruff/mypy config
├── environment.yml                   # conda env (Picasso-friendly)
├── .gitignore
├── .pre-commit-config.yaml           # ruff, mypy, nbstripout
│
├── configs/                          # Hydra configs
│   ├── config.yaml                   # top-level
│   ├── data/atlas_2d.yaml
│   ├── model/swin_unetr_2d.yaml
│   ├── model/lp_qknorm.yaml          # p, learnable alpha, eps
│   ├── training/default.yaml
│   ├── probes/default.yaml
│   └── experiment/p_sweep.yaml       # multirun spec
│
├── docs/                             # agent-facing phase specs
│   ├── phase_01_data.md
│   ├── phase_02_model.md
│   ├── phase_03_training.md
│   ├── phase_04_probes.md
│   └── phase_05_analysis.md
│
├── src/lpqknorm/
│   ├── __init__.py                   # exports public API, __version__
│   │
│   ├── data/                         # Phase 1
│   │   ├── __init__.py
│   │   ├── atlas.py                  # ATLAS v2.0 discovery + metadata
│   │   ├── preprocessing.py          # 2D slice extraction, intensity norm
│   │   ├── splits.py                 # patient-level K-fold with stratification
│   │   ├── stratification.py         # volume-based lesion strata
│   │   ├── transforms.py             # MONAI transform compositions
│   │   └── datamodule.py             # Lightning DataModule
│   │
│   ├── models/                       # Phase 2
│   │   ├── __init__.py
│   │   ├── lp_qknorm.py              # LpQKNorm module + helpers
│   │   ├── attention.py              # LpWindowAttention (drop-in)
│   │   ├── swin_unetr_lp.py          # MONAI SwinUNETR with patched attn
│   │   └── hooks.py                  # forward-hook registry (stage-1)
│   │
│   ├── training/                     # Phase 3
│   │   ├── __init__.py
│   │   ├── module.py                 # LightningModule
│   │   ├── losses.py                 # Dice + BCE compound loss
│   │   ├── metrics.py                # segmentation + lesion-wise metrics
│   │   ├── callbacks.py              # ProbeCallback, ArtifactCallback
│   │   └── logging.py                # JSONL + parquet structured logs
│   │
│   ├── probes/                       # Phase 4
│   │   ├── __init__.py
│   │   ├── base.py                   # Probe abstract base + registry
│   │   ├── peakiness.py              # ρ_p = ||v||_∞ / ||v||_2
│   │   ├── entropy.py                # per-query attention entropy
│   │   ├── lesion_mass.py            # M_i = Σ_{j∈L} A_ij
│   │   ├── logit_gap.py              # Δ_i between lesion and background keys
│   │   ├── attention_iou.py          # top-k attention vs. lesion mask IoU
│   │   └── recorder.py               # single orchestrator, saves to HDF5
│   │
│   ├── analysis/                     # Phase 5
│   │   ├── __init__.py
│   │   ├── stratification.py         # reproduce strata from run manifests
│   │   ├── bootstrap.py              # paired bootstrap (patient-level)
│   │   ├── effect_size.py            # paired Cohen's d, CI
│   │   ├── aggregation.py            # cross-fold, cross-p aggregation
│   │   ├── probe_curves.py           # p-dependent probe trajectories
│   │   └── figures.py                # publication figures
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── exceptions.py             # LpQKNormError, DataIntegrityError, ...
│   │   ├── seeding.py                # global + per-worker determinism
│   │   ├── io.py                     # HDF5, parquet, JSONL helpers
│   │   ├── git.py                    # capture SHA + dirty flag
│   │   └── logging.py                # structlog config
│   │
│   └── cli/
│       ├── __init__.py
│       ├── train.py                  # python -m lpqknorm.cli.train
│       ├── probe.py                  # post-hoc probe extraction
│       ├── analyze.py                # downstream analysis
│       └── preprocess.py             # one-shot ATLAS → 2D slice cache
│
├── scripts/
│   ├── download_atlas.sh             # INDI download + verification
│   ├── submit_sweep.sbatch           # Picasso SLURM array over (p, fold)
│   ├── submit_analysis.sbatch
│   └── verify_env.py                 # CUDA, MONAI, torch versions
│
├── tests/
│   ├── unit/
│   │   ├── test_lp_qknorm.py
│   │   ├── test_attention_equivalence.py   # p=2 ≡ original QKNorm
│   │   ├── test_splits.py                  # no patient leakage
│   │   ├── test_stratification.py
│   │   ├── test_metrics.py
│   │   ├── test_probes_synthetic.py        # theoretical predictions
│   │   └── test_bootstrap.py
│   ├── integration/
│   │   ├── test_forward_pass.py
│   │   ├── test_training_step.py
│   │   ├── test_resume.py
│   │   └── test_probe_pipeline.py
│   └── fixtures/
│       ├── __init__.py
│       ├── synthetic_atlas.py              # 5-patient fake cohort
│       └── tiny_config.yaml
│
│
```

```
/media/mpascual/Sandisk2TB/research/lpqknorm_mri/results/                          
  └── {run_id}/                     # one dir per (p, fold, seed)
      ├── config.yaml               # Hydra snapshot
      ├── manifest.json             # git SHA, env, split hash, timings
      ├── metrics/
      │   ├── train.jsonl           # per-step
      │   ├── val.jsonl             # per-epoch
      │   ├── val_per_patient.parquet
      │   └── test_per_patient.parquet
      ├── probes/
      │   └── epoch_{N}.h5          # all 5 probes, per-token
      ├── predictions/
      │   └── test_masks.h5
      └── checkpoints/
          ├── last.ckpt
          └── best_dice.ckpt
```

## Quick start

```bash
# 1. Environment
conda create -n lpqknorm python=3.11 -y
conda activate lpqknorm
pip install -e .

# 2. Data (Phase 1)
bash scripts/download_atlas.sh /path/to/raw
python -m lpqknorm.cli.preprocess data.raw_root=/path/to/raw

# 3. Train sweep (Phase 3) — locally or SLURM
python -m lpqknorm.cli.train -m \
    model.lp_qknorm.p=vanilla,2.0,2.5,3.0,3.5,4.0 \
    training.fold=0,1,2

# 4. Analyse (Phase 5)
python -m lpqknorm.cli.analyze results_root=results/
```

## Phase-by-phase instructions for the local agent

Work through `docs/phase_01_data.md` → ... → `docs/phase_05_analysis.md`
in order. Each phase ends with an explicit **acceptance test** runnable as
`pytest tests/unit/test_<phase>.py` and/or `pytest tests/integration/...`.
Do not proceed to phase `N+1` until phase `N`'s tests pass on CI and locally.

## References

- Henry, Dachapally, Pawar, Chen. *Query-Key Normalization for Transformers*.
  Findings of EMNLP 2020. arXiv:2010.04245.
- López-Rubio, Montes-Pérez, Palomo. *Enhanced QKNorm Normalization for Neural
  Transformers with the Lp Norm*. 2026. arXiv:2602.05006.
- Liu et al. *Swin Transformer*. ICCV 2021. arXiv:2103.14030.
- Hatamizadeh et al. *Swin UNETR: Swin Transformers for Semantic Segmentation
  of Brain Tumors in MRI Images*. BrainLes 2021. arXiv:2201.01266.
- Cao et al. *Swin-Unet: Unet-like Pure Transformer for Medical Image
  Segmentation*. ECCVW 2022. arXiv:2105.05537.
- Liew et al. *A large, curated, open-source stroke neuroimaging dataset
  (ATLAS v2.0)*. Scientific Data 2022. doi:10.1038/s41597-022-01401-7.
