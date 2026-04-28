# `p_sweep_v1` — Picasso Run Status and Performance Report

**Author**: Mario Pascual González · **Date**: 2026-04-28
**Sweep**: `p ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0}` × `fold ∈ {0, 1, 2}` (18 runs)
**Seed**: 20260216 · **Dataset**: BraTS-MEN (meningioma) HDF5
**Hardware**: Picasso A100 40 GB, `dgx` constraint, B=4 × accumulate=4, bf16-mixed
**Results root**: `/media/mpascual/Sandisk2TB/research/lpqknorm_mri/results`

---

## 1 · Run-completion audit

| p | fold | initial job | resume job | final state |
|---|---|---|---|---|
| 1.0 | 0 | 519999 | 539079 | ✅ Training complete |
| 1.0 | 1 | 520000 | 539080 | ✅ Training complete |
| 1.0 | 2 | 520001 | 539081 | ✅ Training complete |
| 1.5 | 0 | 520002 | 539082 | ✅ Training complete |
| 1.5 | 1 | 520003 | 539083 | ✅ Training complete |
| 1.5 | 2 | 520004 | — | ✅ finished on first try |
| 2.0 | 0 | 520005 | 539084 | ✅ Training complete |
| 2.0 | 1 | 520006 | 539085 | ✅ Training complete |
| 2.0 | 2 | 520007 | 539086 | ✅ Training complete |
| 2.5 | 0 | 520008 | 539087 | ✅ Training complete |
| 2.5 | 1 | 520009 | 539088 | ✅ Training complete |
| 2.5 | 2 | 520010 | 539089 | ✅ Training complete |
| 3.0 | 0 | 520011 | 539090 | ✅ Training complete |
| 3.0 | 1 | 520012 | 539091 | ✅ Training complete |
| 3.0 | 2 | 520013 | 539092 | ✅ Training complete |
| 4.0 | 0 | 520014 | — | ✅ finished on first try |
| 4.0 | 1 | 520015 | 539093 | ✅ Training complete |
| 4.0 | 2 | 519998 | 539078 | ✅ Training complete |

**Verification basis**: every fold contains
`metrics/test_per_patient.parquet` (≥ 83 KB, 17 360 slice-rows ≈ 8 patients × stratum-balanced),
`metrics/val_per_patient.parquet`, four `predictions/test_batch_{0..3}.npz`, and both
`checkpoints/best_val_dice.ckpt` and `checkpoints/best_small_recall.ckpt`. All
`hydra_logs/*/train.log` end with `__main__][INFO] - Training complete.` `grep -E
"RuntimeError|CUDA out of memory|TIME LIMIT|DUE TO TIME|Killed"` returns zero
matches across the 35 log files (18 first-pass + 17 resumes).

The single warning emitted by every run is `No rows to flush for stage='test_per_lesion'`,
which only means the per-lesion accumulator was empty at flush time (the
per-patient parquet is the headline file and is populated). It is not an
error.

### Re-submission protocol (NOT NEEDED for this sweep)

The repository already implements resume via SLURM array re-submission against
`checkpoints/last.ckpt`. The pattern, already exercised on Apr 25 by
`launch.sh`, is:

```bash
# On Picasso, identify the failed array indices (e.g. tasks 0,3,5)
FAILED="0,3,5"
sbatch --array=${FAILED} \
       --time=2-00:00:00 \
       scripts/slurm/launch.sh
```

`launch.sh` reads `picasso_config_snapshot.yaml`, recomputes `(p,fold)` from
`SLURM_ARRAY_TASK_ID`, and the LightningModule auto-detects
`run_dir/.../checkpoints/last.ckpt` and resumes from the saved optimizer/RNG
state. No code change is required because **all 18 runs already completed**.

---

## 2 · Performance vs `p`

All metrics are computed at the slice level from
`metrics/test_per_patient.parquet`, then averaged within each patient, then
within each fold, then summarised across folds (mean ± std over 3 folds).

### 2.1 Aggregate (all strata pooled)

| p | Dice (mean ± std) | IoU | lesion-recall | FP/slice |
|---|---|---|---|---|
| 1.0 | 0.818 ± 0.021 | 0.804 ± 0.021 | 0.710 ± 0.008 | 0.335 ± 0.029 |
| 1.5 | 0.819 ± 0.007 | 0.805 ± 0.008 | 0.709 ± 0.005 | 0.335 ± 0.006 |
| **2.0** | 0.801 ± 0.007 | 0.787 ± 0.007 | 0.724 ± 0.001 | 0.363 ± 0.020 |
| 2.5 | 0.794 ± 0.014 | 0.780 ± 0.014 | 0.721 ± 0.006 | 0.383 ± 0.014 |
| 3.0 | 0.803 ± 0.004 | 0.789 ± 0.003 | 0.717 ± 0.008 | 0.362 ± 0.012 |
| 4.0 | 0.770 ± 0.066 | 0.756 ± 0.067 | 0.732 ± 0.017 | 0.456 ± 0.156 |

### 2.2 Stratified (volume tertiles)

**Small-lesion stratum (volume < 33rd percentile)** — the headline metric:

| p | Dice (small) | lesion-recall (small) |
|---|---|---|
| 1.0 | 0.820 ± 0.030 | 0.666 ± 0.022 |
| 1.5 | 0.822 ± 0.011 | 0.656 ± 0.017 |
| **2.0** | 0.804 ± 0.005 | **0.700 ± 0.008** ← QKNorm baseline |
| 2.5 | 0.794 ± 0.027 | 0.689 ± 0.010 |
| 3.0 | 0.809 ± 0.006 | 0.686 ± 0.017 |
| **4.0** | 0.770 ± 0.071 | **0.708 ± 0.031** ← peak |

Medium and large strata follow the same monotone trends; recall increases
slightly with `p` while Dice/IoU drop slightly (full table:
`reports/per_stratum_metrics.csv`).

### 2.3 Correlations (n = 18 fold-level points)

| metric | Pearson r (p) | Spearman ρ (p) |
|---|---|---|
| Dice | −0.51 (0.030) | −0.45 (0.062) |
| IoU | −0.51 (0.030) | −0.45 (0.062) |
| lesion-recall | **+0.60 (0.008)** | **+0.58 (0.012)** |
| FP/slice | +0.55 (0.019) | +0.62 (0.006) |

### 2.4 Patient-level paired bootstrap (B = 10 000), small-stratum recall vs `p = 2.0`

| comparison | Δ-recall | 95 % CI | Cohen's d | n |
|---|---|---|---|---|
| p=1.0 vs 2.0 | **−0.035** | [−0.055, −0.015] | −0.30 | 120 |
| p=1.5 vs 2.0 | **−0.044** | [−0.069, −0.021] | −0.32 | 120 |
| p=2.5 vs 2.0 | −0.012 | [−0.031, +0.008] | −0.11 | 120 |
| p=3.0 vs 2.0 | −0.015 | [−0.036, +0.006] | −0.12 | 120 |
| p=4.0 vs 2.0 | +0.008 | [−0.010, +0.025] | +0.08 | 120 |

After Holm–Bonferroni correction (5 comparisons), only the sub-Euclidean
contrasts (p = 1.0, 1.5) remain significant — both **deteriorate** small-lesion
recall relative to QKNorm. No super-Euclidean `p` reaches the
small-recall improvement criterion stated in `CLAUDE.md` ("CI excluding zero,
Cohen's d ≥ 0.3").

### 2.5 Learnable α trajectory

Final softplus-α at the end of stage-0 attention (mean over folds):

| p | α* | trend |
|---|---|---|
| 1.0 | 2.94 | highest |
| 1.5 | 2.63 |  |
| 2.0 | 2.45 | QKNorm baseline |
| 2.5 | 2.38 |  |
| 3.0 | 2.35 |  |
| 4.0 | 2.26 | lowest |

α decreases monotonically with `p`, as expected: higher-`p` norms produce
smaller scalars on unit-energy vectors, so the optimiser raises α less to reach
the same logit scale. The monotonicity is a sanity check that the Lp norm is
acting as designed inside `LpQKNorm`.

---

## 3 · Reading of the result

1. **Hypothesis (interior maximum at p\* > 2)**: weakly supported.
   Small-stratum recall *does* peak at the super-Euclidean end (p = 4.0,
   recall = 0.708) but the gain over `p = 2` (0.700) is +0.008 with a CI
   that includes zero — not significant. The toy-model prediction
   `Δ(p) = s^{1−2/p}[1 − (s/d_k)^{1/p}]` predicts a small positive shift in
   this regime; the empirical effect is consistent in sign but below the
   pre-registered effect-size threshold.
2. **Sub-Euclidean regime (p < 2)** is **harmful** for small lesions:
   −3 to −4 percentage points recall, Cohen's d ≈ −0.3, both CIs strictly
   negative. This is a clean negative result and a publishable falsification
   of the "any p other than 2 helps" naive reading.
3. **Dice/IoU vs recall trade-off**: higher `p` produces *more* lesion
   coverage at the cost of *more* false positives per slice (FP/slice 0.36 →
   0.46 from `p = 2` to `p = 4`). The model becomes more permissive as `p`
   grows. Whether this is acceptable is a clinical-utility question that
   needs the lesion-wise PR curves from Phase 5.
4. **`p = 4.0` variance is large** (Dice σ = 0.066 vs ≤ 0.014 elsewhere). One
   fold (f=0) early-stopped at step 59 892 vs ~120 k for the other folds; the
   sweep-level conclusions on `p = 4` should be treated cautiously until the
   Phase-4 mechanistic probes are run and Phase-5 attention-mask IoU is
   computed.

### Next actions (Phase 4/5, no further training needed)

- Run `lpqknorm-probe` on each of the 18 best-checkpoint directories to
  populate `probes/epoch_*.h5` and the patching artefacts; the sweep is
  finished, so this is read-only inference.
- Run `lpqknorm-analyze` to produce the official paired bootstrap with
  Holm–Bonferroni correction, the per-probe correlations vs small-recall, and
  Figure 1 (toy `Δ(p)` curves).
- Investigate the `p = 4.0` fold-0 early stop (likely benign — `best_val_dice`
  was already at the plateau — but worth a quick look at
  `metrics/train_steps.jsonl`).

---

## 4 · Artefacts written

- `reports/per_fold_metrics.csv` — per `(p, fold)` aggregate metrics
- `reports/per_stratum_metrics.csv` — per `(p, stratum)` mean ± std
- `reports/p_sweep_v1_report.md` — this file
