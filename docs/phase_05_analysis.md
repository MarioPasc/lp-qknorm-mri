# Phase 5 — Downstream Analysis, Effect Sizes, and Figures

## Goal

Consume the run artefacts produced by Phases 3 and 4 and produce the
analytical objects that go into the preprint:

1. The toy-model prediction figure — analytical `Δ(p)` curves for several
   sparsity levels, establishing the theoretical expectation before any
   empirical results are shown. **This figure requires no experimental data
   and can be generated on Day 1.**
2. The headline comparison — small-lesion recall at `p*` versus `p = 2`,
   with a patient-level paired bootstrap confidence interval and effect
   size.
3. The stratified performance table — Dice, IoU, HD95, lesion-wise recall,
   per volume stratum, per condition (vanilla + 5 Lp values), averaged
   across folds with dispersion. The vanilla column contextualises how
   much of the improvement is QKNorm itself versus the Lp generalisation.
4. Per-fold `p*` — which `p` wins each fold, and the stability thereof.
5. The five probe curves versus `p`, averaged over blocks, heads, and
   patients — the mechanistic chain of evidence.
6. Correlation between probe trajectories and segmentation gains across
   patients: does "attention lands more on my lesion" predict "my lesion is
   segmented better"?
7. **Cross-dataset p* vs. lesion size** — if multiple datasets have been
   run, plot p* (per stratum) as a function of median lesion volume across
   datasets. This tests whether the optimal Lp parameter systematically
   varies with lesion size across pathologies, or whether the effect is
   dataset-specific.

This phase is **purely offline** — no GPU required. It should run in under
5 minutes on a laptop against the full `results/` directory.

**Multi-dataset support.** The analysis code reads the `dataset_name` from
each run's manifest (or from the HDF5 header path). When results span
multiple datasets (ATLAS, BraTS, MELD, etc.), figures and tables are
stratified by dataset. The cross-dataset analysis enables the secondary
research question: **does the relationship between optimal p and lesion
size generalize across pathologies?** Specifically, if small-stratum p*
differs from large-stratum p* within ATLAS, and a similar pattern appears
in BraTS or MELD, the Lp normalization mechanism is pathology-general
rather than stroke-specific.

## Principle: every figure is reproducible from a single command

```
python -m lpqknorm.cli.analyze results_root=results/experiment_name \
    output=paper_outputs/
```

Produces: all tables as parquet + LaTeX, all figures as PDF + PNG, all
bootstrap distributions as `.npz`, one `analysis_manifest.json` describing
what was run.

## I/O contract

### Inputs

The `results_root` layout from Phase 3:
```
results/{experiment}/p={p}/fold={fold}/seed={seed}/
```

### Outputs (`paper_outputs/`)

```
.
├── analysis_manifest.json           # exact run set analysed + timestamp
├── tables/
│   ├── stratified_metrics.parquet
│   ├── stratified_metrics.tex
│   ├── headline_comparison.parquet  # p* vs p=2 per fold
│   ├── headline_comparison.tex
│   ├── per_fold_argmax.parquet      # argmax p per fold, per metric
│   └── probe_summary.parquet
├── bootstrap/
│   ├── {metric}_{stratum}_p={p}_vs_p=2.npz
│   └── effect_sizes.parquet
├── figures/
│   ├── fig1_toy_model_prediction.pdf  # Δ(p) curves for several sparsity levels
│   ├── fig2_stratified_dice.pdf
│   ├── fig3_small_recall_vs_p.pdf     # includes vanilla baseline marker
│   ├── fig4_probe_trajectory.pdf      # 5 subplots
│   ├── fig5_mechanism_chain.pdf       # correlation plot
│   └── fig6_per_patient_effect.pdf    # per-patient forest plot
└── logs/
    └── analyze.log
```

## Analytical procedures

### Toy-model prediction (Figure 1 — no experimental data required)

Plot the analytical logit-gap function from the design discussion:

```
Δ(p, s, d_k) = s^{1 − 2/p} · [1 − (s / d_k)^{1/p}]
```

where `s` is the sparsity (number of dominant coordinates in the lesion
token's feature vector) and `d_k` is the per-head dimension. Sweep
`p ∈ [1.5, 8]` continuously and overlay curves for
`s ∈ {2, 4, 8, 16}` with `d_k` set to the actual head dimension of the
model (determined after Phase 2 MONAI inspection — likely 24 or 48).

This figure establishes the theoretical prediction *before* any
empirical result: there exists an interior maximum at `p* > 2` whose
location depends on sparsity. Mark the experimental sweep range
`p ∈ {2, 2.5, 3, 3.5, 4}` with a shaded band. The empirical Probe 4
(logit gap) should qualitatively reproduce the shape of this curve.

Implementation: pure NumPy + matplotlib, no model or data dependencies.
Place in `analysis/figures.py::fig_toy_model_prediction`. Can and should
be generated on Day 1 alongside the data pipeline.

### Stratified metrics table

Load `test_per_patient.parquet` and `test_per_lesion.parquet` from every
run. For each `(condition, fold, stratum ∈ {small, medium, large, all})`
where condition ∈ {vanilla, 2.0, 2.5, 3.0, 3.5, 4.0}, compute:

- Dice (mean, std, 95 % CI via patient-level bootstrap).
- IoU (same).
- HD95 (median and IQR — distance metrics are heavy-tailed).
- Lesion-wise recall at FP ≤ 1 per slice (mean, CI).
- Detection rate stratified by lesion volume deciles.

Then average across folds per condition, reporting `mean ± pooled SD`.
The headline row is `stratum = small`. The vanilla column provides a
lower-bound reference — if `p = 2` does not improve over vanilla on
the small stratum, QKNorm itself is not helping and the Lp sweep is
moot.

### Paired bootstrap

For the pair `(p, p_ref = 2.0)` and a metric `m`, per fold:

1. Form per-patient values `m^{p}_i` and `m^{2}_i` from
   `test_per_patient.parquet` for patients in that fold's test set.
2. Compute the paired difference `d_i = m^{p}_i - m^{2}_i`.
3. Resample patients with replacement `B = 10 000` times; compute the mean
   of `d_i` per resample.
4. Report the 95 % percentile CI of the bootstrap mean and the fraction of
   bootstrap means with the same sign as the observed mean (one-sided
   bootstrap p-value).

Aggregate across folds by stacking the per-fold difference vectors and
re-bootstrapping at the patient level. Report the combined CI as the
primary comparison. Store raw bootstrap distributions under
`bootstrap/{metric}_{stratum}_p={p}_vs_p=2.npz` so reviewers can recompute.

### Effect size

Paired Cohen's `d`:

```
d = mean(d_i) / std(d_i),    d_i = m^{p}_i - m^{2}_i
```

Report `d` alongside the CI. Reference interpretations: `|d| < 0.2`
negligible, `0.2 ≤ |d| < 0.5` small, `0.5 ≤ |d| < 0.8` medium, `|d| ≥ 0.8`
large (*Cohen, 1988*).

### Per-fold `p*`

For each fold and each metric `m`, `p*_fold = argmax_p mean_test_patient(m)`.
Report the mode across folds and the fraction of folds that agree with it —
this is the stability check that the headline finding is not fold-idiosyncratic.

### Probe-segmentation correlation

For each patient and each probe `π` at the best checkpoint, compute a
scalar: `π̄_i = mean(π over lesion tokens of patient i)`. For each
`(p, fold)`, compute `Corr(π̄_i, Dice_i)` and `Corr(π̄_i, lesion_recall_i)`
across patients. A positive correlation for Probes 1, 3, 5 (peakiness,
mass, IoU) and a negative correlation for Probe 2 (entropy) would
constitute the mechanistic chain surviving at the **per-patient** level,
not just as sweep-level means. This is the strongest form of the
mechanistic claim.

## Open questions the agent must resolve

1. **Fold-aggregation choice for headline stats**. Default: combine
   per-patient differences across folds into a single vector before the
   bootstrap. Alternative: meta-analysis-style random-effects pool. Use the
   pooled-patients default; note the alternative in analysis docstrings.
2. **Multiple comparison correction across `p`**. Five comparisons against
   `p = 2` (vanilla, 2.5, 3.0, 3.5, 4.0). Use Holm–Bonferroni (*Holm,
   1979*) across the five tests; report unadjusted and adjusted p-values.
   The vanilla-vs-p=2 comparison is included to quantify QKNorm's own
   benefit and to contextualise the Lp improvement.
3. **HD95 when a prediction is empty**. MONAI returns `inf` or `NaN`.
   Convention: drop HD95 from the average when any of `pred` or `target`
   is empty, and separately report the fraction of empty predictions per
   `(p, stratum)`. Document.
4. **Threshold for lesion-wise detection**. IoU ≥ 0.1 is a common
   loose-detection threshold (*Wu et al., Med. Image Anal. 2022*).
   Report sensitivity to the threshold via a small additional table at
   IoU ∈ {0.05, 0.1, 0.25}.

## Public API (`src/lpqknorm/analysis/`)

```python
# aggregation.py
def load_runs(results_root: Path,
              experiment: str | None = None) -> pd.DataFrame: ...
def load_per_patient(results_root: Path) -> pd.DataFrame: ...
def load_probes(results_root: Path,
                checkpoint: Literal["best_dice", "final"] = "best_dice"
                ) -> pd.DataFrame: ...

# stratification.py
def attach_strata(per_patient: pd.DataFrame,
                  strata_path: Path) -> pd.DataFrame: ...

# bootstrap.py
@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    p_value_one_sided: float
    distribution: np.ndarray        # shape (B,)
    n_patients: int

def paired_patient_bootstrap(
    values_treatment: np.ndarray,
    values_control: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 20260216,
    ci: float = 0.95,
) -> BootstrapResult: ...

# effect_size.py
def paired_cohen_d(values_treatment: np.ndarray,
                   values_control: np.ndarray) -> float: ...

# probe_curves.py
def probe_curve(probes: pd.DataFrame, probe_name: str,
                aggregate: Literal["patient", "token"]) -> pd.DataFrame: ...

# figures.py
def fig_toy_model_prediction(
    sparsity_levels: Sequence[int], d_k: int, out: Path,
) -> Path:
    """Plot the analytical Δ(p) = s^{1−2/p}[1−(s/d_k)^{1/p}] for
    several sparsity levels s, highlighting the interior maximum.
    Pure math — no experimental data needed. Can be generated on Day 1."""
    ...

def fig_stratified_dice(tables: Tables, out: Path) -> Path: ...
def fig_small_recall_vs_p(tables: Tables, out: Path) -> Path:
    """Includes a horizontal dashed line / distinct marker for the vanilla
    (no QKNorm) baseline to contextualise the Lp improvement."""
    ...
def fig_probe_trajectory(probes: pd.DataFrame, out: Path) -> Path: ...
def fig_mechanism_chain(per_patient: pd.DataFrame,
                        probes: pd.DataFrame, out: Path) -> Path: ...
def fig_per_patient_effect(per_patient: pd.DataFrame, out: Path) -> Path: ...
```

All figure functions take raw dataframes in, write one file, return the path.
Keep styling in a single `style.py` module (matplotlib rcParams, publication
defaults: font size 9, serif, DPI 300, width 3.5" single-column or 7.0"
double).

## Design notes

**Why patient-level bootstrap, not slice-level.** Slices within a patient
are not independent. Slice-level resampling inflates effective sample size
and produces artificially narrow CIs. See Nadeau & Bengio (2003,
*Machine Learning* 52(3)) for the general argument.

**Why report per-fold `p*` separately.** If `p*` jumps wildly across folds
(e.g., 2.5, 4.0, 3.0), the sweep is under-powered and the headline
recommendation is not reliable. Reporting this honestly is a feature, not
a weakness.

**Why correlate probes with per-patient outcomes.** Sweep-level means can
agree with the mechanistic story by coincidence. Per-patient correlation
is the strictest falsification target: if the probe and the metric are
uncorrelated *within* a run, the mechanism is probably not doing the work.

**Why a single "analyze" command.** So the reviewer response "please
recompute X with IoU threshold Y" is a config change, not a rewrite.

## Implementation checklist

1. `analysis/aggregation.py` — `load_runs`, `load_per_patient`,
   `load_probes`. Each returns a long-format dataframe with
   `(run_id, p, fold, seed, ...)` columns. Cache on disk; re-read only if
   `results/` has changed (mtime check).
2. `analysis/bootstrap.py` — paired bootstrap, deterministic given seed.
   Vectorised with NumPy (no pandas inside the hot loop).
3. `analysis/effect_size.py` — paired Cohen's d + helper for small-sample
   correction (Hedges' g) as an alternative.
4. `analysis/stratification.py` — re-attach strata from Phase 1 outputs to
   the per-patient tables. Guards against stratum mismatch.
5. `analysis/probe_curves.py` — load probe HDF5s, aggregate per patient,
   stack into a dataframe.
6. `analysis/figures.py` — the five figure functions listed above.
7. `analysis/style.py` — matplotlib rcParams helpers.
8. `cli/analyze.py` — orchestrates everything, writes `analysis_manifest.json`
   with the analysed run set and commit SHA.

## Acceptance tests

### 1. Bootstrap recovers known p-value on synthetic data

```python
def test_bootstrap_synthetic():
    rng = np.random.default_rng(0)
    # Construct a paired design with known mean diff of 0.05 and SD 0.1
    n = 200
    control = rng.normal(0.7, 0.1, n)
    treatment = control + rng.normal(0.05, 0.02, n)
    result = paired_patient_bootstrap(treatment, control, n_resamples=5000)
    assert result.mean == pytest.approx(0.05, abs=0.01)
    assert result.p_value_one_sided < 0.01
    assert result.ci_low > 0
```

### 2. Bootstrap correctly returns null for no effect

Synthetic data with zero mean difference: CI contains 0, `p > 0.05`.

### 3. Paired Cohen's d reference values

```python
def test_cohen_d_known():
    d = np.array([0.1] * 100)        # constant diff
    e = np.zeros(100)
    # std(d - e) = 0; expect +inf or protect against it
    result = paired_cohen_d(d + 1e-9 * np.random.randn(100), e)
    assert result > 1.0
```

### 4. Stratum attachment integrity

```python
def test_strata_attachment_no_loss():
    per_patient = _mock_per_patient(50)
    strata = _mock_strata(50)
    joined = attach_strata(per_patient, strata)
    assert len(joined) == len(per_patient)
    assert joined["volume_stratum"].isin({"small", "medium", "large"}).all()
```

### 5. End-to-end analyze CLI on mock results

```python
def test_analyze_cli_on_mock(tmp_path):
    mock_root = _build_mock_results(tmp_path / "results")
    out = tmp_path / "outputs"
    subprocess.check_call([
        "python", "-m", "lpqknorm.cli.analyze",
        f"results_root={mock_root}", f"output={out}",
    ])
    assert (out / "tables/stratified_metrics.parquet").exists()
    assert (out / "figures/fig1_toy_model_prediction.pdf").exists()
    assert (out / "figures/fig3_small_recall_vs_p.pdf").exists()
    assert (out / "analysis_manifest.json").exists()
```

### 6. Figure determinism

Two back-to-back invocations of the figure functions on the same data
produce byte-identical PDFs (seed matplotlib, disable timestamps).

### 7. Holm–Bonferroni implementation

Against a reference implementation (`statsmodels.stats.multitest`), our
Holm adjustment agrees for 10 random p-value vectors.

## Expected runtime

- Full analysis over 18 runs: 2–5 minutes.
- Per-figure: < 20 s.
- Bootstrap with `B = 10 000` over 300 patients: ~2 s per comparison.

## What "the paper works" looks like at the end of Phase 5

The **two** things that must hold for a clean preprint:

> 1. `p = 2` (QKNorm) improves over vanilla on the small stratum — this
>    validates QKNorm itself as a useful baseline and justifies the Lp
>    generalisation inquiry.
> 2. Small-stratum lesion-wise recall at `p = p*` exceeds that at `p = 2`
>    with a 95 % bootstrap CI excluding zero, a paired Cohen's d ≥ 0.3,
>    and per-patient correlation with at least two of the five probes in
>    the theoretically predicted direction (positive for peakiness / lesion
>    mass / attention IoU, negative for entropy) at `|r| ≥ 0.2`.

Additionally, the empirical Probe 4 (logit gap) curve should
qualitatively match the shape of the toy-model prediction in Figure 1 —
an interior maximum, not a monotone increase.

If both hold, the preprint has a headline result with a mechanistic
backing. If only (2) holds without (1), the vanilla baseline is already
as good as QKNorm, and the Lp improvement may be noise — report
honestly. If the probes agree and the headline does not, the paper
becomes a negative result about a plausible mechanism that does not
translate — still publishable, arguably more interesting.

## References

- Cohen. *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed.
  Routledge, 1988.
- Hedges. *Distribution Theory for Glass's Estimator of Effect Size*.
  J. Educ. Stat. 1981.
- Holm. *A Simple Sequentially Rejective Multiple Test Procedure*.
  Scand. J. Stat. 1979.
- Efron & Tibshirani. *An Introduction to the Bootstrap*. Chapman & Hall,
  1993.
- Nadeau & Bengio. *Inference for the Generalization Error*. Mach. Learn.
  2003. doi:10.1023/A:1024068626366.
- Wu et al. *Deep learning for lesion detection in medical imaging: Review
  and outlook*. Med. Image Anal. 2022.
