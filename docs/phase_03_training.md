# Phase 3 — Training and Exhaustive Logging

## Goal

Train the sweep `p ∈ {vanilla, 2.0, 2.5, 3.0, 3.5, 4.0}` ×
`fold ∈ {0, 1, 2}` = **18 runs**, and for each run persist *every quantity
that downstream analysis could conceivably need* to disk in a structured,
typed format. Nothing should require re-running training to answer a
downstream question.

The `vanilla` condition uses MONAI's stock `WindowAttention` with standard
scaled dot-product attention (no QKNorm). It serves as a lower-bound control
that contextualises the Lp sweep: is the gap `p* → p=2` comparable to the
gap `p=2 → vanilla`? This preempts the reviewer question "Does QKNorm itself
help here?" at a cost of only 3 extra runs.

This phase assumes Phases 1 and 2 have passed their acceptance tests. It
produces the Lightning module, callbacks, CLI, and SLURM submission
scaffolding.

## Principle: log everything, aggregate later

The cost of writing a few extra MB to disk per run is trivial compared to
the cost of re-training a sweep because "we forgot to log X". The rule is:
**if it is computed during training or validation, it is logged.**

## I/O contract

### Per-run artefact directory

Each run writes a self-contained directory
`results/{experiment}/p={p}/fold={fold}/seed={seed}/`:

```
.
├── config.yaml                    # full Hydra config snapshot (composed)
├── manifest.json                  # run metadata (see below)
├── code.diff                      # output of `git diff HEAD`
├── env.json                       # python packages + versions
│
├── metrics/
│   ├── train_steps.jsonl          # one line per optimiser step
│   ├── train_epochs.parquet       # aggregated per-epoch training metrics
│   ├── val_epochs.parquet         # per-epoch validation metrics
│   ├── val_per_patient.parquet    # per-epoch × per-patient metrics
│   ├── test_per_patient.parquet   # final-checkpoint test metrics
│   └── test_per_lesion.parquet    # final-checkpoint lesion-wise metrics
│
├── attention_stats/
│   └── epoch_{N}.parquet          # summary stats of attention tensors
│
├── gradient_stats/
│   └── layer_norms.parquet        # per-layer grad norms every K steps
│
├── predictions/
│   └── test_masks.h5              # /pred_logits, /pred_binary, /subject_id
│
├── checkpoints/
│   ├── last.ckpt
│   ├── best_val_dice.ckpt
│   └── best_small_recall.ckpt     # best on the small-stratum recall
│
└── probes/                        # populated in Phase 4 — reserve space here
    └── .keep
```

### `manifest.json` schema

```python
@dataclass(frozen=True)
class RunManifest:
    run_id: str                    # UUID
    experiment: str                # e.g. "p_sweep_v1"
    p: float
    fold: int
    seed: int
    git_sha: str
    git_dirty: bool
    git_branch: str
    started_utc: str               # ISO 8601
    finished_utc: str | None
    host: str
    gpu_model: str
    cuda_version: str
    torch_version: str
    monai_version: str
    lpqknorm_version: str
    config_hash: str               # sha256 of composed config
    split_hash: str                # sha256 of splits JSON
    n_train: int
    n_val: int
    n_test: int
    walltime_sec: float | None
    peak_gpu_memory_mb: float | None
    final_epoch: int | None
    best_val_dice: float | None
    best_small_recall: float | None
```

## Metrics recorded

### Per-step (`train_steps.jsonl`)

`step`, `epoch`, `loss_total`, `loss_bce`, `loss_dice`, `lr`, `grad_norm`,
`alpha_mean_stage0..3`, `alpha_std_stage0..3`, `throughput_samples_per_sec`,
`gpu_mem_mb`.

### Per-epoch training (`train_epochs.parquet`)

Aggregates of the above plus `epoch_duration_sec`.

### Per-epoch validation (`val_epochs.parquet`)

`epoch`, `val_loss_total`, `val_loss_bce`, `val_loss_dice`, `val_dice_mean`,
`val_iou_mean`, `val_lesion_recall_at_fp1`, `val_hd95_mean`.

### Per-epoch × per-patient (`val_per_patient.parquet`)

One row per (epoch, subject_id, volume_stratum):
`dice`, `iou`, `precision`, `recall`, `lesion_recall` (lesion-wise TPR),
`false_positives_per_slice`, `hd95`, `assd`.

This is the **primary data structure for downstream effect-size analysis**.
Patient-level paired bootstrap in Phase 5 operates directly on this table.

### Per-lesion test metrics (`test_per_lesion.parquet`)

Connected-component-level: `subject_id`, `lesion_id`, `lesion_volume_mm3`,
`was_detected` (any overlap > IoU threshold), `pred_iou_with_lesion`,
`pred_dice_with_lesion`. Threshold convention documented in
`training/metrics.py`. This enables per-lesion-size detection curves.

### Attention summary stats (`attention_stats/epoch_{N}.parquet`)

For `N ∈ {1, 5, 10, final}` and for each stage-0 attention block, on a fixed
10-batch subset of validation (seed-fixed, same across epochs):

`block_id`, `mean_entropy`, `median_entropy`, `mean_max_prob`,
`mean_lesion_mass_on_lesion_queries`, `mean_q_peakiness`,
`mean_k_peakiness`, `alpha_value`. This gives a coarse trajectory for free
— the full probe data is collected in Phase 4 at selected epochs.

### Gradient stats (`gradient_stats/layer_norms.parquet`)

Every 50 steps: per-layer `||∇θ||_2` for the QKV projections and the
`alpha_raw` parameters. Useful for diagnosing training instability at
unusual `p`.

## Training configuration (defaults in `configs/training/default.yaml`)

- `max_epochs: 100` (with early stopping).
- `early_stop_metric: val_dice_mean`, `patience: 15`.
- `optimizer: AdamW`, `lr: 3e-4`, `weight_decay: 1e-5`, `betas: (0.9, 0.999)`.
- `scheduler: CosineAnnealingLR`, `eta_min: 1e-6`.
- `batch_size: 16`.
- `accumulate_grad_batches: 1` (adjust if OOM on A100 40GB — unlikely at 2D
  224²).
- `precision: "bf16-mixed"` (A100 supports bfloat16; avoids fp16 scaling
  headaches with Lp normalization).
- `gradient_clip_val: 1.0`.
- `deterministic: true` (Lightning `trainer.deterministic=True`, plus
  `torch.use_deterministic_algorithms(True, warn_only=True)`).
- `num_workers: 8`, `persistent_workers: true`.
- `loss: 0.5 * BCEWithLogits + 0.5 * SoftDiceLoss`, class-weighted BCE with
  `pos_weight` estimated from training set.

All values are overridable via Hydra CLI.

## Open questions the agent must resolve

1. **bf16 + Lp normalisation numerical stability.** Run one short training
   job at `p = 4` in bf16 and confirm loss does not diverge. If it does,
   fall back to fp32 for the `LpQKNorm` module only via an autocast-disabled
   context (`torch.autocast(device_type="cuda", enabled=False)`) wrapping
   the normalisation call, while keeping the rest of the model in bf16.
2. **Class imbalance in BCE.** Compute `pos_weight` from the training set at
   DataModule setup time and inject it into the loss. Document the
   resulting value in the run manifest.
3. **Sampler choice.** Slices-with-lesion-only dataset, so no need for
   positive-oversampling. Verify by inspecting the manifest.
4. **Wandb vs. local-only logging.** Default to **local-only** (JSONL +
   parquet) because Picasso compute nodes may lack outbound network access.
   Make W&B opt-in via `training.wandb.enabled=false` default.

## Public API (`src/lpqknorm/training/`)

```python
# module.py
class LpSegmentationModule(pl.LightningModule):
    def __init__(self, model_cfg: ModelConfig, lp_cfg: LpQKNormConfig,
                 training_cfg: TrainingConfig) -> None: ...
    def training_step(self, batch, batch_idx): ...
    def validation_step(self, batch, batch_idx): ...
    def test_step(self, batch, batch_idx): ...
    def configure_optimizers(self): ...

# losses.py
class CompoundSegLoss(nn.Module):
    def __init__(self, bce_weight: float, dice_weight: float,
                 pos_weight: Tensor | None) -> None: ...

# metrics.py
def dice_score(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor: ...
def lesion_wise_detection(pred: Tensor, target: Tensor,
                          iou_threshold: float = 0.1) -> LesionDetectionResult: ...
def hd95(pred: Tensor, target: Tensor) -> Tensor: ...

# callbacks.py
class PerPatientMetricsCallback(pl.Callback): ...     # writes val_per_patient
class AttentionSummaryCallback(pl.Callback): ...      # stage-0 stats per epoch
class GradientNormCallback(pl.Callback): ...
class ManifestCallback(pl.Callback): ...              # writes manifest.json
class ArtefactDirectoryCallback(pl.Callback): ...     # ensures dir structure

# logging.py
class StructuredLogger:
    """Writes JSONL for steps, parquet for epochs/patients."""
    def __init__(self, run_dir: Path) -> None: ...
    def log_step(self, payload: dict) -> None: ...
    def log_epoch(self, stage: str, payload: dict) -> None: ...
    def log_per_patient(self, stage: str, epoch: int,
                        rows: list[dict]) -> None: ...
    def close(self) -> None: ...
```

## Design notes

**Why Lightning.** Built-in deterministic seeding, automatic checkpointing,
clean separation of training/validation/test loops, trivial multi-GPU if we
later scale. The `pl.Callback` hierarchy maps cleanly to the "log everything"
principle — each logged quantity is one callback.

**Why JSONL + parquet instead of TensorBoard/W&B only.** Parquet tables are
the native format for pandas/polars downstream analysis. JSONL is append-only
and crash-safe for per-step logs. TensorBoard is lossy (summarised, not
complete). W&B is fine as a secondary viewer but cannot be the source of
truth because of network constraints.

**Why fix the attention-summary batch across epochs.** Comparing entropy or
peakiness across epochs on different batches confounds "the model changed"
with "the input changed". Use the same 10 batches every epoch.

**Why best-on-small-recall in addition to best-on-dice.** Small-lesion
recall is the headline metric of the paper. The "best" checkpoint for the
actual claim might differ from the "best Dice" checkpoint.

## Implementation checklist

1. `utils/git.py` — `capture_git_state() -> GitState` (SHA, branch, dirty,
   diff text).
2. `training/losses.py` — `CompoundSegLoss` with tests on synthetic masks
   (Dice = 1 for perfect, = 0 for empty prediction vs. non-empty target).
3. `training/metrics.py` — Dice, IoU, HD95, lesion-wise detection. Prefer
   `monai.metrics` where available; wrap for consistent interface.
4. `training/logging.py` — `StructuredLogger` with fsync on each write for
   crash-safety.
5. `training/callbacks.py` — five callbacks listed above.
6. `training/module.py` — `LpSegmentationModule`.
7. `cli/train.py` — Hydra entry, instantiates DataModule + Module + Trainer
   + callbacks, runs `fit` then `test`.
8. `scripts/submit_sweep.sbatch` — SLURM array `0-17` mapping to
   (condition, fold) pairs where condition ∈ {vanilla, 2.0, 2.5, 3.0, 3.5,
   4.0}. Plus a separate `--array=0-2` for repeated seeds if time permits.
9. `scripts/verify_env.py` — dump `env.json`-compatible package info; refuse
   to train if deterministic flags cannot be set.

## Acceptance tests

### 1. Training step executes (`tests/integration/test_training_step.py`)

```python
def test_single_training_step_runs(tiny_datamodule, tmp_path):
    module = LpSegmentationModule(
        model_cfg=ModelConfig(feature_size=12),
        lp_cfg=LpQKNormConfig(p=3.0),
        training_cfg=TrainingConfig(lr=1e-3),
    )
    trainer = pl.Trainer(
        max_steps=2, logger=False, enable_checkpointing=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, tiny_datamodule)
    assert trainer.global_step == 2
```

### 2. Logged artefacts exist after a 1-epoch run

```python
def test_artefacts_after_one_epoch(tiny_datamodule, tmp_path):
    # ... run a 1-epoch training with all callbacks attached ...
    run_dir = tmp_path / "run"
    for expected in [
        "config.yaml", "manifest.json",
        "metrics/train_steps.jsonl",
        "metrics/train_epochs.parquet",
        "metrics/val_epochs.parquet",
        "metrics/val_per_patient.parquet",
        "checkpoints/last.ckpt",
    ]:
        assert (run_dir / expected).exists()
```

### 3. Determinism (two identical runs produce identical loss curves)

```python
def test_determinism(tiny_datamodule, tmp_path):
    def run_once(seed):
        pl.seed_everything(seed)
        # ... run 3 steps ...
        return load_train_steps_jsonl(run_dir)
    s1 = run_once(42)
    s2 = run_once(42)
    assert np.allclose([r["loss_total"] for r in s1],
                       [r["loss_total"] for r in s2], atol=1e-5)
```

### 4. Resume-from-checkpoint produces the same final state

Train 4 epochs; train 2 epochs + resume for 2 more from the `last.ckpt`;
assert final `val_dice_mean` is identical to 4 epochs atol=1e-4.

### 5. Manifest captures the split hash

```python
def test_manifest_split_hash(...):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "split_hash" in manifest
    expected = sha256_of_file(splits_json_path)
    assert manifest["split_hash"] == expected
```

### 6. Per-patient table has the right cardinality

`val_per_patient.parquet` must have `n_val_patients × n_epochs_run` rows
after training, with unique `(epoch, subject_id)` pairs.

### 7. Attention summary callback produces monotone-valid numbers

Entropy values are in `[0, log(W²)]`; lesion mass values are in `[0, 1]`.

### 8. Smoke test of SLURM submission (dry run)

`sbatch --test-only scripts/submit_sweep.sbatch` should return a valid
allocation estimate.

## Expected per-run resource profile

- Walltime: 45–75 min on A100 40GB at 2D 224² with bf16 and batch 16.
- Peak GPU memory: ~12 GB.
- Disk per run: 200–400 MB excluding probes.

The full 18-run sweep (6 conditions × 3 folds) fits comfortably in an
overnight window on a single A100 node; on two nodes, under 5 h.

## References

- Falcon et al. *PyTorch Lightning*. doi:10.5281/zenodo.3828935.
- Loshchilov & Hutter. *Decoupled Weight Decay Regularization*. ICLR 2019.
  arXiv:1711.05101.
- Isensee et al. *nnU-Net*. Nat. Methods 2021.
  doi:10.1038/s41592-020-01008-z (metric and loss conventions).
