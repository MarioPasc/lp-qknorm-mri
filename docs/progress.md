# Development Progress

## Phase 0 — Project Scaffolding
- [x] Repository initialized
- [x] `.gitignore` configured (Python + project-specific exclusions)
- [x] `LICENSE` (MIT)
- [x] `CLAUDE.md` — agent-facing project instructions
- [x] `pyproject.toml` — metadata, dependencies, tool configuration
- [x] `.claude/settings.json` — permissions, hooks (ruff auto-format, sensitive file blocking)
- [x] `.claude/settings.local.json` — local permission overrides
- [x] `.claude/agents/implementation-scientist.md` — mathematical rigor agent
- [x] `.claude/agents/proposal-guard.md` — hypothesis alignment guard
- [x] `.claude/agents/test-runner.md` — fast verification agent
- [x] `.claude/commands/test-and-verify.md` — full verification pipeline
- [x] `.claude/hooks/compact-context.sh` — context recovery after compaction
- [x] `docs/` — all five phase specifications written
- [X] Install editable package (`pip install -e ".[dev]"`)

## Phase 1 — Data Pipeline
- [ ] `src/lpqknorm/utils/exceptions.py` (add SchemaValidationError, ConverterError)
- [ ] `src/lpqknorm/utils/seeding.py`
- [ ] `src/lpqknorm/data/schema.py` (DatasetHeader, validate_h5)
- [ ] `src/lpqknorm/data/converter.py` (DatasetConverter protocol, SubjectRecord, SubjectVolume, write_standardized_h5)
- [ ] `src/lpqknorm/data/converters/__init__.py` (converter registry)
- [ ] `src/lpqknorm/data/converters/atlas.py` (AtlasConverter)
- [ ] `src/lpqknorm/data/stratification.py`
- [ ] `src/lpqknorm/data/splits.py`
- [ ] `src/lpqknorm/data/transforms.py` (2D + 3D transforms)
- [ ] `src/lpqknorm/data/datamodule.py` (SegmentationDataModule, dual-mode 2D/3D)
- [ ] `src/lpqknorm/cli/preprocess.py`
- [ ] `tests/unit/test_schema.py`
- [ ] `tests/unit/test_splits.py`
- [ ] `tests/unit/test_stratification.py`
- [ ] `tests/fixtures/synthetic_dataset.py` (5-patient generic fixture)

## Phase 2 — Model
- [X] `src/lpqknorm/models/lp_qknorm.py`
- [X] `src/lpqknorm/models/attention.py`
- [X] `src/lpqknorm/models/swin_unetr_lp.py`
- [X] `src/lpqknorm/models/hooks.py`
- [X] `tests/unit/test_lp_qknorm.py`
- [X] `tests/unit/test_attention_equivalence.py`
- [X] `tests/integration/test_forward_pass.py`

## Phase 3 — Training
- [ ] `src/lpqknorm/training/module.py`
- [ ] `src/lpqknorm/training/losses.py`
- [ ] `src/lpqknorm/training/metrics.py`
- [ ] `src/lpqknorm/training/callbacks.py`
- [ ] `src/lpqknorm/training/logging.py`
- [ ] `src/lpqknorm/cli/train.py`
- [ ] `scripts/submit_sweep.sbatch`
- [ ] `tests/integration/test_training_step.py`
- [ ] `tests/integration/test_resume.py`

## Phase 4 — Probes
- [X] `src/lpqknorm/probes/base.py` (with `per_block` result field)
- [X] `src/lpqknorm/probes/peakiness.py` — Probe 1
- [X] `src/lpqknorm/probes/entropy.py` — Probe 2
- [X] `src/lpqknorm/probes/lesion_mass.py` — Probe 3
- [X] `src/lpqknorm/probes/logit_gap.py` — Probe 4
- [X] `src/lpqknorm/probes/attention_iou.py` — Probe 5
- [X] `src/lpqknorm/probes/spatial_loc_error.py` — Probe 6 (SLE, NEW)
- [X] `src/lpqknorm/probes/linear_probe.py` — Probe 7 (L1-logistic, NEW)
- [X] `src/lpqknorm/probes/spectral.py` — Probe 8 (PR / stable rank, NEW)
- [X] `src/lpqknorm/probes/attention_maps.py` — reconstruction, rollout, overlay (NEW)
- [X] `src/lpqknorm/probes/patching.py` — `PatchingConfig`, `ActivationPatcher` (NEW)
- [X] `src/lpqknorm/probes/tokenization.py` — + `window_boundary_distance`
- [X] `src/lpqknorm/probes/recorder.py` — full HDF5 schema: `/metadata`, `/inputs`,
      `/block_0_wmsa`, `/block_1_swmsa` with attention/logits maps, rel-pos bias
      and entropy, per-head linear-probe metrics, spectral scalars + eigenvalues
- [X] `src/lpqknorm/cli/probe.py` — includes Probes 6–8 in default list
- [X] `src/lpqknorm/cli/patching.py` — standalone post-hoc patching CLI (NEW)
- [X] `src/lpqknorm/training/callbacks.py` — `AlphaLogger`, `PatchingCallback` (NEW)
- [X] `tests/unit/test_probes_synthetic.py` — AT1 (Probes 1–6) + AT2 logit-gap
- [X] `tests/unit/test_tokenization.py` — AT3, AT4, window-boundary distance
- [X] `tests/unit/test_attention_maps.py` — AT8 identity rollout & reconstruction (NEW)
- [X] `tests/unit/test_patching.py` — AT9 self-patch identity + cross-model delta (NEW)
- [X] `tests/integration/test_probe_pipeline.py` — AT5 (full HDF5), AT6 determinism,
      AT7 no-autograd, AT10 fp16 round-trip, AT11 α-log monotonicity

## Phase 5 — Analysis
- [ ] `src/lpqknorm/analysis/aggregation.py`
- [ ] `src/lpqknorm/analysis/bootstrap.py`
- [ ] `src/lpqknorm/analysis/effect_size.py`
- [ ] `src/lpqknorm/analysis/stratification.py`
- [ ] `src/lpqknorm/analysis/probe_curves.py`
- [ ] `src/lpqknorm/analysis/figures.py`
- [ ] `src/lpqknorm/cli/analyze.py`
- [ ] `tests/unit/test_bootstrap.py`
