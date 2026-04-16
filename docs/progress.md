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
- [ ] `environment.yml` — conda environment definition
- [ ] `.pre-commit-config.yaml` — ruff, mypy, nbstripout hooks
- [ ] Install editable package (`pip install -e ".[dev]"`)

## Phase 1 — Data Pipeline
- [ ] `src/lpqknorm/utils/exceptions.py`
- [ ] `src/lpqknorm/utils/seeding.py`
- [ ] `src/lpqknorm/data/atlas.py`
- [ ] `src/lpqknorm/data/preprocessing.py`
- [ ] `src/lpqknorm/data/stratification.py`
- [ ] `src/lpqknorm/data/splits.py`
- [ ] `src/lpqknorm/data/transforms.py`
- [ ] `src/lpqknorm/data/datamodule.py`
- [ ] `src/lpqknorm/cli/preprocess.py`
- [ ] `tests/unit/test_splits.py`
- [ ] `tests/unit/test_stratification.py`
- [ ] `tests/fixtures/synthetic_atlas.py`

## Phase 2 — Model
- [ ] `src/lpqknorm/models/lp_qknorm.py`
- [ ] `src/lpqknorm/models/attention.py`
- [ ] `src/lpqknorm/models/swin_unetr_lp.py`
- [ ] `src/lpqknorm/models/hooks.py`
- [ ] `tests/unit/test_lp_qknorm.py`
- [ ] `tests/unit/test_attention_equivalence.py`
- [ ] `tests/integration/test_forward_pass.py`

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
- [ ] `src/lpqknorm/probes/base.py`
- [ ] `src/lpqknorm/probes/peakiness.py`
- [ ] `src/lpqknorm/probes/entropy.py`
- [ ] `src/lpqknorm/probes/lesion_mass.py`
- [ ] `src/lpqknorm/probes/logit_gap.py`
- [ ] `src/lpqknorm/probes/attention_iou.py`
- [ ] `src/lpqknorm/probes/recorder.py`
- [ ] `src/lpqknorm/cli/probe.py`
- [ ] `tests/unit/test_probes_synthetic.py`
- [ ] `tests/integration/test_probe_pipeline.py`

## Phase 5 — Analysis
- [ ] `src/lpqknorm/analysis/aggregation.py`
- [ ] `src/lpqknorm/analysis/bootstrap.py`
- [ ] `src/lpqknorm/analysis/effect_size.py`
- [ ] `src/lpqknorm/analysis/stratification.py`
- [ ] `src/lpqknorm/analysis/probe_curves.py`
- [ ] `src/lpqknorm/analysis/figures.py`
- [ ] `src/lpqknorm/cli/analyze.py`
- [ ] `tests/unit/test_bootstrap.py`
