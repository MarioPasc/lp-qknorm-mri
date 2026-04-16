#!/usr/bin/env bash
# Compaction recovery hook -- re-injects critical context after /compact

cat <<'CONTEXT'
## Post-Compact Context Recovery — Lp-QKNorm MRI

Project: Mechanistic study of Lp query-key normalization in windowed
self-attention for small-lesion segmentation (ATLAS v2.0).

Hypothesis: p > 2 in Lp-QKNorm improves attention concentration on small
lesions via: peakiness ↑ → entropy ↓ → lesion mass ↑ → logit gap ↑ →
attention-mask IoU ↑ → small-lesion recall ↑.

Environment: conda env `lpqknorm`, Python 3.11
Project root: /home/mpascual/research/code/lp-qknorm-mri
Results: /media/mpascual/Sandisk2TB/research/lpqknorm_mri/results/

Key invariants:
- p=2 MUST recover original QKNorm exactly (most important test)
- Patient-level splits only (no slice-level — data leakage)
- Only WindowAttention.forward is modified; rest of SwinUNETR is stock MONAI
- Probes on stage-1 attention only, fixed batch across epochs/runs
- Headline metric: small-lesion recall, NOT full-cohort Dice
- 18 runs total: {vanilla, 2.0, 2.5, 3.0, 3.5, 4.0} × 3 folds
- Vanilla = stock MONAI attention (no QKNorm), lower-bound control
- Figure 1 = toy-model Δ(p) prediction (pure math, no data needed)

Phases: 1-Data → 2-Model → 3-Training → 4-Probes → 5-Analysis
See docs/phase_0X_*.md for specifications.
See CLAUDE.md for full instructions.

Run tests: ~/.conda/envs/lpqknorm/bin/python -m pytest tests/ -v --tb=short
Run linter: ~/.conda/envs/lpqknorm/bin/python -m ruff check src/ tests/
Run types:  ~/.conda/envs/lpqknorm/bin/python -m mypy src/lpqknorm/
CONTEXT
