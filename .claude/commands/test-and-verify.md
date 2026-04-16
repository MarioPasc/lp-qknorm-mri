---
name: test-and-verify
description: |
  Run the full Lp-QKNorm MRI verification pipeline: pytest, ruff, mypy. Then check
  that test names and coverage relate to the scientific hypothesis (Lp norm effect
  on small-lesion attention and segmentation). Use after any code change.
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# Test and Verify Pipeline

Run the full Lp-QKNorm MRI verification pipeline in the `lpqknorm` conda environment
and report results.

## Step 1: Run pytest

```bash
cd /home/mpascual/research/code/lp-qknorm-mri && ~/.conda/envs/lpqknorm/bin/python -m pytest tests/ -v --tb=short 2>&1
```

Report: total passed, failed, errors, skipped.

## Step 2: Run ruff

```bash
cd /home/mpascual/research/code/lp-qknorm-mri && ~/.conda/envs/lpqknorm/bin/python -m ruff check src/ tests/ 2>&1
```

Report: number of issues, or "All checks passed".

## Step 3: Run mypy

```bash
cd /home/mpascual/research/code/lp-qknorm-mri && ~/.conda/envs/lpqknorm/bin/python -m mypy src/lpqknorm/ 2>&1
```

Report: number of errors, or "Success".

## Step 4: Hypothesis Alignment Check

Grep through test files for keywords that indicate alignment with the scientific
hypothesis and experimental design:

- "qknorm" or "lp_norm" -- tests for the Lp normalization module
- "equivalence" or "p.*=.*2" -- tests for p=2 QKNorm equivalence (critical)
- "leakage" or "patient" -- tests for patient-level split integrity
- "peakiness" or "entropy" or "lesion_mass" -- tests for mechanistic probes
- "bootstrap" or "effect_size" or "cohen" -- tests for statistical analysis
- "stratif" -- tests for volume-based lesion stratification
- "interior.*max" or "logit.*gap" -- tests connecting to toy-model prediction

Report which hypothesis-relevant test categories exist and which are still missing.

## Output Format

Present results as a summary table:

| Check | Status | Details |
|-------|--------|---------|
| pytest | PASS/FAIL | X passed, Y failed |
| ruff | PASS/FAIL | X issues |
| mypy | PASS/FAIL | X errors |
| Hypothesis coverage | X/7 | Which categories present |

If ALL pass, conclude with: "All verification checks passed."
If any fail, list the failures and suggest fixes.
