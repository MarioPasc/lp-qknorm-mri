#!/usr/bin/env bash
# scripts/slurm/setup.sh
# Pre-flight validation for the Lp-QKNorm Picasso sweep.
#
# Reads configs/picasso/default.yaml, verifies the env + data + imports are
# usable, and prints a one-line summary per check.
#
# Usage:
#   bash scripts/slurm/setup.sh [picasso_config_yaml]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PICASSO_CFG="${1:-${REPO_ROOT}/configs/picasso/default.yaml}"

echo "=========================================="
echo "LP-QKNORM PICASSO PRE-FLIGHT CHECKS"
echo "=========================================="
echo "Config:   ${PICASSO_CFG}"
echo "Repo:     ${REPO_ROOT}"
echo ""

if [ ! -f "${PICASSO_CFG}" ]; then
    echo "  [FAIL] Picasso config not found: ${PICASSO_CFG}"
    exit 1
fi

# Pick up conda from the usual suspects.
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done
if [ "$module_loaded" -eq 0 ]; then
    echo "[env] No conda module loaded; assuming conda already in PATH."
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lpqknorm}"

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

FAIL=0

# 1. lpqknorm importable
python -c "
import lpqknorm
from lpqknorm.cli.train import main as _train_main
from lpqknorm.data.splits import make_patient_kfold
print('  [OK]   lpqknorm imports (version=%s)' % lpqknorm.__version__)
" 2>/dev/null || { echo "  [FAIL] lpqknorm imports failed"; FAIL=1; }

# 2. Parse picasso config and verify paths exist.
python - <<PY || FAIL=1
from omegaconf import OmegaConf
from pathlib import Path
import sys

cfg = OmegaConf.load("${PICASSO_CFG}")

problems = 0

h5 = Path(cfg.paths.h5_path)
if h5.exists():
    print(f"  [OK]   HDF5 dataset: {h5}")
else:
    print(f"  [FAIL] HDF5 not found: {h5}")
    problems += 1

repo = Path(cfg.paths.repo_src)
if repo.exists():
    print(f"  [OK]   Repo:  {repo}")
else:
    print(f"  [FAIL] Repo not found: {repo}")
    problems += 1

run_parent = Path(cfg.paths.run_dir).parent
if run_parent.exists() or run_parent == Path("/"):
    print(f"  [OK]   Run-dir parent writable: {run_parent}")
else:
    print(f"  [WARN] Run-dir parent does not exist yet: {run_parent}")

p_vals = list(cfg.sweep.p_values)
folds = list(cfg.sweep.folds)
n_tasks = len(p_vals) * len(folds)
print(f"  [OK]   Sweep: {len(p_vals)} p-values × {len(folds)} folds = {n_tasks} tasks")
print(f"  [OK]   p_values: {p_vals}")
print(f"  [OK]   folds:    {folds}")

if problems:
    sys.exit(1)
PY

# 3. Verify the HDF5 split scheme (must be common_test_holdout for this sweep).
python - <<PY || FAIL=1
from omegaconf import OmegaConf
import h5py
import sys

cfg = OmegaConf.load("${PICASSO_CFG}")
with h5py.File(cfg.paths.h5_path, "r") as f:
    scheme = f["splits"].attrs.get("scheme", b"unknown")
    if isinstance(scheme, bytes):
        scheme = scheme.decode()
    n_test_patients = f["splits"].attrs.get("fixed_test_patients", -1)
    n_folds = f["splits"].attrs.get("n_folds", -1)
    if scheme != "common_test_holdout":
        print(f"  [FAIL] Split scheme is '{scheme}', expected 'common_test_holdout'")
        sys.exit(1)
    print(f"  [OK]   Split scheme: {scheme}")
    print(f"  [OK]   Fixed test patients: {n_test_patients}")
    print(f"  [OK]   Folds in H5: {n_folds}")
PY

# 4. GPU check (optional — login node typically has none).
python - <<PY 2>/dev/null || true
import torch
if torch.cuda.is_available():
    print(f"  [OK]   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  [WARN] No GPU detected (expected on login node)")
PY

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "All pre-flight checks PASSED"
    exit 0
else
    echo "Pre-flight checks FAILED"
    exit 1
fi
