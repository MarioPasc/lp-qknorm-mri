#!/usr/bin/env bash
#SBATCH --job-name=lpqk_probe
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=dgx
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/users/tic_163_uma/mpascual/execs/lpqknorm_mri/p_sweep_v1/logs/probe_%j.out
#SBATCH --error=/mnt/home/users/tic_163_uma/mpascual/execs/lpqknorm_mri/p_sweep_v1/logs/probe_%j.err
#
# =============================================================================
# LP-QKNORM PHASE 4 — PROBES + ACTIVATION PATCHING (single A100 job)
#
# Phase 4 is forward-only and bounded by 18 ckpts × ~10 s probes + 3 folds ×
# ~60 s patching ≈ 6–10 min on an A100.  A single non-array job is enough.
#
# Submit (Picasso login node):
#
#     cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/lp-qknorm-mri
#     git pull                                  # ensure scripts/run_phase4_local.py is present
#     sbatch scripts/slurm/probe_phase4.sbatch
#
# Optional overrides via env:
#     P_STAR=4.0  P_BASELINE=2.0  N_PROBE=32  EXPERIMENT_NAME=p_sweep_v1
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)

REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/lp-qknorm-mri}"
H5_PATH="${H5_PATH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/lpqknorm/brats_men.h5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-p_sweep_v1}"
RUN_DIR="${RUN_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/lpqknorm_mri/${EXPERIMENT_NAME}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lpqknorm}"
P_STAR="${P_STAR:-4.0}"
P_BASELINE="${P_BASELINE:-2.0}"
N_PROBE="${N_PROBE:-32}"
SEED="${SEED:-20260216}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${RUN_DIR}/logs"

echo "=========================================="
echo "LP-QKNORM PHASE 4 — PROBES + PATCHING"
echo "=========================================="
echo "Started:       $(date)"
echo "Hostname:      $(hostname)"
echo "SLURM Job:     ${SLURM_JOB_ID:-local}"
echo "Repo:          ${REPO_SRC}"
echo "H5 path:       ${H5_PATH}"
echo "Run dir:       ${RUN_DIR}"
echo "p* / p_2:      ${P_STAR} / ${P_BASELINE}"
echo "Probe samples: ${N_PROBE}"
echo ""

# -----------------------------------------------------------------------------
# Conda activation (Picasso convention: try modules, fall back to PATH).
# -----------------------------------------------------------------------------
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done
if [ "$module_loaded" -eq 0 ]; then
    echo "[env] No conda module loaded; assuming conda already in PATH."
fi
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "=========================================="
echo "ENVIRONMENT"
echo "=========================================="
echo "[python] $(which python)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import lpqknorm; print('lpqknorm', lpqknorm.__version__)"
echo ""

# -----------------------------------------------------------------------------
# Pre-flight: repo + data + driver script must be visible.
# -----------------------------------------------------------------------------
cd "${REPO_SRC}"
DRIVER="${REPO_SRC}/scripts/run_phase4_local.py"
if [ ! -f "${DRIVER}" ]; then
    echo "[FAIL] Phase-4 driver not found at ${DRIVER}"
    echo "       Run \`git pull\` in ${REPO_SRC} or sync scripts/run_phase4_local.py."
    exit 1
fi
if [ ! -f "${H5_PATH}" ]; then
    echo "[FAIL] HDF5 not found: ${H5_PATH}"
    exit 1
fi
if [ ! -d "${RUN_DIR}" ]; then
    echo "[FAIL] Run directory not found: ${RUN_DIR}"
    exit 1
fi

echo "GPU status (pre-run):"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader || true
echo ""

# -----------------------------------------------------------------------------
# Execute the Phase-4 sweep driver.
#
# The driver discovers all (p, fold) run directories under ${RUN_DIR},
# loads the best_val_dice checkpoint, runs the 9 probes, writes
# probes/epoch_best_dice.h5, and then performs activation patching
# p* -> p_baseline at stage-0 across all folds.  Idempotent: existing
# outputs are skipped unless OVERWRITE=1.
# -----------------------------------------------------------------------------
OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" -eq 1 ]; then
    OVERWRITE_FLAG="--overwrite"
fi

set -x
python -u "${DRIVER}" \
    --results-root "${RUN_DIR}" \
    --h5-path      "${H5_PATH}" \
    --seed         "${SEED}" \
    --n-probe-samples "${N_PROBE}" \
    --device cuda \
    --num-workers  "${NUM_WORKERS}" \
    --p-star       "${P_STAR}" \
    --p-baseline   "${P_BASELINE}" \
    ${OVERWRITE_FLAG}
RC=$?
set +x

ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo "Phase 4 finished with exit code ${RC} in ${ELAPSED}s"
echo "Probes:   ${RUN_DIR}/p=*/fold=*/seed=${SEED}/probes/epoch_best_dice.h5"
echo "Patching: ${RUN_DIR}/p=${P_STAR}/fold=*/seed=${SEED}/probes/patching_best_dice.h5"
echo "=========================================="

exit ${RC}
