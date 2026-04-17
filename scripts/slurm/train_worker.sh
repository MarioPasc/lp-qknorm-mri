#!/usr/bin/env bash
#SBATCH --ntasks=1
# Remaining --job-name, --array, --gres, --cpus-per-task, --mem, --time,
# --partition, --output and --error are supplied by launch.sh at sbatch
# submission time so every (p, fold) pair gets a human-readable job name.
#
# =============================================================================
# LP-QKNORM TRAINING WORKER
#
# One SLURM array task trains a single (p, fold) pair.  Mapping:
#   p_idx   = SLURM_ARRAY_TASK_ID / N_FOLDS
#   fold    = SLURM_ARRAY_TASK_ID % N_FOLDS
# The launcher exports P_VALUES (space-separated) and N_FOLDS so this
# mapping can be reproduced without re-reading the Picasso config.
#
# Expected env vars (exported by launch.sh):
#   REPO_SRC, CONDA_ENV_NAME, H5_PATH, RUN_DIR, P_VALUES, N_FOLDS,
#   SEED, NUM_WORKERS, BATCH_SIZE, PRECISION, EXPERIMENT_NAME
# =============================================================================

set -euo pipefail

START_TIME=$(date +%s)

# -----------------------------------------------------------------------------
# Resolve (p, fold) from the array task ID.
# -----------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID}"
read -r -a P_ARR <<< "${P_VALUES}"
N_FOLDS_INT=$(( N_FOLDS + 0 ))

P_IDX=$(( TASK_ID / N_FOLDS_INT ))
FOLD=$(( TASK_ID % N_FOLDS_INT ))
P="${P_ARR[${P_IDX}]}"

# Map 'vanilla' to a Hydra null override, numeric p to a float.
if [ "${P}" = "vanilla" ]; then
    P_OVERRIDE="model.p=null"
    P_TAG="vanilla"
else
    P_OVERRIDE="model.p=${P}"
    P_TAG="p${P}"
fi

echo "=========================================="
echo "LP-QKNORM TRAINING WORKER"
echo "=========================================="
echo "Started:       $(date)"
echo "Hostname:      $(hostname)"
echo "SLURM Job:     ${SLURM_JOB_ID:-local}"
echo "Array Task ID: ${TASK_ID}"
echo "Condition:     p=${P}  fold=${FOLD}"
echo "Run dir:       ${RUN_DIR}"
echo "H5 path:       ${H5_PATH}"
echo "Repo:          ${REPO_SRC}"
echo ""

# -----------------------------------------------------------------------------
# Environment setup
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
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import lpqknorm; print('lpqknorm', lpqknorm.__version__)"
echo ""

# -----------------------------------------------------------------------------
# Pre-flight: repo + data must be visible from the compute node.
# -----------------------------------------------------------------------------
cd "${REPO_SRC}"

if [ ! -f "${H5_PATH}" ]; then
    echo "[FAIL] HDF5 not found: ${H5_PATH}"
    exit 1
fi

# -----------------------------------------------------------------------------
# GPU monitoring (per task, 60 s cadence).
# -----------------------------------------------------------------------------
GPU_LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${GPU_LOG_DIR}"

echo "GPU status (pre-training):"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader
echo ""

GPU_CSV="${GPU_LOG_DIR}/gpu_${P_TAG}_f${FOLD}_${SLURM_JOB_ID}.csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv -l 60 > "${GPU_CSV}" 2>/dev/null &
GPU_MONITOR_PID=$!
echo "[gpu-monitor] PID=${GPU_MONITOR_PID}, output=${GPU_CSV}"

# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
echo "=========================================="
echo "TRAINING ${P_TAG} fold=${FOLD}"
echo "=========================================="

python -m lpqknorm.cli.train \
    experiment="${EXPERIMENT_NAME}" \
    "${P_OVERRIDE}" \
    data.fold="${FOLD}" \
    data.use_mock=false \
    data.h5_path="${H5_PATH}" \
    run_dir="${RUN_DIR}" \
    training.seed="${SEED}" \
    training.num_workers="${NUM_WORKERS}" \
    training.batch_size="${BATCH_SIZE}" \
    training.precision="${PRECISION}"

TRAIN_RC=$?

# Stop GPU monitor
if [ -n "${GPU_MONITOR_PID:-}" ] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    echo "[gpu-monitor] Stopped"
fi

echo "GPU status (post-training):"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "WORKER COMPLETED: ${P_TAG} fold=${FOLD}"
echo "=========================================="
echo "Duration: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
echo "Finished: $(date)"
echo "Exit rc:  ${TRAIN_RC}"

exit "${TRAIN_RC}"
