#!/usr/bin/env bash
# =============================================================================
# LP-QKNORM — 30-MIN SMOKE TEST ON THE `exa` NODE (Titan H100, ~30 GB VRAM)
#
# Purpose: validate the full training+test+probe pipeline end-to-end on one
# (p, fold) pair before launching the 18-task sweep.  Confirms that every
# run-directory subfolder is populated:
#
#   attention_stats/   ← AttentionSummaryCallback  (epoch 1 + final)
#   checkpoints/       ← ModelCheckpoint           (last + best_val_dice + best_small_recall)
#   gradient_stats/    ← GradientNormCallback      (layer_norms.parquet)
#   metrics/           ← StructuredLogger          (train_steps.jsonl, val_per_patient, test_per_patient)
#   predictions/       ← TestPredictionWriter      (first 4 test batches as .npz)
#   probes/            ← AlphaLogger + cli/probe   (alpha_trajectory.jsonl + epoch_*.h5)
#
# Usage (from an allocated exa session with 1 GPU, ≤ 30 min budget):
#
#   bash scripts/slurm/test_exa_30min.sh                 # defaults: p=3.0 fold=0
#   P_VAL=2.5 FOLD=1 bash scripts/slurm/test_exa_30min.sh
#   MAX_EPOCHS=5 bash scripts/slurm/test_exa_30min.sh    # coarse-grained tuning
#
# This script assumes the caller already holds a GPU allocation (e.g. via
# `salloc --gres=gpu:1 --time=00:30:00 -w <exa-node>`).  It does not call
# sbatch.  Training runs to completion OR until the inner `timeout` wrapper
# fires at TRAIN_BUDGET_SEC; either way the post-hoc probe step is
# attempted on whatever checkpoint was saved.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# -----------------------------------------------------------------------------
# Tunables (override via environment)
# -----------------------------------------------------------------------------
P_VAL="${P_VAL:-3.0}"
FOLD="${FOLD:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-bf16-mixed}"
SEED="${SEED:-20260216}"

# Budget split: leave ~5 min for test + probe + I/O flush on the 30-min slot.
TRAIN_BUDGET_SEC="${TRAIN_BUDGET_SEC:-1500}"     # 25 min
PROBE_BUDGET_SEC="${PROBE_BUDGET_SEC:-240}"      # 4 min

# Paths — match configs/picasso/default.yaml exactly, but redirect run_dir
# to a separate "sweep_preflight" subtree so the preflight never overwrites
# a real run.
REPO_SRC="${REPO_SRC:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/lp-qknorm-mri}"
H5_PATH="${H5_PATH:-/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/lpqknorm/brats_men.h5}"
RUN_DIR="${RUN_DIR:-/mnt/home/users/tic_163_uma/mpascual/execs/lpqknorm_mri/sweep_preflight}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lpqknorm}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sweep_preflight}"

echo "=========================================="
echo "LP-QKNORM — 30-MIN PREFLIGHT ON exa"
echo "=========================================="
echo "Started:      $(date)"
echo "Hostname:     $(hostname)"
echo "Repo:         ${REPO_SRC}"
echo "H5:           ${H5_PATH}"
echo "Run dir:      ${RUN_DIR}"
echo "Condition:    p=${P_VAL}  fold=${FOLD}"
echo "Epochs:       ${MAX_EPOCHS}   (hard stop at ${TRAIN_BUDGET_SEC}s)"
echo "Batch:        ${BATCH_SIZE} × accumulate=${ACCUMULATE_GRAD_BATCHES}  precision=${PRECISION}"
echo "Workers:      ${NUM_WORKERS}"
echo ""

# -----------------------------------------------------------------------------
# Conda activation — mirror scripts/slurm/train_worker.sh
# -----------------------------------------------------------------------------
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done
[ "${module_loaded}" -eq 0 ] && echo "[env] No conda module loaded; assuming conda on PATH."

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
else
    source activate "${CONDA_ENV_NAME}"
fi

echo "[python] $(which python)"
python -c "import torch, lpqknorm; print('torch', torch.__version__, 'lpqknorm', lpqknorm.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
cd "${REPO_SRC}"
[ -f "${H5_PATH}" ] || { echo "[FAIL] HDF5 not found: ${H5_PATH}"; exit 1; }

LOG_DIR="${RUN_DIR}/logs"
HYDRA_DIR="${RUN_DIR}/hydra_logs/preflight_p${P_VAL}_f${FOLD}_$(date +%Y%m%dT%H%M%S)"
mkdir -p "${LOG_DIR}" "${HYDRA_DIR}"

# Per-task GPU envelope log
GPU_CSV="${LOG_DIR}/gpu_preflight_p${P_VAL}_f${FOLD}_$(date +%Y%m%dT%H%M%S).csv"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv -l 30 > "${GPU_CSV}" 2>/dev/null &
GPU_PID=$!
trap '[ -n "${GPU_PID:-}" ] && kill "${GPU_PID}" 2>/dev/null || true' EXIT
echo "[gpu-monitor] PID=${GPU_PID}, output=${GPU_CSV}"

# -----------------------------------------------------------------------------
# Train — wrapped in `timeout` so the session can still write a post-hoc
# probe before the 30-min wall clock.  On timeout the exit status is 124;
# we treat it as "partial success" and continue to probe + summary.
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=========================================="
echo "TRAINING (budget ${TRAIN_BUDGET_SEC}s)"
echo "=========================================="
set +e
timeout --signal=SIGINT --kill-after=30s "${TRAIN_BUDGET_SEC}s" \
  python -u -m lpqknorm.cli.train \
    experiment="${EXPERIMENT_NAME}" \
    model.p="${P_VAL}" \
    data.fold="${FOLD}" \
    data.use_mock=false \
    data.h5_path="${H5_PATH}" \
    data.augment=false \
    run_dir="${RUN_DIR}" \
    training.seed="${SEED}" \
    training.num_workers="${NUM_WORKERS}" \
    training.batch_size="${BATCH_SIZE}" \
    training.precision="${PRECISION}" \
    training.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
    training.max_epochs="${MAX_EPOCHS}" \
    model.use_checkpoint=true \
    hydra.run.dir="${HYDRA_DIR}" \
    2>&1
TRAIN_RC=$?
set -e
echo "[train] exit rc=${TRAIN_RC}  (124 = timeout — expected for a preflight)"

# Surface Hydra log in case of silent failure during config composition.
if [ -f "${HYDRA_DIR}/train.log" ]; then
    echo "----- hydra train.log (tail 120) -----"
    tail -n 120 "${HYDRA_DIR}/train.log" || true
    echo "----- end train.log -----"
fi

# -----------------------------------------------------------------------------
# Probe — run on whatever checkpoint was written (best_val_dice preferred).
# -----------------------------------------------------------------------------
RUN_SUBDIR="${RUN_DIR}/p=${P_VAL}/fold=${FOLD}/seed=${SEED}"
CKPT=""
for name in best_val_dice.ckpt last.ckpt; do
    if [ -f "${RUN_SUBDIR}/checkpoints/${name}" ]; then
        CKPT="${RUN_SUBDIR}/checkpoints/${name}"
        break
    fi
done

if [ -n "${CKPT}" ]; then
    echo "=========================================="
    echo "PROBE (budget ${PROBE_BUDGET_SEC}s) — ${CKPT}"
    echo "=========================================="
    set +e
    timeout --signal=SIGINT --kill-after=20s "${PROBE_BUDGET_SEC}s" \
      python -u -m lpqknorm.cli.probe \
        --checkpoint "${CKPT}" \
        --output-dir "${RUN_SUBDIR}/probes" \
        --epoch-tag "preflight" \
        --device "cuda" \
        --n-probe-samples 16 \
        2>&1
    PROBE_RC=$?
    set -e
    echo "[probe] exit rc=${PROBE_RC}"
else
    echo "[warn] No checkpoint found under ${RUN_SUBDIR}/checkpoints/; skipping probe."
    PROBE_RC="skipped"
fi

# -----------------------------------------------------------------------------
# Folder-population check — the whole point of the preflight.
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "POPULATED-FOLDER CHECK"
echo "=========================================="
REQUIRED_SUBDIRS=(
    metrics
    checkpoints
    gradient_stats
    attention_stats
    predictions
    probes
)
MISSING=0
for sub in "${REQUIRED_SUBDIRS[@]}"; do
    d="${RUN_SUBDIR}/${sub}"
    if [ -d "${d}" ]; then
        n=$(find "${d}" -type f ! -name ".keep" | wc -l)
        if [ "${n}" -gt 0 ]; then
            printf '  OK    %-18s  %d files\n' "${sub}" "${n}"
        else
            printf '  EMPTY %-18s\n' "${sub}"
            MISSING=$(( MISSING + 1 ))
        fi
    else
        printf '  MISS  %-18s\n' "${sub}"
        MISSING=$(( MISSING + 1 ))
    fi
done

# Specific artefacts that must exist for the downstream dice-vs-p analysis
echo ""
echo "Critical artefact presence:"
for rel in \
    metrics/test_per_patient.parquet \
    metrics/val_per_patient.parquet \
    checkpoints/best_val_dice.ckpt \
    probes/alpha_trajectory.jsonl \
    gradient_stats/layer_norms.parquet
do
    if [ -s "${RUN_SUBDIR}/${rel}" ]; then
        sz=$(stat -c%s "${RUN_SUBDIR}/${rel}" 2>/dev/null || echo 0)
        printf '  OK   %-40s  (%d B)\n' "${rel}" "${sz}"
    else
        printf '  MISS %-40s\n' "${rel}"
        MISSING=$(( MISSING + 1 ))
    fi
done

# NaN scan on train_steps.jsonl (the production failure mode signature).
NAN_SCAN="N/A"
if [ -f "${RUN_SUBDIR}/metrics/train_steps.jsonl" ]; then
    NAN_SCAN=$(python - <<PY
import json
n = 0
with open("${RUN_SUBDIR}/metrics/train_steps.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        for v in r.values():
            if isinstance(v, float) and v != v:
                n += 1; break
print(n)
PY
)
fi
echo "  NaN steps in train_steps.jsonl: ${NAN_SCAN}   (must be 0)"

echo ""
echo "=========================================="
if [ "${MISSING}" -eq 0 ] && [ "${NAN_SCAN}" = "0" ]; then
    echo "PREFLIGHT: PASS"
    echo "Ready to submit the full sweep with: bash scripts/slurm/launch.sh"
else
    echo "PREFLIGHT: FAIL  (${MISSING} folder(s)/artefact(s) missing, NaN=${NAN_SCAN})"
    echo "Do NOT submit the sweep until every folder populates and NaN==0."
fi
echo "Artefacts: ${RUN_SUBDIR}"
echo "Finished:  $(date)"
echo "=========================================="

# Exit status: 0 on clean pass, 2 on preflight failure.
if [ "${MISSING}" -eq 0 ] && [ "${NAN_SCAN}" = "0" ]; then
    exit 0
else
    exit 2
fi
