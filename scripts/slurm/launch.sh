#!/usr/bin/env bash
# =============================================================================
# LP-QKNORM — PICASSO LAUNCHER
#
# Submits the p × fold sweep as a single SLURM array.  Each array task
# trains one (p, fold) pair on a dedicated GPU.  Job names embed the
# condition (e.g. "lpqk_p2.5_f1"), so `squeue` shows the full sweep at
# a glance.
#
# Usage (from the Picasso login node):
#   cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/lp-qknorm-mri
#   bash scripts/slurm/launch.sh
#
# Options:
#   --config <yaml>   Override Picasso config (default: configs/picasso/default.yaml)
#   --dry-run         Resolve config, print sbatch command, but do not submit
#   --no-preflight    Skip setup.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================="
echo "LP-QKNORM — PICASSO LAUNCHER"
echo "=========================================="
echo "Time: $(date)"
echo ""

# -----------------------------------------------------------------------------
# Parse args
# -----------------------------------------------------------------------------
PICASSO_CFG="${REPO_ROOT}/configs/picasso/default.yaml"
DRY_RUN=0
SKIP_PREFLIGHT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      PICASSO_CFG="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --no-preflight) SKIP_PREFLIGHT=1; shift ;;
        *) echo "[warn] ignoring unknown arg: $1"; shift ;;
    esac
done

if [ ! -f "${PICASSO_CFG}" ]; then
    echo "[FAIL] Picasso config not found: ${PICASSO_CFG}"
    exit 1
fi

# Conda for the OmegaConf parse below.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lpqknorm}"
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || source activate "${CONDA_ENV_NAME}"
fi

# -----------------------------------------------------------------------------
# Resolve picasso config → export variables for the worker.
# -----------------------------------------------------------------------------
echo "Resolving Picasso configuration..."
eval "$(
python - <<PY
from omegaconf import OmegaConf
import shlex, sys

cfg = OmegaConf.load("${PICASSO_CFG}")

# Cast lists → space-separated strings (for bash arrays).
p_values = " ".join(str(p) for p in cfg.sweep.p_values)
folds    = [int(f) for f in cfg.sweep.folds]
n_folds  = len(folds)
n_p      = len(cfg.sweep.p_values)
n_tasks  = n_p * n_folds

exports = {
    "REPO_SRC":        str(cfg.paths.repo_src),
    "H5_PATH":         str(cfg.paths.h5_path),
    "RUN_DIR_BASE":    str(cfg.paths.run_dir),
    "CONDA_ENV_NAME":  str(cfg.paths.conda_env),
    "EXPERIMENT_NAME": str(cfg.experiment.name),
    "P_VALUES":        p_values,
    "N_FOLDS":         str(n_folds),
    "N_TASKS":         str(n_tasks),
    "SEED":            str(cfg.training.seed),
    "NUM_WORKERS":     str(cfg.training.num_workers),
    "BATCH_SIZE":      str(cfg.training.batch_size),
    "PRECISION":       str(cfg.training.precision),
    "SLURM_PARTITION": str(cfg.slurm.partition),
    "SLURM_GRES":      str(cfg.slurm.gres),
    "SLURM_CPUS":      str(cfg.slurm.cpus_per_task),
    "SLURM_MEM":       str(cfg.slurm.mem),
    "SLURM_TIME":      str(cfg.slurm.time),
    "SLURM_CONSTRAINT": str(cfg.slurm.constraint or ""),
}
for k, v in exports.items():
    print(f"export {k}={shlex.quote(v)}")
PY
)"

# Validate required exports.
: "${REPO_SRC:?}" "${H5_PATH:?}" "${RUN_DIR_BASE:?}" "${CONDA_ENV_NAME:?}"
: "${P_VALUES:?}" "${N_FOLDS:?}" "${N_TASKS:?}"

RUN_DIR="${RUN_DIR_BASE}"
SLURM_LOG_DIR="${RUN_DIR}/logs"

# On a dry-run we never try to touch the Picasso-only filesystem, so the
# launcher can be exercised from a developer workstation.
if [ "${DRY_RUN}" -eq 0 ]; then
    mkdir -p "${SLURM_LOG_DIR}"
    # Snapshot the resolved config alongside the run for reproducibility.
    cp "${PICASSO_CFG}" "${RUN_DIR}/picasso_config_snapshot.yaml"
fi

echo ""
echo "Resolved:"
echo "  Repo:          ${REPO_SRC}"
echo "  H5:            ${H5_PATH}"
echo "  Run dir:       ${RUN_DIR}"
echo "  Experiment:    ${EXPERIMENT_NAME}"
echo "  Conda env:     ${CONDA_ENV_NAME}"
echo "  p values:      ${P_VALUES}"
echo "  n_folds:       ${N_FOLDS}"
echo "  n_tasks:       ${N_TASKS}"
echo "  seed:          ${SEED}"
echo "  batch:         ${BATCH_SIZE}  precision=${PRECISION}  workers=${NUM_WORKERS}"
echo ""

# -----------------------------------------------------------------------------
# Pre-flight (sanity-check imports + paths + split scheme).
# -----------------------------------------------------------------------------
if [ "${SKIP_PREFLIGHT}" -eq 0 ]; then
    echo "Running pre-flight checks..."
    bash "${SCRIPT_DIR}/setup.sh" "${PICASSO_CFG}"
    echo ""
fi

# -----------------------------------------------------------------------------
# Build an auxiliary "job-name map" so squeue output is interpretable.
# We cannot parameterize --job-name per array element natively, so we use
# the %a suffix and write a human-readable mapping file alongside logs.
# -----------------------------------------------------------------------------
if [ "${DRY_RUN}" -eq 1 ]; then
    MAP_FILE="/tmp/lpqk_task_map_$$.tsv"
else
    MAP_FILE="${SLURM_LOG_DIR}/task_map.tsv"
fi
python - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("${PICASSO_CFG}")
p_vals = list(cfg.sweep.p_values)
folds  = list(cfg.sweep.folds)
with open("${MAP_FILE}", "w") as fh:
    fh.write("task_id\tp\tfold\n")
    for i, p in enumerate(p_vals):
        for j, fold in enumerate(folds):
            tid = i * len(folds) + j
            fh.write(f"{tid}\t{p}\t{fold}\n")
print(f"[map] wrote {('${MAP_FILE}')}")
PY

# -----------------------------------------------------------------------------
# Submit array job.
# -----------------------------------------------------------------------------
ARRAY_MAX=$(( N_TASKS - 1 ))
JOB_NAME="lpqk_${EXPERIMENT_NAME}"

# Build sbatch args.  --partition, --constraint, --export are only added
# when they are explicitly set in the Picasso config; Picasso's default
# partition (and submission environment via --export=ALL) is otherwise
# used.  All sweep parameters are exported to the worker via --export=ALL
# because (a) they are already in the login-shell environment from the
# earlier `export` block, and (b) P_VALUES contains spaces, which would
# be ambiguous inside a comma-separated --export= value.
SBATCH_ARGS=(
    --array="0-${ARRAY_MAX}"
    --job-name="${JOB_NAME}"
    --gres="${SLURM_GRES}"
    --cpus-per-task="${SLURM_CPUS}"
    --mem="${SLURM_MEM}"
    --time="${SLURM_TIME}"
    --output="${SLURM_LOG_DIR}/${JOB_NAME}_%A_%a.out"
    --error="${SLURM_LOG_DIR}/${JOB_NAME}_%A_%a.err"
    --export=ALL
)

if [ -n "${SLURM_PARTITION}" ]; then
    SBATCH_ARGS+=( --partition="${SLURM_PARTITION}" )
fi

if [ -n "${SLURM_CONSTRAINT}" ]; then
    SBATCH_ARGS+=( --constraint="${SLURM_CONSTRAINT}" )
fi

SBATCH_ARGS+=( "${SCRIPT_DIR}/train_worker.sh" )

echo "Submitting array job ($N_TASKS tasks)..."
printf '  sbatch'
for a in "${SBATCH_ARGS[@]}"; do printf ' %q' "$a"; done
printf '\n\n'

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[dry-run] Not submitting; the map is in ${MAP_FILE}"
    exit 0
fi

# Capture sbatch output + exit status without pipefail tripping on us.
SBATCH_LOG="${SLURM_LOG_DIR}/sbatch_submit_$(date +%Y%m%dT%H%M%S).log"
set +e
SUBMIT_OUT="$(sbatch "${SBATCH_ARGS[@]}" 2>&1)"
SBATCH_RC=$?
set -e

# Always print + persist the raw sbatch output.  This used to be lost when
# the subsequent grep-pipeline failed under `set -o pipefail`.
echo "----- sbatch output (rc=${SBATCH_RC}) -----"
printf '%s\n' "${SUBMIT_OUT}"
echo "----- end sbatch output -----"
printf '%s\n' "${SUBMIT_OUT}" > "${SBATCH_LOG}"
echo "(saved to ${SBATCH_LOG})"

if [ "${SBATCH_RC}" -ne 0 ]; then
    echo "[FAIL] sbatch returned exit code ${SBATCH_RC}"
    exit "${SBATCH_RC}"
fi

# Extract job ID robustly.  Keep pipefail off for the extraction so a
# missing pattern doesn't kill the script without a diagnostic.
set +o pipefail
JOB_ID="$(printf '%s\n' "${SUBMIT_OUT}" | grep -oE 'job[[:space:]]+[0-9]+' | head -1 | grep -oE '[0-9]+' || true)"
if [ -z "${JOB_ID}" ]; then
    JOB_ID="$(printf '%s\n' "${SUBMIT_OUT}" | grep -oE '[0-9]+' | head -1 || true)"
fi
set -o pipefail

if [ -z "${JOB_ID}" ]; then
    echo "[FAIL] Could not extract job ID from sbatch output (see ${SBATCH_LOG})"
    exit 1
fi

# -----------------------------------------------------------------------------
# Rename each array element's job for human-readable squeue output.
# scontrol lets us set a per-element JobName after submission; this works on
# Picasso (SLURM >=20.11).  Failures are non-fatal.
# -----------------------------------------------------------------------------
if command -v scontrol >/dev/null 2>&1; then
    while IFS=$'\t' read -r tid p fold; do
        [ "$tid" = "task_id" ] && continue
        if [ "$p" = "vanilla" ]; then
            tag="vanilla"
        else
            tag="p${p}"
        fi
        scontrol update JobName="lpqk_${tag}_f${fold}" JobId="${JOB_ID}_${tid}" \
            2>/dev/null || true
    done < "${MAP_FILE}"
fi

echo ""
echo "=========================================="
echo "SUBMITTED"
echo "=========================================="
echo "Job ID:   ${JOB_ID}  (array 0-${ARRAY_MAX})"
echo "Map:      ${MAP_FILE}"
echo "Logs:     ${SLURM_LOG_DIR}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  squeue -j ${JOB_ID}"
echo ""
echo "Cancel all:"
echo "  scancel ${JOB_ID}"
