#!/bin/bash
#SBATCH --job-name=lpqknorm_sweep
#SBATCH --array=0-17
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#
# Phase 3 sweep: p ∈ {vanilla, 2.0, 2.5, 3.0, 3.5, 4.0} × fold ∈ {0, 1, 2}
# = 18 runs total.
#
# Mapping:
#   SLURM_ARRAY_TASK_ID // 3 = condition index (0..5)
#   SLURM_ARRAY_TASK_ID %  3 = fold index      (0..2)
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/submit_sweep.sbatch
#
# Dry-run (no allocation):
#   sbatch --test-only scripts/submit_sweep.sbatch

set -euo pipefail

CONDITIONS=(vanilla 2.0 2.5 3.0 3.5 4.0)
CONDITION_IDX=$((SLURM_ARRAY_TASK_ID / 3))
FOLD=$((SLURM_ARRAY_TASK_ID % 3))
P=${CONDITIONS[$CONDITION_IDX]}

echo "=============================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: p=${P}, fold=${FOLD}"
echo "Host: $(hostname)"
echo "Date: $(date -u --iso-8601=seconds)"
echo "=============================================="

# Picasso-specific: load Singularity module
module load singularity 2>/dev/null || true

# Map p value to Hydra override
if [ "$P" = "vanilla" ]; then
    P_OVERRIDE="model.p=null"
else
    P_OVERRIDE="model.p=${P}"
fi

# Run training via the CLI entry point
singularity exec --nv \
    "${LPQKNORM_SIF:-/path/to/lpqknorm.sif}" \
    /opt/conda/envs/lpqknorm/bin/python -m lpqknorm.cli.train \
    "${P_OVERRIDE}" \
    data.fold="${FOLD}" \
    data.use_mock=false \
    training.seed=20260216

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date -u --iso-8601=seconds)"
