#!/bin/bash
#SBATCH --job-name=perturbation-pack
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$PROJECT_ROOT"
PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
NATIVE="$PROJECT_ROOT/models/state/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing $PYTHON; run uv sync --locked before submission." >&2
    exit 1
fi
if [[ ! -x "$NATIVE" ]]; then
    echo "Missing $NATIVE; run uv sync --locked --project models/state before submission." >&2
    exit 1
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PYTHON" scripts/run_workflow.py --python "$NATIVE" distribution "$@"
