#!/bin/bash
#SBATCH --job-name=perturbation-inputs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$PROJECT_ROOT"
PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing $PYTHON; run uv sync --locked before submission." >&2
    exit 1
fi
export TMPDIR="${TMPDIR:-/tmp}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PYTHON" scripts/package_training_inputs.py "$@"
