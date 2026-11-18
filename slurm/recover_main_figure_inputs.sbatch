#!/bin/bash
#SBATCH --job-name=figure-input-recovery
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$PROJECT_ROOT"
PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing $PYTHON; run uv sync --locked --extra analysis before submission." >&2
    exit 1
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" scripts/recover_main_figure_inputs.py "$@"
