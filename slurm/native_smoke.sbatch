#!/bin/bash
#SBATCH --job-name=perturbation-native-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PYTHON" scripts/native_smoke.py "$@"
