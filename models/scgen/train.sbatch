#!/bin/bash
#SBATCH --job-name=scgen-transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
cd "$PROJECT_ROOT"

PYTHON="$SLURM_SUBMIT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing uv environment: $PYTHON" >&2
    echo "Run 'uv sync --locked' from the repository root before submitting." >&2
    exit 1
fi

ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    ARGS=(--shard "$SLURM_ARRAY_TASK_ID")
fi
"$PYTHON" train.py --execute "${ARGS[@]}" "$@"
