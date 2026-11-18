"""Run bounded real-checkpoint operational checks in the prepared model environments."""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", choices=("state", "scgpt", "all"), default="all")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Native checkpoint smoke inference requires a SLURM allocation.")
    root = Path(__file__).resolve().parents[1]
    targets = ("state", "scgpt") if args.model == "all" else (args.model,)
    for model in targets:
        python = root / "models" / model / ".venv/bin/python"
        if not python.is_file():
            parser.error(f"Missing {python}; run uv sync --locked --project models/{model} before submission.")
        identifier = "state/source_only/k562/seed0" if model == "state" else "scgpt/within/k562/seed0"
        controls = 32 if model == "state" else 1
        command = [str(python), "-B", "-m", "models.release", "--bundle", str(args.bundle),
                   "--id", identifier, "--out_dir", str(args.out_dir / model),
                   "--limit_perts", "1", "--limit_controls", str(controls),
                   "--batch_size", "1", "--control_batches", "1", "--device", "cuda:0",
                   "--trust_checkpoint", "--execute"]
        environment = {
            **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled",
            "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1",
            "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(root),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "MPLCONFIGDIR": str(Path(os.environ.get("TMPDIR", "/tmp")) / f"{model}-matplotlib"),
            "NUMBA_CACHE_DIR": str(Path(os.environ.get("TMPDIR", "/tmp")) / f"{model}-numba"),
            "XDG_CACHE_HOME": str(Path(os.environ.get("TMPDIR", "/tmp")) / f"{model}-cache")}
        subprocess.run(
            [str(python), "-B", "-m", "models.runtime", "--model", model,
             "--import_check", "--gpu_check", "--device", "cuda:0"],
            cwd=root, check=True, env=environment)
        subprocess.run(command, cwd=root, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
