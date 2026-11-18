"""Dispatch a workflow using the selected, already prepared uv environment."""

import argparse
from pathlib import Path
import subprocess
import sys
import os
import tempfile


MODULES = {"predict": "models.predict", "prepare": "data.inference",
           "training-inputs": "data.training_inputs",
           "distribution": "data.distribution", "release": "models.release",
           "prepare-tahoe": "data.tahoe_inference",
           "controls": "data.controls", "onehot": "data.onehot",
           "decompose": "decomposition.run", "pipeline": "benchmarks.pipeline",
           "artifacts": "models.artifacts", "baseline": "scripts.run_benchmark",
           "train-morph": "models.morph.train", "train-state": "models.state.train",
           "train-scgpt": "models.scgpt.train", "train-biolord": "models.biolord.train",
           "train-scgen": "models.scgen.train"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("workflow", choices=MODULES)
    args, rest = parser.parse_known_args()
    if not args.python.is_file() or args.python.parent.parent.name != ".venv":
        parser.error("--python must point to a prepared project .venv/bin/python")
    root = Path(__file__).resolve().parents[1]
    cache = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "perturbation-runtime-cache"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled",
                   "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(root),
                   "XDG_CACHE_HOME": str(cache), "HF_HOME": str(cache / "huggingface"),
                   "TORCH_HOME": str(cache / "torch"), "MPLCONFIGDIR": str(cache / "matplotlib")}
    return subprocess.call([str(args.python), "-B", "-m", MODULES[args.workflow], *rest],
                           cwd=root, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
