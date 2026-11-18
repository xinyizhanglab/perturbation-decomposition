"""Input checks, execution previews, and run records shared by benchmark CLIs."""

import json
import os
from pathlib import Path
import platform
from importlib.metadata import PackageNotFoundError, version


def require_compute(workload):
    if Path("/mnt/storage01/home").is_dir() and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(f"{workload} requires a SLURM allocation on this cluster.")


def resolve_device(requested):
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def prepare_run(args, paths):
    paths = [Path(p) for p in paths]
    missing = [str(p) for p in paths if not p.is_file()]
    args.device = resolve_device(args.device) if hasattr(args, "device") else "cpu"
    record = {"arguments": vars(args), "inputs": [str(p) for p in paths],
              "missing_inputs": missing, "python": platform.python_version(),
              "packages": {}}
    for name in ("numpy", "scikit-learn", "torch"):
        try:
            record["packages"][name] = version(name)
        except PackageNotFoundError:
            record["packages"][name] = None
    if args.dry_run:
        print(json.dumps(record, indent=2, default=str))
        return False
    if missing:
        raise FileNotFoundError("Missing benchmark inputs:\n  " + "\n  ".join(missing) +
                                "\nSee README.md for processed-data installation.")
    require_compute("Benchmark execution")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.json").open("x") as stream:
        json.dump(record, stream, indent=2, default=str)
        stream.write("\n")
    return True
