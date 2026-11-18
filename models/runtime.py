"""Check native inference dependencies without reading model weights or data."""

import argparse
import importlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import sys


REQUIRED = {
    "ridge": ("numpy",), "mlp": ("numpy",),
    "morph": ("numpy", "torch"),
    "state": ("numpy", "torch", "state.tx.models.state_transition"),
    "scgpt": ("numpy", "torch", "anndata", "scipy", "gears", "torch_geometric", "scgpt"),
    "biolord": ("numpy", "torch", "anndata", "scvi", "biolord"),
    "scgen": ("numpy", "torch", "anndata", "scvi", "scgen"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REQUIRED, required=True)
    parser.add_argument("--import_check", action="store_true", help="Import the actual SDK classes")
    parser.add_argument("--gpu_check", action="store_true", help="Probe CUDA kernels in an allocation")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    modules, missing = {}, []
    for name in REQUIRED[args.model]:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            spec = None
        modules[name] = spec.origin if spec else None
        if spec is None:
            missing.append(name)
    versions = {}
    for name in ("numpy", "torch", "scvi-tools", "scgen", "biolord", "scgpt", "pytorch-lightning",
                 "arc-state", "lightning", "transformers", "cell-gears", "torchtext", "torch-geometric"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    errors = {}
    if (args.import_check or args.gpu_check) and not missing:
        for name in REQUIRED[args.model]:
            try:
                importlib.import_module(name)
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        if args.model == "state" and not errors:
            try:
                from models.state.compat import install
                install()
            except Exception as exc:
                errors["recorded_state_compatibility"] = f"{type(exc).__name__}: {exc}"
    gpu = None
    if args.gpu_check and not missing and not errors:
        try:
            from models.gpu_runtime import check_gpu
            gpu = check_gpu(args.device, ("torch_scatter", "torch_sparse") if args.model == "scgpt" else ())
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            errors["gpu_runtime"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"python": sys.executable, "model": args.model, "modules": modules,
                      "versions": versions, "missing": missing, "import_errors": errors,
                      "imports_checked": args.import_check or args.gpu_check,
                      "gpu": gpu,
                      "scope": "SDK availability/import compatibility, not checkpoint replay equivalence"}, indent=2))
    if missing or errors:
        parser.exit(2, "Use the matching prepared uv environment; see README.md.\n")


if __name__ == "__main__":
    main()
