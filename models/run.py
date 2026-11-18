"""Select released predictions or weights by model and target."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from models.release import main as release_main, registry, verify_selected
from reproduction.io import check_output_location, read_json, safe_path, sha256, write_json


PROJECT = Path(__file__).resolve().parents[1]
TARGETS = ("k562", "rpe1", "hepg2", "jurkat")
MODELS = ("ridge", "mlp", "morph", "state", "scgpt", "biolord", "scgen")
DEFAULT_IDS = {
    "ridge": "ridge/matched/3to1/{target}/seed{repeat}/source_only",
    "mlp": "mlp/matched/3to1/{target}/seed{repeat}/source_only",
    "morph": "morph/within/{target}/fold{repeat}",
    "state": "state/source_only/{target}/seed0",
    "scgpt": "scgpt/within/{target}/seed{repeat}",
    "biolord": "biolord/transfer/{target}/seed{repeat}",
}


def identifier(model, target, repeat):
    if model == "state" and repeat != 0:
        raise ValueError("The released STATE source-only example has repeat 0 only.")
    return DEFAULT_IDS[model].format(target=target, repeat=repeat)


def interpreter(model, saved=False):
    if saved or model in ("ridge", "mlp"):
        return Path(sys.executable).absolute()
    return PROJECT / f"models/{model}/.venv/bin/python"


def _environment():
    cache = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "perturbation-model-runtime"
    return {**os.environ, "PYTHONPATH": str(PROJECT), "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1", "WANDB_MODE": "disabled", "MPLBACKEND": "Agg",
            "XDG_CACHE_HOME": str(cache), "HF_HOME": str(cache / "huggingface"),
            "TORCH_HOME": str(cache / "torch"), "MPLCONFIGDIR": str(cache / "matplotlib")}


def selection(args):
    targets = list(TARGETS) if args.target == "all" else [args.target]
    output = args.out_dir.resolve()
    python = interpreter(args.model, args.saved)
    device = "cpu" if args.device == "auto" and args.model in ("ridge", "mlp", "biolord") else args.device
    plan = {"model": args.model, "targets": targets, "repeat": args.repeat,
            "saved": args.saved, "execute": args.execute, "device": device,
            "interpreter": str(python), "environment_ready": python.is_file(),
            "out_dir": str(output), "selections": []}
    if not python.is_file():
        plan["setup"] = f"uv sync --locked --project models/{args.model}"
    if args.checkpoint:
        if args.model != "scgen" or args.saved or len(targets) != 1 or not args.inputs:
            raise ValueError("New scGen fits require --model scgen, one --target, --checkpoint, and --inputs.")
        if args.results_dir:
            raise ValueError("--results_dir is not used with a new checkpoint.")
        if args.limit_perts is not None or args.limit_controls is not None:
            raise ValueError("Sample limits apply only to released checkpoint recipes.")
        checkpoint, inputs = args.checkpoint.resolve(), args.inputs.resolve()
        protected = [checkpoint if checkpoint.is_dir() else checkpoint.parent, inputs.parent]
        if any(output.is_relative_to(path) for path in protected):
            raise ValueError("Outputs must be outside the checkpoint and input directories.")
        command = ["--model", "scgen", "--checkpoint", str(checkpoint), "--inputs", str(inputs),
                   "--target", targets[0], "--seed", str(args.repeat), "--device", device,
                   "--batch_size", str(args.batch_size), "--control_batches", str(args.control_batches),
                   "--out_dir", str(output / targets[0]), "--execute"]
        if args.trust_checkpoint:
            command.append("--trust_checkpoint")
        plan["selections"].append({"target": targets[0], "experiment": "new_binary_fit",
                                   "module": "models.predict", "arguments": command})
        return plan, None, None
    if args.inputs:
        raise ValueError("--inputs requires a new scGen --checkpoint.")
    if args.model == "scgen" and not args.saved:
        raise ValueError("Historical scGen weights were not retained. Use --saved, or --checkpoint and --inputs for a new fit.")
    if args.saved and (args.limit_perts is not None or args.limit_controls is not None):
        raise ValueError("Saved prediction copies do not accept inference sample limits.")
    root = (args.results_dir or PROJECT / (
        "results/predictions" if args.saved else "checkpoints/released-models"
    )).resolve()
    check_output_location(root, output)
    if not (root / "manifest.json").is_file():
        kind = "saved predictions" if args.saved else "model weights"
        raise FileNotFoundError(f"Install the {kind} first, or select their location with --results_dir: {root}")
    index, files = registry(root)
    plan["results_dir"] = str(root)
    for target in targets:
        if args.model == "scgen":
            relative = f"predictions/scgen/{target}/repeat{args.repeat}"
            required = [f"{relative}/predictions.npz", f"{relative}/metadata.json"]
            if any(name not in files or not safe_path(root, name).is_file() for name in required):
                raise ValueError(f"The complete saved scGen cohort is missing for {target}, repeat {args.repeat}.")
            metadata_path = safe_path(root, required[1])
            if sha256(metadata_path) != files[required[1]]["sha256"]:
                raise ValueError("Saved scGen metadata checksum mismatch.")
            metadata = read_json(metadata_path)
            if (metadata.get("model"), metadata.get("target"), metadata.get("repeat")) != (
                args.model, target, args.repeat
            ):
                raise ValueError("Saved scGen metadata does not match the selected experiment.")
            plan["selections"].append({"target": target, "experiment": metadata["protocol"],
                                       "saved": relative, "module": "native_scgen_saved"})
            continue
        name = identifier(args.model, target, args.repeat)
        matches = [row for row in index["records"] if row["id"] == name]
        if len(matches) != 1:
            raise ValueError(f"The requested released experiment is missing or duplicated: {name}")
        record = matches[0]
        if (record.get("model"), record.get("target"), record.get("repeat")) != (
            args.model, target, args.repeat
        ):
            raise ValueError("Released experiment metadata does not match the model selection.")
        if args.saved and not record.get("saved"):
            raise ValueError(f"Saved predictions are not available for {name}.")
        recipe = record.get("inference")
        if not args.saved:
            if not recipe:
                raise ValueError(f"No checkpoint was retained for {name}; saved predictions are not an inference fallback.")
            if (recipe.get("model"), recipe.get("target"), recipe.get("seed")) != (
                args.model, target, args.repeat
            ):
                raise ValueError("Inference recipe does not match the selected model, target, and repeat.")
        command = ["--bundle", str(root), "--id", name, "--out_dir", str(output / target)]
        if args.saved:
            command.append("--saved")
        else:
            command += ["--execute", "--device", device, "--batch_size", str(args.batch_size),
                        "--control_batches", str(args.control_batches)]
            if args.trust_checkpoint:
                command.append("--trust_checkpoint")
            for key in ("limit_perts", "limit_controls"):
                if getattr(args, key) is not None:
                    command += [f"--{key}", str(getattr(args, key))]
        plan["selections"].append({"target": target, "id": name, "saved": record.get("saved"),
                                   "experiment": record.get("experiment"),
                                   "setting": record.get("setting"), "direction": record.get("direction"),
                                   "module": "models.release", "arguments": command})
    return plan, root, files


def run(args):
    plan, source, files = selection(args)
    if args.dry_run or not (args.saved or args.execute):
        print(json.dumps(plan, indent=2))
        return 0
    output = Path(plan["out_dir"])
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if args.saved:
        for item in plan["selections"]:
            verify_selected(source, item, files, saved=True)
        for item in plan["selections"]:
            if item["module"] == "native_scgen_saved":
                from reproduction.predictions import Predictions
                Predictions.load(safe_path(source, item["saved"])).save(output / item["target"])
            else:
                release_main(item["arguments"])
    else:
        from benchmarks._runtime import require_compute
        require_compute("Checkpoint inference")
        python = Path(plan["interpreter"])
        if not python.is_file():
            raise FileNotFoundError(f"Prepare the matching environment first: {plan['setup']}")
        for item in plan["selections"]:
            subprocess.run([str(python), "-B", "-m", item["module"], *item["arguments"]],
                           cwd=PROJECT, env=_environment(), check=True)
    write_json(output / "selection.json", {**plan, "training_performed": False,
                                         "checkpoint_inference": not args.saved})
    print(json.dumps({"output": str(output), "targets": plan["targets"],
                      "saved": args.saved, "training_performed": False}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--target", choices=(*TARGETS, "all"), default="all")
    parser.add_argument("--repeat", type=int, choices=range(5), default=0, help="Seed or CV fold; STATE source-only has repeat 0")
    parser.add_argument("--out_dir", type=Path, required=True, help="New output root; each target gets its own subdirectory")
    parser.add_argument("--results_dir", type=Path, help="Override the installed predictions or weights directory")
    parser.add_argument("--saved", action="store_true", help="Copy the saved predictions without loading a model")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Run inference in the matching prepared environment")
    action.add_argument("--dry_run", action="store_true", help="Inspect selection without copying predictions or running inference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint from a new binary scGen fit")
    parser.add_argument("--inputs", type=Path, help="Prepared input request for the new scGen fit")
    parser.add_argument("--limit_perts", type=int)
    parser.add_argument("--limit_controls", type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--control_batches", type=int, default=5)
    args = parser.parse_args(argv)
    if args.saved and args.execute:
        parser.error("Choose saved predictions or checkpoint inference, not both.")
    if min(args.batch_size, args.control_batches) < 1 or any(
        value is not None and value < 1 for value in (args.limit_perts, args.limit_controls)
    ):
        parser.error("Batch and sample counts must be positive.")
    try:
        return run(args)
    except subprocess.CalledProcessError as exc:
        parser.exit(2, f"ERROR: Native inference exited with status {exc.returncode}.\n")
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
