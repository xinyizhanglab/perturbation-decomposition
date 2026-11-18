"""Inspect or run a model from an installed self-contained artifact bundle."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from reproduction.io import read_json, safe_path, sha256, write_json


PATH_FLAGS = {"checkpoint", "perturbation_map", "vocab", "model_config", "source_dir",
              "training_data", "go_features", "token_policy"}


def registry(root):
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported model bundle manifest.")
    files = {entry["path"]: entry for entry in manifest["files"]}
    if len(files) != len(manifest["files"]) or "registry.json" not in files:
        raise ValueError("Bundle manifest has duplicate paths or lacks its registry.")
    path = safe_path(root, "registry.json")
    if sha256(path) != files["registry.json"]["sha256"]:
        raise ValueError("Bundle registry checksum differs from its manifest.")
    return read_json(path), files


def verify_selected(root, record, files, saved=False):
    paths = {record["saved"]} if saved else {
        value for key, value in record["inference"].items()
        if key in PATH_FLAGS | {"inputs", "controls"}}
    for relative in paths:
        path = safe_path(root, relative)
        names = [relative] if path.is_file() else [
            name for name in files if name.startswith(relative.rstrip("/") + "/")]
        if not names:
            raise ValueError(f"Missing declared inference companion: {relative}")
        for name in names:
            if name not in files or sha256(safe_path(root, name)) != files[name]["sha256"]:
                raise ValueError(f"Inference companion checksum mismatch: {name}")


def prepared_inputs(root, recipe, destination, limit_perts=None, limit_controls=None):
    with np.load(safe_path(root, recipe["inputs"]), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if limit_perts is not None:
        if limit_perts < 1:
            raise ValueError("Perturbation limits must be positive.")
        n = len(arrays["test_perts"])
        for key in ("test_perts", "features", "descriptors", "true_deltas"):
            if key in arrays and arrays[key].shape[0] == n:
                arrays[key] = arrays[key][:limit_perts]
    if "controls" in recipe:
        with np.load(safe_path(root, recipe["controls"]), allow_pickle=False) as archive:
            if not np.array_equal(arrays["feature_names"], archive["feature_names"]):
                raise ValueError("Shared controls and inference inputs have different gene axes.")
            arrays["controls"] = archive["controls"].copy()
            arrays["control_mean"] = archive["control_mean"].copy()
        if limit_controls is not None:
            if limit_controls < 1:
                raise ValueError("Control limits must be positive.")
            arrays["controls"] = arrays["controls"][:limit_controls]
    np.savez_compressed(destination, **arrays)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--saved", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust_checkpoint", action="store_true")
    parser.add_argument("--limit_perts", type=int, help="Bounded operational run, not full-cohort replay")
    parser.add_argument("--limit_controls", type=int, help="Bounded control pool, not historical replay")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--control_batches", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        index, files = registry(args.bundle)
        matches = [record for record in index["records"] if record["id"] == args.id]
        if len(matches) != 1:
            raise ValueError("Unknown or duplicate artifact ID.")
        record = matches[0]
        if args.saved:
            from models.predict import saved_predictions
            verify_selected(args.bundle, record, files, saved=True)
            print(json.dumps(saved_predictions(safe_path(args.bundle, record["saved"]), args.out_dir), indent=2))
            return
        recipe = record.get("inference")
        if not recipe:
            raise ValueError("No historical checkpoint is available; use --saved.")
        if not args.execute:
            print(json.dumps({"artifact": args.id, "inference": recipe, "device": args.device,
                              "execute": False, "self_contained": True}, indent=2))
            return
        if not args.out_dir:
            parser.error("--execute requires a new --out_dir")
        from benchmarks._runtime import require_compute
        from reproduction.io import check_output_location
        require_compute("Packaged checkpoint inference")
        check_output_location(args.bundle, args.out_dir)
        verify_selected(args.bundle, record, files)
        with tempfile.TemporaryDirectory(prefix="perturbation-inference-") as directory:
            inputs = prepared_inputs(args.bundle, recipe, Path(directory) / "inputs.npz",
                                     args.limit_perts, args.limit_controls)
            command = ["--model", recipe["model"], "--inputs", str(inputs), "--out_dir", str(args.out_dir),
                       "--device", args.device, "--seed", str(recipe["seed"]), "--target", recipe["target"],
                       "--batch_size", str(args.batch_size), "--control_batches", str(args.control_batches),
                       "--execute"]
            for flag in PATH_FLAGS:
                if flag in recipe:
                    command.extend(["--" + flag, str(safe_path(args.bundle, recipe[flag]))])
            if "architecture" in recipe:
                command.extend(["--architecture", recipe["architecture"]])
            if args.trust_checkpoint:
                command.append("--trust_checkpoint")
            from models.predict import main as predict
            predict(command)
            write_json(args.out_dir / "release.json", {
                "artifact_id": args.id, "bundle_manifest_sha256": sha256(args.bundle / "manifest.json"),
                "registry_sha256": files["registry.json"]["sha256"], "recipe": recipe,
                "limit_perts": args.limit_perts, "limit_controls": args.limit_controls,
                "device": args.device, "historical_replay_claimed": False,
                "mode": "bounded_operational_check" if args.limit_perts or args.limit_controls
                        else "checkpoint_inference",
            })
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
