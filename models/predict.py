"""Predict from checkpoints or inspect/copy saved predictions without retraining."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .inference_io import prediction_labels, read_inputs, write_predictions
from reproduction.io import new_directory, sha256, write_json


def predict_checkpoint(args):
    data = read_inputs(args.inputs)
    if args.model in ("ridge", "mlp"):
        from .checkpoints import Predictor
        model = Predictor(args.checkpoint)
        if ((args.model == "ridge" and model.kind != "ridge") or
                (args.model == "mlp" and (model.kind not in ("mlp", "ensemble") or
                 (model.kind == "ensemble" and any(m.kind != "mlp" for m in model.members))))):
            raise ValueError("Requested model family differs from the checkpoint")
        axis = str(data["descriptor_axis_id"].item())
        values = data.get("features", data.get("descriptors"))
        if values is None:
            raise ValueError("Baseline inputs require features or descriptors")
        predicted = model(values, axis)
        names, metadata = model.feature_names, {
            "architecture": model.kind, "descriptor_axis_id": axis,
            "checkpoint_sha256": sha256(Path(args.checkpoint) / "model.json")}
        target = model.metadata.get("target", model.metadata.get("cell_line"))
        if args.target and target and args.target != target:
            raise ValueError("Requested target differs from the fitted checkpoint")
        metadata["target"] = target or args.target
        for key in ("experiment", "setting", "direction", "fold", "repeat"):
            if key in model.metadata:
                metadata[key] = model.metadata[key]
    else:
        from . import native_inference
        predicted, names, metadata = getattr(native_inference, args.model)(args.checkpoint, data, args)
        metadata["checkpoint_sha256"] = sha256(args.checkpoint / "model.pt" if args.checkpoint.is_dir() else args.checkpoint)
    metadata.update(model=args.model, seed=args.seed, device=args.device, output_space="response_delta",
                    input_sha256=sha256(args.inputs),
                    inference_only=True, retrained=False,
                    replay_equivalence="Requires identical prepared inputs and protocol-specific RNG state.")
    reference_path = getattr(args, "expected", None)
    if reference_path:
        with np.load(reference_path, allow_pickle=False) as reference:
            if "feature_names" not in reference or not np.array_equal(reference["feature_names"].astype(str), names):
                raise ValueError("Replay comparison requires an expected archive with identical ordered gene labels")
            labels = prediction_labels(reference).tolist()
            if len(labels) != len(set(labels)):
                raise ValueError("Duplicate expected perturbation labels")
            requested = data["test_perts"].astype(str).tolist()
            if set(requested)-set(labels):
                raise ValueError("Expected predictions do not cover every requested perturbation")
            expected = reference["predictions"][[labels.index(name) for name in requested]]
        atol, rtol = getattr(args, "atol", 2e-5), getattr(args, "rtol", 2e-5)
        np.testing.assert_allclose(predicted, expected, atol=atol, rtol=rtol)
        metadata["checkpoint_replay_check"] = {
            "expected_sha256": sha256(reference_path), "atol": atol, "rtol": rtol,
            "n_rows": len(predicted), "max_abs_difference": float(np.max(np.abs(predicted-expected)))}
    return write_predictions(args.out_dir, data, predicted, names, metadata)


def saved_predictions(path, output=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        if "predictions" not in saved or saved["predictions"].ndim != 2:
            raise ValueError("Saved archive needs a prediction matrix")
        try:
            labels = prediction_labels(saved)
        except ValueError as exc:
            if "Object arrays cannot be loaded" in str(exc):
                raise ValueError("Legacy object-encoded identifiers require the trusted-source "
                                 "distribution export; see README.md.") from exc
            raise
        if len(set(labels)) != len(labels) or len(labels) != len(saved["predictions"]):
            raise ValueError("Invalid perturbation axis")
        if not np.isfinite(saved["predictions"]).all():
            raise ValueError("Nonfinite saved predictions")
        description = {"n_perturbations": len(labels), "n_genes": saved["predictions"].shape[1],
                       "has_gene_names": "feature_names" in saved,
                       "has_truth": "true_deltas" in saved, "sha256": sha256(path)}
    if output:
        output = Path(output).resolve()
        if output.is_relative_to(path.parent.resolve()):
            raise ValueError("Export must be outside the source archive directory")
        output = new_directory(output)
        shutil.copyfile(path, output / "predictions.npz")
        if sha256(output / "predictions.npz") != description["sha256"]:
            raise ValueError("Archive changed during export")
        write_json(output / "metadata.json", {**description, "mode": "saved_prediction_copy",
                                             "checkpoint_inference": False})
    return description


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ridge", "mlp", "morph", "state", "scgpt", "biolord", "scgen"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--saved", type=Path, help="Existing NPZ predictions; no model is loaded")
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inputs", type=Path, help="Prepared NPZ with explicit axes")
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="Run checkpoint inference rather than print a plan")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--architecture", choices=("benchmark_mean_conditioned", "upstream_pickled_model"))
    parser.add_argument("--source_dir", type=Path, help="Matching upstream MORPH source checkout")
    parser.add_argument("--trust_checkpoint", action="store_true")
    parser.add_argument("--control_samples", type=int, default=100)
    parser.add_argument("--cell_set_len", type=int, default=32)
    parser.add_argument("--control_batches", type=int, default=5)
    parser.add_argument("--perturbation_map", type=Path)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--token_policy", type=Path, help="Recorded scGPT vocabulary-default protocol")
    parser.add_argument("--training_data", type=Path)
    parser.add_argument("--go_features", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--expected", type=Path, help="Optional gene-labeled reference predictions for replay verification")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    args = parser.parse_args(argv)
    try:
        if args.saved:
            print(json.dumps(saved_predictions(args.saved, args.out_dir), indent=2))
            return
        if not args.inputs or not args.out_dir:
            parser.error("--checkpoint requires --inputs and a new --out_dir")
        if min(args.control_samples, args.cell_set_len, args.control_batches, args.batch_size) < 1:
            parser.error("Sample and batch sizes must be positive")
        if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
            parser.error("Replay tolerances must be finite and nonnegative")
        if args.out_dir.exists():
            raise FileExistsError(args.out_dir)
        if not args.checkpoint.exists() or not args.inputs.is_file():
            raise FileNotFoundError("Checkpoint and prepared input files must exist")
        protected = [args.inputs.resolve().parent,
                     args.checkpoint.resolve() if args.checkpoint.is_dir() else args.checkpoint.resolve().parent]
        if any(args.out_dir.resolve().is_relative_to(p) for p in protected):
            raise ValueError("Prediction outputs must be outside all input/checkpoint directories")
        if not args.execute:
            print(json.dumps({"model": args.model, "checkpoint": str(args.checkpoint),
                              "inputs": str(args.inputs), "output": str(args.out_dir),
                              "device": args.device, "execute": False}, indent=2))
            return
        from benchmarks._runtime import require_compute, resolve_device
        require_compute("Checkpoint inference")
        args.device = "cpu" if args.model == "biolord" and args.device == "auto" else resolve_device(args.device)
        print(predict_checkpoint(args))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, AssertionError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
