"""Shared command-line dispatch for the native transfer experiments."""

import argparse
import importlib.util
import os
from pathlib import Path


def train(model, argv=None):
    parser = argparse.ArgumentParser(description=f"{model} cross-cell-line transfer")
    stages = ("prepare", "train", "report", "calibration-prepare", "calibration-train",
              "calibration-report") if model == "biolord" else ("prepare", "train")
    parser.add_argument("--stage", choices=stages, default="train")
    parser.add_argument("--data_dir", type=Path, help="Installed genetic single-cell input root")
    parser.add_argument("--prepared_inputs", type=Path,
                        help="Installed frozen BioLORD/calibration bundle; skips repeat preprocessing")
    parser.add_argument("--reference_dir", required=True, type=Path, help="Prepared reference-input root")
    parser.add_argument("--out_dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/native_experiments")
    parser.add_argument("--experiment_dir", type=Path, help="Override the experiment subdirectory")
    parser.add_argument("--training_log_dir", type=Path)
    parser.add_argument("--go_file", type=Path)
    parser.add_argument("--task", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--execute", action="store_true", help="Execute inside a SLURM allocation")
    if model == "scgen":
        parser.add_argument("--save_checkpoints", action="store_true",
                            help="Export native weights for new binary fits (can require substantial storage)")
        parser.add_argument("--checkpoint_tasks", help="Optional comma-separated task IDs to retain")
    args = parser.parse_args(argv)
    if args.prepared_inputs and model != "biolord":
        parser.error("--prepared_inputs is supported for BioLORD, not binary scGen.")
    if args.data_dir is None:
        if args.prepared_inputs:
            args.data_dir = args.prepared_inputs
        else:
            parser.error("--data_dir is required unless --prepared_inputs is supplied.")
    if args.prepared_inputs and args.stage in {"prepare", "calibration-prepare"}:
        from models.prepared_inputs import EXPERIMENTS, stage
        calibration = args.stage == "calibration-prepare"
        output = args.experiment_dir or args.out_dir / EXPERIMENTS["calibration" if calibration else "biolord"]
        if not output.resolve().is_relative_to(args.out_dir.resolve()):
            parser.error("--experiment_dir must remain inside --out_dir")
        if not args.execute:
            import json
            print(json.dumps({"execute": False, "prepared_inputs": str(args.prepared_inputs),
                              "output": str(output), "resampling": False, "training": False}, indent=2))
            return 0
        try:
            print(stage(args.prepared_inputs, output, args.reference_dir, calibration))
            return 0
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            parser.exit(2, f"ERROR: {exc}\n")
    if args.stage.endswith("train") and model == "biolord" and args.task is None:
        parser.error("--task is required for training")
    if args.stage == "train" and model == "scgen" and args.shard is None:
        parser.error("--shard is required for scGen training")
    if args.task is not None:
        maximum = 40 if args.stage.startswith("calibration") else 20
        if not 0 <= args.task < maximum:
            parser.error(f"--task must be in 0..{maximum-1}")
    if model == "scgen" and (not 1 <= args.workers <= 128 or
                             (args.shard is not None and not 0 <= args.shard < args.workers)):
        parser.error("Use 1..128 workers and a shard in 0..workers-1")
    dispatch_model = model
    native = []
    if model == "biolord":
        if args.stage.startswith("calibration-"):
            dispatch_model = "figure4c"
            native = [args.stage.removeprefix("calibration-")]
        else:
            native = [args.stage]
        if args.task is not None:
            native += ["--task", str(args.task)]
    elif args.stage == "prepare":
        dispatch_model = "scgen_prepare"
    else:
        native = ["--shard", str(args.shard), "--workers", str(args.workers)]
    if args.experiment_dir:
        native += ["--root", str(args.experiment_dir)]
    if args.stage in ("prepare", "train", "calibration-train"):
        native += ["--training-log-dir", str(args.training_log_dir or args.out_dir / "logs")]
    common = ["--model", dispatch_model, "--data-root", str(args.data_dir),
              "--reference-root", str(args.reference_dir), "--results-root", str(args.out_dir)]
    if args.go_file:
        common += ["--go-file", str(args.go_file)]
    if args.execute:
        common.append("--execute")
    path = Path(__file__).resolve().parent / "_context_generalization/run.py"
    spec = importlib.util.spec_from_file_location("_transfer_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = {name: os.environ.get(name) for name in
                ("PD_SAVE_SCGEN_CHECKPOINTS", "PD_SCGEN_CHECKPOINT_TASKS")}
    if model == "scgen":
        if args.checkpoint_tasks and not args.save_checkpoints:
            parser.error("--checkpoint_tasks requires --save_checkpoints")
        if args.checkpoint_tasks:
            try:
                selected = [int(value) for value in args.checkpoint_tasks.split(",")]
            except ValueError:
                parser.error("--checkpoint_tasks must contain integer task IDs")
            if any(not 0 <= task < 5486 for task in selected):
                parser.error("scGen task IDs must be in 0..5485")
        os.environ["PD_SAVE_SCGEN_CHECKPOINTS"] = "1" if args.save_checkpoints else "0"
        os.environ["PD_SCGEN_CHECKPOINT_TASKS"] = args.checkpoint_tasks or ""
    try:
        return module.main([*common, "--", *native])
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def predictions(model, argv=None):
    from reproduction.io import check_output_location, verify
    from reproduction.predictions import Predictions
    parser = argparse.ArgumentParser(description=f"Inspect or export saved {model} transfer predictions")
    parser.add_argument("--results_dir", type=Path, default=Path("results/predictions"),
                        help="Installed predictions bundle (default: results/predictions)")
    parser.add_argument("--target", required=True, choices=("k562", "rpe1", "hepg2", "jurkat"))
    parser.add_argument("--repeat", "--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--out_dir", type=Path, help="Copy the selected prediction archive to a new directory")
    args = parser.parse_args(argv)
    try:
        verify(args.results_dir)
        path = args.results_dir / "predictions" / model / args.target / f"repeat{args.repeat}"
        if not (path / "predictions.npz").is_file():
            parser.error("Predictions are not in this bundle. Install the predictions bundle "
                         "or pass its location with --results_dir; see README.md.")
        data = Predictions.load(path)
        if data.metadata["model"] != model or data.metadata["target"] != args.target or data.metadata["repeat"] != args.repeat:
            parser.error("Prediction metadata does not match the requested experiment")
        import json
        print(json.dumps({"model": model, "target": args.target, "repeat": args.repeat,
                          "n_perturbations": len(data.perturbations),
                          "n_genes": len(data.features), "protocol": data.metadata["protocol"]}, indent=2))
        if args.out_dir:
            check_output_location(args.results_dir, args.out_dir)
            data.save(args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
