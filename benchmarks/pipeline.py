"""Replay context checkpoints, assemble components, and render their analysis."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from reproduction.io import new_directory, read_json, sha256, write_json


def run(spec_path, output):
    from models.predict import predict_checkpoint
    from decomposition.run import run as decompose
    from figures.components import render
    spec_path = Path(spec_path).resolve()
    spec = read_json(spec_path)
    contexts = spec["contexts"]
    if len(contexts) < 2 or len(set(contexts)) != len(contexts):
        raise ValueError("The pipeline requires at least two distinct contexts")
    output = Path(output).resolve()
    entries = {}
    families = {spec["models"][context]["model"] for context in contexts}
    if len(families) != 1:
        raise ValueError("A component panel must not mix model families across contexts")
    for context in contexts:
        entry = spec["models"][context]
        checkpoint = (spec_path.parent / entry["checkpoint"]).resolve()
        inputs = (spec_path.parent / entry["inputs"]).resolve()
        if not checkpoint.is_dir() or not inputs.is_file():
            raise FileNotFoundError(f"Missing checkpoint or prepared inputs for {context}")
        if output.is_relative_to(checkpoint) or output.is_relative_to(inputs.parent):
            raise ValueError("Pipeline output must be outside all input/checkpoint directories")
        entries[context] = (entry, checkpoint, inputs)
    output = new_directory(output)
    predictions = {}
    for context, (entry, checkpoint, inputs) in entries.items():
        if entry["model"] not in ("ridge", "mlp"):
            raise ValueError("This pipeline replays portable Ridge/MLP checkpoints; native inputs use their model command")
        target = output / "predictions" / context
        predict_checkpoint(SimpleNamespace(model=entry["model"], checkpoint=checkpoint,
            inputs=inputs, out_dir=target, seed=entry.get("seed", 0), device="cpu", target=context,
            expected=(spec_path.parent / entry["expected"]).resolve() if entry.get("expected") else None,
            atol=spec.get("atol", 2e-5), rtol=spec.get("rtol", 2e-5)))
        predictions[context] = str(target / "predictions.npz")
    manifest = output / "panel_manifest.json"
    write_json(manifest, {"contexts": contexts, "predictions": predictions,
                         **({"perturbations": spec["perturbations"]} if "perturbations" in spec else {})})
    decompose(output / "components", manifest=manifest)
    render(output / "components", output / "figures")
    write_json(output / "provenance.json", {"spec_sha256": sha256(spec_path),
        "training": False, "checkpoint_replay": True,
        "assembly": "context-specific checkpoints; no cross-repeat pooling"})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.execute:
            print(f"Replay pipeline: {args.spec}\nOutput: {args.out_dir}\nUse --execute to run.")
            return
        from benchmarks._runtime import require_compute
        require_compute("Checkpoint/decomposition pipeline")
        print(run(args.spec, args.out_dir))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
