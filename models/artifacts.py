"""Discover and package experiment-specific checkpoints and saved predictions."""

import argparse
from pathlib import Path
import shutil

from reproduction.io import new_directory, read_json, record_file, safe_path, sha256, write_json


CELLS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
         "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}
PAIRS = {"k562": "rpe1", "rpe1": "k562", "hepg2": "jurkat", "jurkat": "hepg2"}
CONTEXT = "benchmarks/context_generalization"


def asset(root, path, role, destination=None):
    return {"root": root, "path": path, "role": role,
            "destination": destination or f"{role}/{Path(path).name}"}


def expected_records():
    records = []
    for cell, dataset in CELLS.items():
        for fold in range(5):
            for model in ("ridge", "mlp"):
                records.append({"id": f"{model}/within/{cell}/fold{fold}", "model": model,
                    "experiment": "within_cl", "target": cell, "repeat": fold,
                    "architecture": model, "checkpoint_status": "not_saved_in_original_run",
                    "assets": [asset("benchmark", f"runs/{'Ridge' if model == 'ridge' else 'MLP'}_5foldCV/{dataset}/fold{fold}/predictions.npz", "saved")]})
            records.append({"id": f"morph/within/{cell}/fold{fold}", "model": "morph",
                "experiment": "within_cl", "target": cell, "repeat": fold,
                "architecture": "benchmark_mean_conditioned",
                "assets": [asset("benchmark", f"runs/MORPH_5foldCV/{dataset}/fold{fold}/model.pt", "weights"),
                           asset("benchmark", f"runs/MORPH_5foldCV/{dataset}/fold{fold}/predictions.npz", "saved")]})
        for seed in range(5):
            pair = f"{PAIRS[cell]}_to_{cell}"
            base = f"morph_crosscl/runs/{pair}/seed{seed}"
            for setting, name in (("source_only", "zeroshot"), ("controls", "finetuned"), ("target30", "ft30pct")):
                assets = [asset("benchmark", f"{base}/predictions_{name}.npz", "saved")]
                if setting == "source_only":
                    assets += [asset("benchmark", f"{base}/source_model/best_model.pt", "weights"),
                               asset("benchmark", f"{base}/source_model/config.json", "config")]
                records.append({"id": f"morph/transfer/{cell}/seed{seed}/{setting}", "model": "morph",
                    "target": cell, "repeat": seed, "setting": setting, "experiment": "cross_cl",
                    "architecture": "upstream_pickled_model",
                    "checkpoint_status": "located" if setting == "source_only" else "adapted_weights_not_saved",
                    "assets": assets})
            scgpt = f"runs/scGPT/{dataset}/seed{seed}"
            records.append({"id": f"scgpt/within/{cell}/seed{seed}", "model": "scgpt",
                "experiment": "within_cl", "target": cell, "repeat": seed,
                "architecture": "fine_tuned_transformer_expr_decoder",
                "assets": [asset("benchmark", f"{scgpt}/model_{dataset}_seed{seed}/best_model.pt", "weights"),
                           asset("benchmark", f"{scgpt}/predictions.npz", "saved"),
                           asset("agop", "data/raw/scgpt_pretrained/args.json", "config"),
                           asset("agop", "data/raw/scgpt_pretrained/vocab.json", "vocabulary")]})
            biolord = f"{CONTEXT}/results/biolord/three_to_one_cellline_go_author_v1_exportfixed_v2"
            records.append({"id": f"biolord/transfer/{cell}/seed{seed}", "model": "biolord",
                "experiment": "cross_cl", "target": cell, "repeat": seed,
                "architecture": "go_cellline_condition_means",
                "assets": [asset("agop", f"checkpoints/biolord_cellline_go_author_v1/seed{seed}/{cell}/model.pt", "weights"),
                           asset("agop", f"{biolord}/seed{seed}/{cell}/predictions.npz", "saved"),
                           asset("agop", f"{biolord}/seed{seed}/{cell}/metrics.json", "provenance"),
                           asset("agop", f"{CONTEXT}/results/biolord/three_to_one_cellline_go_author_v1/inputs/{cell}/training.h5ad", "training_data"),
                           asset("agop", f"{CONTEXT}/results/biolord/three_to_one_cellline_go_author_v1/inputs/go_features.npz", "go_features")]})
            for setting in ("source_only", "target30"):
                root = f"{CONTEXT}/results/figure4c_author_aligned_v2_retry1"
                records.append({"id": f"biolord/calibration/{cell}/seed{seed}/{setting}",
                    "model": "biolord", "experiment": "target_calibration", "target": cell,
                    "repeat": seed, "setting": setting, "architecture": "go_cellline_condition_means",
                    "assets": [asset("agop", f"checkpoints/biolord_fig4c_author_aligned_v2/{cell}/seed{seed}/{setting}/model.pt", "weights"),
                               asset("agop", f"{root}/biolord/{cell}/seed{seed}/{setting}/predictions.npz", "saved"),
                               asset("agop", f"{root}/biolord/{cell}/seed{seed}/{setting}/metrics.json", "provenance"),
                               asset("agop", f"{root}/inputs/{cell}/seed{seed}/{setting}.h5ad", "training_data"),
                               asset("agop", f"{root}/inputs/go_features.npz", "go_features")]})
            records.append({"id": f"scgen/calibration_cohort/{cell}/seed{seed}", "model": "scgen",
                "experiment": "rescored_binary_baseline", "target": cell, "repeat": seed,
                "architecture": "binary_per_perturbation", "checkpoint_status": "not_saved_in_original_run",
                "assets": [asset("agop", f"{CONTEXT}/results/figure4c_author_aligned_v2_retry1/scgen_rescored/{cell}/seed{seed}/predictions.npz", "saved")]})
            state_base = f"runs/STATE_crossCL/{cell}/seed{seed}"
            state_run = f"{state_base}/run/crossCL_{cell}_seed{seed}"
            records.append({"id": f"state/calibration/{cell}/seed{seed}", "model": "state",
                "experiment": "target_calibration", "target": cell, "repeat": seed,
                "architecture": "state_transition", "pairing": "checkpoint_replay_not_yet_verified",
                "assets": [asset("benchmark", f"{state_run}/checkpoints/final.ckpt", "weights"),
                           asset("benchmark", f"{state_run}/config.yaml", "config"),
                           asset("benchmark", f"{state_run}/pert_onehot_map.pt", "perturbation_map"),
                           asset("benchmark", f"{state_base}/predictions.npz", "saved")]})
            for direction in ("1to1", "3to1"):
                for setting in ("source_only", "target30"):
                    folder = f"{direction}/{cell}/seed{seed}/{setting}"
                    for model in ("ridge", "mlp"):
                        checkpoint = "ridge_pca.npz" if model == "ridge" else "mlp.pt"
                        assets = [asset("agop", f"checkpoints/component_alignment_all_models_v1_retry1/{folder}/{checkpoint}", "weights"),
                                  asset("agop", f"{CONTEXT}/results/component_alignment_all_models_v1_retry1/baselines/{folder}/{model}_predictions.npz", "saved"),
                                  asset("agop", f"{CONTEXT}/results/component_alignment_all_models_v1_retry1/baselines/{folder}/metadata.json", "provenance")]
                        if model == "mlp":
                            assets.append(asset("agop", f"checkpoints/component_alignment_all_models_v1_retry1/{folder}/ridge_pca.npz",
                                                "preprocessing", "weights/ridge_pca.npz"))
                        records.append({"id": f"{model}/matched/{folder}", "model": model,
                            "experiment": "matched_transfer", "target": cell, "repeat": seed,
                            "setting": setting, "direction": direction,
                            "architecture": "matched_ridge_pca_mlp", "assets": assets})
        state_base = f"runs/STATE_zeroshot/{cell}/seed0"
        state_run = f"{state_base}/run/zeroshot_{cell}_seed0"
        records.append({"id": f"state/source_only/{cell}/seed0", "model": "state",
            "experiment": "source_only_transfer", "target": cell, "repeat": 0,
            "architecture": "state_transition", "pairing": "checkpoint_replay_not_yet_verified",
            "assets": [asset("benchmark", f"{state_run}/checkpoints/final.ckpt", "weights"),
                       asset("benchmark", f"{state_run}/config.yaml", "config"),
                       asset("benchmark", f"{state_run}/pert_onehot_map.pt", "perturbation_map"),
                       asset("benchmark", f"{state_base}/predictions.npz", "saved")]})
    return records


def discover(roots):
    roots = {key: Path(value).resolve() for key, value in roots.items()}
    records = expected_records()
    for record in records:
        for entry in record["assets"]:
            path = safe_path(roots[entry["root"]], entry["path"])
            entry["available"] = path.is_file()
            entry["size"] = path.stat().st_size if path.is_file() else None
        weights = [entry for entry in record["assets"] if entry["role"] == "weights"]
        record["weights_available"] = bool(weights) and all(e["available"] for e in weights)
        record["saved_predictions_available"] = any(
            e["available"] for e in record["assets"] if e["role"] == "saved")
        record["companion_assets_missing"] = [e["role"] for e in record["assets"]
                                               if not e["available"]]
        record["runtime_validation"] = "not_established_by_file_discovery"
        for entry in record["assets"]:
            if entry["role"] == "provenance" and entry["available"]:
                meta = read_json(safe_path(roots[entry["root"]], entry["path"]))
                hashes = meta.get("checkpoint_sha256", {})
                if isinstance(hashes, str):
                    hashes = {"model.pt": hashes}
                recovered = meta.get("prediction_recovery", {}).get("checkpoint_sha256")
                if recovered:
                    hashes["model.pt"] = recovered
                for weight in record["assets"]:
                    if weight["role"] in ("weights", "preprocessing") and Path(weight["path"]).name in hashes:
                        weight["expected_sha256"] = hashes[Path(weight["path"]).name]
    return {"schema_version": 1, "records": records,
            "description": "File-level availability, not proof of replay equivalence."}


def package(registry, identifier, roots, destination, include="all"):
    from benchmarks._runtime import require_compute
    roots = {key: Path(value).resolve() for key, value in roots.items()}
    destination = Path(destination).resolve()
    if any(destination.is_relative_to(path) for path in roots.values()):
        raise ValueError("The destination must be outside original source roots")
    matches = [r for r in registry["records"] if r["id"] == identifier]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate artifact ID")
    record = matches[0]
    roles = ({"saved", "provenance"} if include == "saved" else
             {"weights", "preprocessing", "config", "vocabulary", "perturbation_map", "provenance"})
    if include == "all":
        roles = {entry["role"] for entry in record["assets"]}
    selected = [e for e in record["assets"] if e["role"] in roles and e["available"]]
    if include in ("all", "weights") and not record["weights_available"]:
        raise ValueError("This experiment has no saved checkpoint; use --include saved")
    if not selected:
        raise ValueError("No assets are available for this selection")
    require_compute("Checkpoint/prediction packaging")
    output = new_directory(destination)
    provenance = []
    for entry in selected:
        source = safe_path(roots[entry["root"]], entry["path"])
        digest = sha256(source)
        if entry.get("expected_sha256") and digest != entry["expected_sha256"]:
            raise ValueError("Checkpoint differs from the saved prediction's recorded weight hash")
        target = safe_path(output, entry["destination"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Artifact destination collision")
        shutil.copyfile(source, target)
        if sha256(source) != digest or sha256(target) != digest:
            raise ValueError("Artifact changed during packaging")
        provenance.append({**entry, "sha256": digest})
    write_json(output / "artifact.json", {**record, "packaged_assets": provenance,
        "redistribution": "Review upstream model and dataset terms before public distribution.",
        "training_inputs": "Check packaged roles; raw controls/descriptors may require a separate input bundle."})
    write_json(output / "provenance.json", {
        "sources_before": {k: {"path": str(v)} for k, v in roots.items()}})
    write_json(output / "manifest.json", {"schema_version": 1, "version": "models-v1",
        "profile": "model", "files": [record_file(output, p) for p in sorted(output.rglob("*")) if p.is_file()]})
    return output


def bundle_run(model, checkpoint, output, predictions=None, companions=None):
    """Package a new portable/native run without changing its checkpoint layout."""
    from benchmarks._runtime import require_compute
    checkpoint = Path(checkpoint).resolve()
    sources = [checkpoint, *([Path(predictions).resolve()] if predictions else []),
               *(Path(p).resolve() for p in (companions or {}).values())]
    output = Path(output).resolve()
    protected = [p if p.is_dir() else p.parent for p in sources]
    if any(output.is_relative_to(p) for p in protected):
        raise ValueError("Bundle output must be outside source checkpoint/prediction directories")
    if checkpoint.is_dir():
        if not any((checkpoint / name).is_file() for name in ("model.json", "model.pt", "best_model.pt")):
            raise ValueError("Select a checkpoint directory, not a complete project or run tree")
        files = [p for p in checkpoint.rglob("*") if p.is_file()]
        if any(p.is_symlink() or not p.resolve().is_relative_to(checkpoint) for p in files):
            raise ValueError("Checkpoint directories must not contain external file links")
    else:
        files = [checkpoint]
    require_compute("Model release packaging")
    output = new_directory(output)
    for path in files:
        relative = path.relative_to(checkpoint) if checkpoint.is_dir() else Path(path.name)
        target = output / "checkpoint" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(path)
        shutil.copyfile(path, target)
        if before != sha256(path) or before != sha256(target):
            raise ValueError("Checkpoint changed during export")
    for role, path in {**({"saved": predictions} if predictions else {}), **(companions or {})}.items():
        if role not in ("saved", "inputs", "vocab", "model_config", "perturbation_map", "training_data", "go_features"):
            raise ValueError(f"Unsupported companion role: {role}")
        path = Path(path)
        target = output / "companions" / role / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(path)
        shutil.copyfile(path, target)
        if digest != sha256(path) or digest != sha256(target):
            raise ValueError("Companion input changed during export")
    write_json(output / "artifact.json", {"model": model, "checkpoint_directory": "checkpoint",
        "has_saved_predictions": predictions is not None, "runtime_validation": "retain the run's validation record",
        "redistribution": "Review model and dataset terms before publishing."})
    write_json(output / "provenance.json", {
        "sources_before": {str(i): {"path": str(p)} for i, p in enumerate(protected)}})
    write_json(output / "manifest.json", {"schema_version": 1, "version": "models-v1", "profile": "model",
        "files": [record_file(output, p) for p in sorted(output.rglob("*")) if p.is_file()]})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("discover", "list", "package", "bundle"))
    parser.add_argument("--roots", type=Path, help="JSON mapping agop/benchmark/release to local read-only roots")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--model", choices=("ridge", "mlp", "morph", "state", "scgpt", "biolord", "scgen"))
    parser.add_argument("--id")
    parser.add_argument("--include", choices=("all", "weights", "saved"), default="all")
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--companion", action="append", default=[], help="ROLE=/path/to/file")
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            if not args.roots or not args.out_dir:
                parser.error("discover requires --roots and --out_dir")
            roots = read_json(args.roots)
            if any(args.out_dir.resolve().is_relative_to(Path(p).resolve()) for p in roots.values()):
                raise ValueError("The registry output must be outside original source roots")
            output = new_directory(args.out_dir)
            write_json(output / "registry.json", discover(roots))
            print(output / "registry.json")
        elif args.command == "list":
            records = read_json(args.registry)["records"] if args.registry else expected_records()
            print(f"{'Experiment':65} {'Weights':9} Saved predictions")
            for record in records:
                if not args.model or record["model"] == args.model:
                    print(f"{record['id']:65} {str(record.get('weights_available', 'check')):9} "
                          f"{record.get('saved_predictions_available', 'check')}")
        elif args.command == "package":
            if not all((args.registry, args.roots, args.id, args.out_dir)):
                parser.error("package requires --registry, --roots, --id, and --out_dir")
            print(package(read_json(args.registry), args.id, read_json(args.roots), args.out_dir, args.include))
        else:
            if not all((args.model, args.checkpoint, args.out_dir)):
                parser.error("bundle requires --model, --checkpoint, and --out_dir")
            companions = {}
            for value in args.companion:
                role, separator, path = value.partition("=")
                if not separator or role in companions:
                    parser.error("--companion must contain unique ROLE=PATH entries")
                companions[role] = path
            print(bundle_run(args.model, args.checkpoint, args.out_dir, args.predictions, companions))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
