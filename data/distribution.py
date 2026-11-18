"""Plan or export self-contained research artifacts without fitting or scoring."""

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from data.artifacts import prepare_processed, required_files
from models.artifacts import CELLS, CONTEXT, discover
from models.inference_io import prediction_labels
from reproduction.catalog import BIOLORD, MATCHING, SCGEN
from reproduction.download import pack
from reproduction.io import new_directory, read_json, record_file, safe_path, sha256, verify, write_json


ALIGNMENT = f"{CONTEXT}/results/component_alignment_all_models_v1_retry1"


def _source(roots, root, relative):
    if root not in roots:
        raise ValueError(f"Missing source alias: {root}")
    return safe_path(roots[root], relative)


def plan(roots):
    roots = {name: Path(value).resolve() for name, value in roots.items()}
    if set(roots) != {"agop", "benchmark", "release", "figures"}:
        raise ValueError("Roots must name agop, benchmark, release, and figures.")
    registry = discover(roots)
    assets = {}
    for record in registry["records"]:
        for asset in record["assets"]:
            if asset["available"]:
                assets[(asset["root"], asset["path"])] = asset["size"]
    processed = required_files("all") + ["hallmark.gmt"]
    missing = [str(_source(roots, "release", "data/processed/" + name))
               for name in processed
               if not _source(roots, "release", "data/processed/" + name).is_file()]
    missing.extend(f"{record['id']}: {asset['role']} {asset['path']}"
                   for record in registry["records"] for asset in record["assets"]
                   if not asset["available"])
    required = [
        ("agop", f"{ALIGNMENT}/inputs/embeddings.npz"),
        ("agop", f"{ALIGNMENT}/inputs/anova.npz"),
        ("agop", f"{MATCHING}/assembly_manifest.json"),
        ("agop", f"{CONTEXT}/manifests/scgen_3to1.jsonl"),
    ]
    for alias, relative in required:
        if not _source(roots, alias, relative).is_file():
            missing.append(f"{alias}/{relative}")
    controls = []
    for target, dataset in CELLS.items():
        for model, relative in (
                ("genetic", f"data/{target}/replogle.h5ad"),
                ("scgpt", f"data/gears/{dataset}/perturb_processed.h5ad")):
            path = _source(roots, "benchmark", relative)
            controls.append({"model": model, "target": target, "root": "benchmark",
                             "path": relative, "available": path.is_file(),
                             "source_size": path.stat().st_size if path.is_file() else None})
            if not path.is_file():
                missing.append(f"benchmark/{relative}")
    return {
        "schema_version": 1, "version": "distribution-v1",
        "roots": {name: str(path) for name, path in roots.items()},
        "registry": registry, "control_sources": controls,
        "bundles": ["figures", "processed", "analysis", "predictions", "models"],
        "catalog_unique_assets": len(assets), "catalog_source_bytes": sum(assets.values()),
        "processed_source_bytes": sum(
            _source(roots, "release", "data/processed/" + name).stat().st_size
            for name in processed if _source(roots, "release", "data/processed/" + name).is_file()),
        "missing_required_inputs": missing,
        "no_training": True, "no_score_calculation": True,
        "execution": "SLURM CPU allocation; the STATE uv environment supplies Torch/AnnData for export.",
    }


def _copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    before = sha256(source)
    if expected and before != expected:
        raise ValueError(f"Source differs from its recorded hash: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != before:
            raise ValueError(f"Conflicting bundle destination: {destination}")
        return before
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if sha256(source) != before or sha256(destination) != before:
        raise ValueError(f"Source changed during export: {source.name}")
    return before


def _manifest(root, profile, roots):
    write_json(root / "provenance.json", {
        "sources_before": {name: {"path": str(path)} for name, path in roots.items()},
        "operation": "artifact_export_only", "models_fitted": False, "scores_calculated": False})
    write_json(root / "manifest.json", {
        "schema_version": 1, "version": f"{profile}-v1", "profile": profile,
        "files": [record_file(root, path) for path in sorted(root.rglob("*")) if path.is_file()]})
    verify(root)


def read_legacy_prediction(path, trust_legacy_arrays):
    """Read explicitly trusted local archives; exported identifiers are Unicode."""
    if not trust_legacy_arrays:
        raise ValueError("Historical archives require --trust_legacy_arrays during export.")
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, values in arrays.items():
        if values.dtype.hasobject:
            if name not in {"test_perts", "feature_names", "gene_names", "perturbations"}:
                raise ValueError(f"Unexpected object-encoded field {name}: {path}")
            if values.ndim != 1 or not all(isinstance(value, str) for value in values):
                raise ValueError(f"Legacy field is not a string axis: {name}")
            arrays[name] = values.astype(str)
    if "predictions" not in arrays:
        raise ValueError(f"Historical prediction schema is unsupported: {path}")
    arrays["test_perts"] = prediction_labels(arrays)
    labels = arrays["test_perts"]
    if labels.ndim != 1 or len(set(labels)) != len(labels):
        raise ValueError("Prediction perturbations must be unique.")
    if arrays["predictions"].ndim != 2 or len(arrays["predictions"]) != len(labels):
        raise ValueError("Prediction axes do not match.")
    return arrays


def _gene_axes(roots):
    import h5py
    import pickle
    axes = {}
    for target, dataset in CELLS.items():
        with _source(roots, "release", f"data/processed/genetic/{target}.pkl").open("rb") as stream:
            axes["genetic", target] = np.asarray(pickle.load(stream)["hvg_names"], dtype=str)
        with h5py.File(_source(roots, "benchmark",
                       f"data/gears/{dataset}/perturb_processed.h5ad"), "r") as source:
            node = source["var/gene_name"]
            if isinstance(node, h5py.Dataset):
                names = node.asstr()[...]
            else:
                codes = node["codes"][...]
                if (codes < 0).any():
                    raise ValueError("The recorded scGPT gene axis has missing names.")
                names = node["categories"].asstr()[...][codes]
            axes["scgpt", target] = np.asarray(names, dtype=str)
    for values in axes.values():
        if values.ndim != 1 or len(set(values)) != len(values):
            raise ValueError("Recorded gene axes must be unique one-dimensional identifiers.")
    return axes


def _cached(registry, roots, output, trust, axes=None):
    records = []
    for original in registry["records"]:
        record = copy.deepcopy(original)
        cached = next(asset for asset in record["assets"] if asset["role"] == "saved")
        source = _source(roots, cached["root"], cached["path"])
        before = sha256(source)
        arrays = read_legacy_prediction(source, trust)
        attached = False
        if "feature_names" not in arrays and "gene_names" in arrays:
            arrays["feature_names"] = arrays["gene_names"].astype(str)
        if axes is not None and "feature_names" not in arrays:
            kind = "scgpt" if record["model"] == "scgpt" else "genetic"
            expected = axes[kind, record["target"]]
            if len(expected) != arrays["predictions"].shape[1]:
                raise ValueError(f"{record['id']}: saved columns differ from their recorded gene axis.")
            arrays["feature_names"] = expected.copy()
            attached = True
        if "feature_names" in arrays:
            names = arrays["feature_names"].astype(str)
            if names.shape != (arrays["predictions"].shape[1],) or len(set(names)) != len(names):
                raise ValueError("Saved gene labels must be complete, unique and ordered.")
        relative = Path("saved") / record["id"] / "predictions.npz"
        destination = safe_path(output, relative)
        destination.parent.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(destination, **arrays)
        if sha256(source) != before:
            raise ValueError("Historical predictions changed during export.")
        record["saved"] = str(relative)
        record["saved_source_sha256"] = before
        record["normalization"] = "Object string axes converted to Unicode; numeric arrays unchanged."
        record["verified_gene_axis_attached"] = attached
        write_json(destination.parent / "metadata.json", {
            "model": record.get("model"), "target": record.get("target"),
            "repeat": record.get("repeat"), "experiment": record.get("experiment"),
            "setting": record.get("setting"), "source_sha256": before,
            "numeric_arrays_changed": False, "verified_gene_axis_attached": attached,
        })
        records.append(record)
    return records


def _transfer_predictions(roots, output):
    from models.biolord import from_recovered_archive
    from models.scgen import from_binary_archives
    from reproduction.release import _metadata
    tasks_path = _source(roots, "agop", f"{CONTEXT}/manifests/scgen_3to1.jsonl")
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    for repeat in range(5):
        for target in CELLS:
            source = _source(roots, "agop", f"{BIOLORD}/seed{repeat}/{target}/predictions.npz")
            data = from_recovered_archive(source, _metadata(
                "biolord", target, repeat, "biolord_norman_go_pseudobulk_cellline_categorical_v1"))
            data.save(output / "predictions/biolord" / target / f"repeat{repeat}")
            paths = [_source(roots, "agop",
                     f"{SCGEN}/seed{repeat}/{target}/{task['perturbation']}/predictions.npz")
                     for task in tasks if task["target"] == target]
            data = from_binary_archives(paths, _metadata(
                "scgen", target, repeat, "scgen_binary_target_controls_real_cell_labels_native_all_v1"))
            data.save(output / "predictions/scgen" / target / f"repeat{repeat}")


def _model_export(registry, roots, output, trust, axes):
    from models.release_inputs import prepare_model_inputs
    records = _cached(registry, roots, output, trust, axes)
    copied = {}
    for record in records:
        record["packaged_assets"] = []
        for asset in record["assets"]:
            if asset["role"] == "saved" or not asset["available"]:
                continue
            key = (asset["root"], asset["path"])
            relative = str(Path("objects") / asset["root"] / asset["path"])
            if key not in copied:
                source = _source(roots, *key)
                if source.stat().st_size != asset["size"]:
                    raise ValueError(f"Source size changed after planning: {asset['path']}")
                copied[key] = _copy(source, safe_path(output, relative), asset.get("expected_sha256"))
            record["packaged_assets"].append({
                "role": asset["role"], "path": relative, "sha256": copied[key]})
        record["inference"] = None
    prepare_model_inputs(records, roots, output)
    write_json(output / "registry.json", {"schema_version": 1, "records": records})


def execute(specification, destination, trust_legacy_arrays=False):
    from benchmarks._runtime import require_compute
    require_compute("Distribution packaging and real-control export")
    roots = {name: Path(value).resolve() for name, value in specification["roots"].items()}
    destination = Path(destination).resolve()
    if any(destination.is_relative_to(path) for path in roots.values()):
        raise ValueError("Distribution output must be outside every read-only source root.")
    if specification["missing_required_inputs"]:
        raise ValueError("Required distribution inputs are missing; regenerate the plan after supplying them.")
    if not trust_legacy_arrays:
        raise ValueError("Use --trust_legacy_arrays only for the declared trusted source projects.")
    output = new_directory(destination)
    bundles = {}
    figures = output / "figures"
    figure_manifest = verify(roots["figures"])
    figures.mkdir()
    for relative in ["manifest.json", *(item["path"] for item in figure_manifest["files"])]:
        _copy(safe_path(roots["figures"], relative), safe_path(figures, relative))
    bundles["figures"] = figures
    processed = prepare_processed(roots["release"] / "data/processed", output / "processed", "all")
    bundles["processed"] = processed
    analysis = new_directory(output / "analysis")
    for relative in ["manifest.json", *(item["path"] for item in figure_manifest["files"])]:
        if relative not in {"manifest.json", "provenance.json"}:
            _copy(safe_path(roots["figures"], relative), safe_path(analysis, relative))
    from reproduction.release import export_genetic, export_matching
    export_matching(roots, analysis, {})
    export_genetic(roots, analysis)
    _manifest(analysis, "analysis", roots)
    bundles["analysis"] = analysis
    predictions = new_directory(output / "predictions")
    axes = _gene_axes(roots)
    records = _cached(specification["registry"], roots, predictions, trust_legacy_arrays, axes)
    _transfer_predictions(roots, predictions)
    write_json(predictions / "registry.json", {"schema_version": 1, "records": records})
    _manifest(predictions, "predictions", roots)
    bundles["predictions"] = predictions
    models = new_directory(output / "models")
    _model_export(specification["registry"], roots, models, trust_legacy_arrays, axes)
    _manifest(models, "models", roots)
    bundles["models"] = models
    indices = {}
    for name, directory in bundles.items():
        index = pack(directory, output / f"{name}-v1.tar.gz")
        indices[name] = {"index": index.name, **read_json(index)}
    write_json(output / "distribution.json", {
        "schema_version": 1, "version": specification["version"], "bundles": indices,
        "training_performed": False, "scores_recalculated": False,
        "usage": "Install each archive with its checksum index; model recipes use relative bundle paths."})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, help="Private JSON with agop/benchmark/release/figures roots")
    parser.add_argument("--plan", type=Path, help="Previously reviewed export plan")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--trust_legacy_arrays", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            if not args.plan:
                parser.error("--execute requires a reviewed --plan")
            print(execute(read_json(args.plan), args.out_dir, args.trust_legacy_arrays))
        else:
            if not args.roots:
                parser.error("Planning requires --roots")
            specification = plan(read_json(args.roots))
            if any(args.out_dir.resolve().is_relative_to(Path(path).resolve())
                   for path in specification["roots"].values()):
                raise ValueError("Plan output must be outside all declared read-only source roots.")
            output = new_directory(args.out_dir)
            write_json(output / "plan.json", specification)
            print(json.dumps({key: value for key, value in specification.items()
                              if key not in {"roots", "registry", "control_sources"}}, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
