"""Assemble labeled response panels and run the existing component analysis."""

import argparse
from pathlib import Path

import numpy as np

from models.inference_io import prediction_labels
from reproduction.decomposition import compare, decompose, second_moment_fractions, validate_decomposition
from reproduction.io import new_directory, read_json, sha256, write_json


def _axis(values, size, name):
    values = np.asarray(values, dtype=str)
    if values.shape != (size,) or len(set(values)) != size:
        raise ValueError(f"{name} must contain unique ordered identifiers")
    return values


def load_panel(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    tensor = arrays.get("true_deltas", arrays.get("predictions"))
    if tensor is None or tensor.ndim != 3:
        raise ValueError("A panel needs context × perturbation × gene responses")
    for key, size in zip(("contexts", "perturbations", "feature_names"), tensor.shape):
        arrays[key] = _axis(arrays[key], size, key)
    for key in ("true_deltas", "predictions"):
        if key in arrays and (arrays[key].shape != tensor.shape or not np.isfinite(arrays[key]).all()):
            raise ValueError("All response tensors must have matching finite complete axes")
    return arrays


def assemble(manifest_path):
    manifest_path = Path(manifest_path)
    spec = read_json(manifest_path)
    contexts = list(spec["contexts"])
    if len(contexts) < 2 or len(set(contexts)) != len(contexts):
        raise ValueError("At least two unique contexts are required")
    archives, genes, sources = {}, None, []
    used_paths = set()
    for context in contexts:
        relative = spec["predictions"][context]
        path = Path(relative) if Path(relative).is_absolute() else manifest_path.parent / relative
        if path.resolve() in used_paths:
            raise ValueError("Do not duplicate a target-specific archive across contexts")
        used_paths.add(path.resolve())
        metadata_path = path.parent / "metadata.json"
        if metadata_path.is_file():
            target = read_json(metadata_path).get("target")
            if target is not None and target != context:
                raise ValueError("Prediction metadata does not match its assigned context")
        with np.load(path, allow_pickle=False) as data:
            if "feature_names" not in data:
                raise ValueError("Prediction files require verified gene labels; positional guesses are not accepted")
            names = data["feature_names"].astype(str)
            labels = prediction_labels(data)
            if len(set(names)) != len(names) or len(set(labels)) != len(labels):
                raise ValueError("Duplicate gene or perturbation labels")
            if genes is not None and not np.array_equal(genes, names):
                raise ValueError("Context gene identities/order differ")
            genes = names
            keys = ["predictions"] + (["true_deltas"] if "true_deltas" in data else [])
            arrays = {key: data[key].copy() for key in keys}
            if any(value.shape != (len(labels), len(names)) or not np.isfinite(value).all()
                   for value in arrays.values()):
                raise ValueError("Invalid saved response dimensions")
            archives[context] = (labels.tolist(), arrays)
        sources.append({"context": context, "path": str(path.resolve()), "sha256": sha256(path)})
    common = set.intersection(*(set(labels) for labels, _ in archives.values()))
    panel = spec.get("perturbations", sorted(common))
    if len(panel) < 2 or len(set(panel)) != len(panel) or not set(panel) <= common:
        raise ValueError("The balanced panel requires at least two distinct perturbations present in every context")
    has_truth = ["true_deltas" in values for _, values in archives.values()]
    if any(has_truth) and not all(has_truth):
        raise ValueError("Evaluation truth must be present for all contexts or none")
    result = {"contexts": np.asarray(contexts), "perturbations": np.asarray(panel),
              "feature_names": genes}
    for key in (["predictions", "true_deltas"] if all(has_truth) else ["predictions"]):
        result[key] = np.stack([archives[c][1][key][[archives[c][0].index(p) for p in panel]]
                                for c in contexts])
    return result, {"sources": sources, "cohort_selection": "explicit_balanced_panel" if
        "perturbations" in spec else "intersection_across_contexts",
        "assembly": "context-specific fitted outputs; not necessarily one joint model"}


def run(output, input_path=None, manifest=None):
    if (input_path is None) == (manifest is None):
        raise ValueError("Choose one labeled tensor input or prediction manifest")
    if manifest:
        arrays, provenance = assemble(manifest)
    else:
        arrays = load_panel(input_path)
        provenance = {"input_sha256": sha256(input_path), "input": str(Path(input_path).resolve())}
    output = Path(output).resolve()
    source_dirs = ([Path(p["path"]).parent for p in provenance["sources"]] if manifest else
                   [Path(input_path).resolve().parent])
    if any(output.is_relative_to(path) for path in source_dirs):
        raise ValueError("Analysis output must be outside input directories")
    results, component_arrays = {}, {}
    for key in ("true_deltas", "predictions"):
        if key in arrays:
            parts = decompose(arrays[key])
            results[key] = {"reconstruction_error": validate_decomposition(arrays[key], parts),
                           "second_moment_fractions": second_moment_fractions(parts)}
            component_arrays[key] = parts
    if len(component_arrays) == 2:
        scores, _ = compare(arrays["predictions"], arrays["true_deltas"])
        results["matching_components"] = scores
    output = new_directory(output)
    axes = {k: arrays[k] for k in ("contexts", "perturbations", "feature_names")}
    for key, values in component_arrays.items():
        np.savez_compressed(output / f"{key}_components.npz", **axes, **values)
    np.savez_compressed(output / "panel.npz", **arrays)
    write_json(output / "summary.json", {"protocol": "independent_balanced_components",
        "contexts": arrays["contexts"], "n_perturbations": len(arrays["perturbations"]),
        "n_genes": len(arrays["feature_names"]), "results": results,
        "note": "Existing component definitions; not the historical noise-corrected Figure 1 calculation."})
    write_json(output / "provenance.json", provenance)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Labeled NPZ response tensor")
    source.add_argument("--manifest", type=Path, help="JSON mapping contexts to prediction archives")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.execute:
            print(f"Analysis input: {args.input or args.manifest}\nOutput: {args.out_dir}\nUse --execute to run.")
            return
        from benchmarks._runtime import require_compute
        require_compute("Component decomposition")
        print(run(args.out_dir, args.input, args.manifest))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
