"""Prepare labeled, truth-optional inference arrays from processed inputs."""

import argparse
from pathlib import Path
import pickle

import numpy as np

from reproduction.io import new_directory, read_json, sha256, write_json
from models.inference_io import prediction_labels


def perturbation_names(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            values = prediction_labels(data)
    else:
        values = np.asarray(read_json(path), dtype=str)
    if values.ndim != 1 or not len(values) or len(set(values)) != len(values):
        raise ValueError("Perturbations must be unique ordered names")
    return values


def prepare(dataset, depmap, perturbations, output, controls=None, conditioning=None,
            perturbation_labels=None, pair_sum=False, gene_names=None):
    names = perturbation_names(perturbations)
    with Path(dataset).open("rb") as stream:
        raw = pickle.load(stream)
    feature_names = np.asarray(read_json(gene_names) if gene_names else
                               raw.get("hvg_names", raw.get("var_names")), dtype=str)
    if feature_names.ndim != 1 or len(set(feature_names)) != len(feature_names):
        raise ValueError("Processed data require unique ordered gene names")
    arrays = {"test_perts": names, "feature_names": feature_names,
              "control_mean": np.asarray(raw["ctrl_mean"])}
    if arrays["control_mean"].shape != (len(feature_names),):
        raise ValueError("Control mean differs from gene labels")
    if all(name in raw["deltas"] for name in names):
        arrays["true_deltas"] = np.stack([raw["deltas"][name] for name in names])
    if depmap:
        depmap = Path(depmap)
        if depmap.suffix == ".npz":
            with np.load(depmap, allow_pickle=False) as saved:
                descriptors = dict(zip(saved["perturbations"].astype(str), saved["embeddings"]))
        else:
            with depmap.open("rb") as stream:
                descriptors = pickle.load(stream)
        values = []
        for name in names:
            genes = name.split("+") if pair_sum else [name]
            if pair_sum and (len(genes) != 2 or "ctrl" in genes):
                raise ValueError("pair_sum requires two-gene conditions")
            vectors = []
            for gene in genes:
                key = gene if gene in descriptors else gene + "+ctrl"
                if key not in descriptors:
                    raise ValueError(f"Descriptor is missing; no perturbations are silently skipped: {gene}")
                vectors.append(descriptors[key])
            values.append(vectors[0] + vectors[1] if pair_sum else vectors[0])
        arrays["features"] = np.asarray(values, dtype=np.float32)
        arrays["descriptors"] = arrays["features"]
        arrays["descriptor_axis_id"] = np.asarray("sha256:" + sha256(depmap) + (":pair_sum" if pair_sum else ""))
    if controls:
        with np.load(controls, allow_pickle=False) as saved:
            if not np.array_equal(saved["feature_names"].astype(str), feature_names):
                raise ValueError("Control-cell genes differ from processed responses")
            if "control_mean" in saved and not np.allclose(saved["control_mean"], arrays["control_mean"], atol=1e-5, rtol=1e-5):
                raise ValueError("Raw and processed control means differ; check expression space/normalization")
            arrays["controls"] = saved["controls"].copy()
    elif "ctrl_X" in raw:
        arrays["controls"] = np.asarray(raw["ctrl_X"])
    if conditioning:
        with np.load(conditioning, allow_pickle=False) as saved:
            arrays["conditioning_mean"] = saved["train_mean"].copy()
    if perturbation_labels:
        arrays["perturbation_labels"] = np.asarray(read_json(perturbation_labels), dtype=str)
    for key, value in arrays.items():
        if value.dtype.kind in "biufc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite array: {key}")
    output = Path(output).resolve()
    protected = [Path(p).resolve().parent for p in
                 (dataset, depmap, perturbations, controls, conditioning, perturbation_labels, gene_names) if p]
    if any(output.is_relative_to(path) for path in protected):
        raise ValueError("Prepared outputs must be outside source data directories")
    directory = new_directory(output)
    np.savez_compressed(directory / "inputs.npz", **arrays)
    write_json(directory / "metadata.json", {
        "dataset_sha256": sha256(dataset), "perturbation_list_sha256": sha256(perturbations),
        "descriptor_sha256": sha256(depmap) if depmap else None,
        "real_controls_available": "controls" in arrays,
        "evaluation_truth_available": "true_deltas" in arrays})
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Trusted processed cell-line pickle")
    parser.add_argument("--depmap", type=Path)
    parser.add_argument("--perturbations", required=True, type=Path, help="JSON list or saved prediction NPZ")
    parser.add_argument("--controls", type=Path, help="NPZ with real controls and feature_names")
    parser.add_argument("--conditioning", type=Path, help="Recorded benchmark MORPH prediction archive")
    parser.add_argument("--perturbation_labels", type=Path, help="Recorded STATE one-hot ordering JSON")
    parser.add_argument("--pair_sum", action="store_true", help="Norman two-gene summed DepMap representation")
    parser.add_argument("--gene_names", type=Path, help="Verified ordered genes for legacy processed files")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.execute:
            print(f"Prepare {args.dataset} → {args.out_dir}/inputs.npz; use --execute to run.")
            return
        from benchmarks._runtime import require_compute
        require_compute("Inference input preparation")
        print(prepare(args.dataset, args.depmap, args.perturbations, args.out_dir,
                      args.controls, args.conditioning, args.perturbation_labels, args.pair_sum, args.gene_names))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
