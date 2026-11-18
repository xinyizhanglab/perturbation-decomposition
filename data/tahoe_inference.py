"""Prepare the recorded compound/source-response representations for Tahoe checkpoints."""

import argparse
from pathlib import Path

import numpy as np

from benchmarks.tahoe_drugs import CLS_TAHOE, _condition, load_tahoe
from models.checkpoints import response_axis_id
from reproduction.io import new_directory, read_json, sha256, write_json


def prepare(data_dir, target, conditions, gene_names, mode, output):
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if output.is_relative_to(data_dir):
        raise ValueError("Inference output must be outside the input dataset directory")
    _, responses, features = load_tahoe(data_dir)
    genes = read_json(gene_names)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Verified ordered gene identifiers are required")
    requested = read_json(conditions)
    labels, vectors, truths = [], [], []
    sources = [c for c in CLS_TAHOE if c != target]
    has_truth = True
    for drug, dose in requested:
        key = (drug, dose)
        label = _condition(drug, dose)
        if label in labels:
            raise ValueError("Duplicate drug/dose conditions")
        labels.append(label)
        if mode == "source":
            available = [responses[c][key] for c in sources if key in responses[c]]
            if not available:
                raise ValueError(f"No observed source response for {label}")
            vectors.append(np.mean(available, axis=0).astype(np.float32))
        else:
            name = "Morgan" if mode == "morgan" else "CC"
            if name not in features or drug not in features[name]:
                raise ValueError(f"Missing {name} compound descriptor: {drug}")
            vectors.append(features[name][drug])
        if key in responses[target]:
            truths.append(responses[target][key])
        else:
            has_truth = False
    if not vectors:
        raise ValueError("No requested conditions")
    if mode == "source":
        axis = response_axis_id(genes, sources)
    else:
        filename = "morgan_fingerprints.pkl" if mode == "morgan" else "cc_features.pkl"
        axis = "sha256:" + sha256(data_dir / filename)
    arrays = {"test_perts": np.asarray(labels), "features": np.asarray(vectors, dtype=np.float32),
              "descriptor_axis_id": np.asarray(axis), "feature_names": np.asarray(genes)}
    if has_truth:
        arrays["true_deltas"] = np.asarray(truths)
        if arrays["true_deltas"].shape != (len(labels), len(genes)):
            raise ValueError("Tahoe gene axis does not match the supplied identifiers")
    for value in arrays.values():
        if value.dtype.kind in "fci" and not np.isfinite(value).all():
            raise ValueError("Nonfinite prepared input")
    output = new_directory(output)
    np.savez_compressed(output / "inputs.npz", **arrays)
    write_json(output / "metadata.json", {"target": target, "representation": mode,
        "conditions_sha256": sha256(conditions), "gene_names_sha256": sha256(gene_names),
        "source_contexts": sources if mode == "source" else [], "evaluation_truth": has_truth})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/tahoe"))
    parser.add_argument("--target", choices=CLS_TAHOE, required=True)
    parser.add_argument("--conditions", type=Path, required=True, help="JSON list of [drug, dose] pairs")
    parser.add_argument("--gene_names", type=Path, required=True)
    parser.add_argument("--mode", choices=("source", "morgan", "cc"), required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(f"Tahoe {args.mode} inputs → {args.out_dir}; use --execute to run.")
        return
    from benchmarks._runtime import require_compute
    require_compute("Tahoe inference preparation")
    print(prepare(args.data_dir, args.target, args.conditions, args.gene_names, args.mode, args.out_dir))


if __name__ == "__main__":
    main()
