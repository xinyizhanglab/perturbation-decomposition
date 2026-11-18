"""Export real control cells and their declared gene axis for native inference."""

import argparse
from pathlib import Path

import numpy as np

from data.process import aligned_feature_names
from reproduction.io import new_directory, write_json


def export(source, output, matrix_key="X_hvg", label_key="gene", control_label="non-targeting",
           gene_name_key=None, max_cells=None, seed=42):
    import anndata as ad
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source.parent):
        raise ValueError("Control exports must be outside the raw input directory")
    data = ad.read_h5ad(source, backed="r")
    try:
        labels = data.obs[label_key].astype(str).to_numpy()
        rows = np.flatnonzero(labels == control_label)
        if not len(rows):
            raise ValueError("No control cells with the requested label")
        matrix = data.X if matrix_key == "X" else data.obsm[matrix_key]
        controls = matrix[rows]
        if hasattr(controls, "toarray"):
            controls = controls.toarray()
        controls = np.asarray(controls)
        if controls.ndim != 2 or not np.isfinite(controls).all():
            raise ValueError("Control expression must be finite cells × genes")
        supplied = None
        if gene_name_key:
            supplied = data.var[gene_name_key].astype(str).to_numpy()
            if len(supplied) != controls.shape[1]:
                if "highly_variable" not in data.var:
                    raise ValueError("The gene-name column does not identify X_hvg columns")
                supplied = supplied[data.var["highly_variable"].to_numpy(dtype=bool)]
        genes = aligned_feature_names(data, controls.shape[1], supplied)
        mean = controls.mean(axis=0)
        population_size = len(rows)
        if max_cells is not None:
            if max_cells < 1:
                raise ValueError("--max_cells must be positive")
            if len(rows) > max_cells:
                selected = np.sort(np.random.RandomState(seed).choice(len(rows), max_cells, replace=False))
                rows, controls = rows[selected], controls[selected]
    finally:
        data.file.close()
    output = new_directory(output)
    np.savez_compressed(output / "controls.npz", controls=controls, control_mean=mean,
                        feature_names=np.asarray(genes), source_row_indices=rows)
    write_json(output / "feature_names.json", genes)
    write_json(output / "metadata.json", {"source": str(source), "matrix_key": matrix_key,
        "label_key": label_key, "control_label": control_label, "population_size": population_size,
        "exported_cells": len(controls), "sampling_seed": seed,
        "control_mean_definition": "all matching source controls before optional subsampling"})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--matrix_key", default="X_hvg")
    parser.add_argument("--label_key", default="gene")
    parser.add_argument("--control_label", default="non-targeting")
    parser.add_argument("--gene_name_key")
    parser.add_argument("--max_cells", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.execute:
            print(f"Control export: {args.h5ad} → {args.out_dir}/controls.npz; use --execute to run.")
            return
        from benchmarks._runtime import require_compute
        require_compute("Raw control-cell extraction")
        print(export(args.h5ad, args.out_dir, args.matrix_key, args.label_key,
                     args.control_label, args.gene_name_key, args.max_cells, args.seed))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
