"""Data processing pipeline: raw h5ad → pseudobulk deltas + DepMap embeddings.

Usage:
    python -m data.process --dataset k562 --data_dir ./data/raw --out_dir ./data/processed
    python -m data.process --dataset all
"""
import argparse, pickle, logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)

GENETIC_DATASETS = {
    "k562": {"gene_col": "gene", "ctrl_label": "non-targeting"},
    "rpe1": {"gene_col": "gene", "ctrl_label": "non-targeting"},
    "hepg2": {"gene_col": "gene", "ctrl_label": "non-targeting"},
    "jurkat": {"gene_col": "gene", "ctrl_label": "non-targeting"},
}
MIN_CELLS = 5


def aligned_feature_names(adata, n_features, supplied=None):
    if supplied is not None:
        names = np.asarray(supplied, dtype=str)
    elif len(adata.var_names) == n_features:
        names = np.asarray(adata.var_names, dtype=str)
    elif "highly_variable" in adata.var.columns:
        mask = np.asarray(adata.var["highly_variable"], dtype=bool)
        names = np.asarray(adata.var_names, dtype=str)[mask]
    else:
        raise ValueError("Cannot identify X_hvg gene order; supply --hvg_names as an ordered JSON list")
    if names.shape != (n_features,) or len(set(names)) != n_features:
        raise ValueError("X_hvg requires exactly one unique gene name per feature")
    return names.tolist()


def process_genetic(h5ad_path, out_dir, dataset_name, min_cells=MIN_CELLS, hvg_names=None):
    """Process a genetic perturbation dataset into pseudobulk deltas.

    Input: h5ad with obs['gene'] (perturbation name) and obsm['X_hvg'] (HVG expression)
    Output: {dataset}.pkl with responses, controls, counts, and ordered genes
    """
    import anndata as ad
    output_path = Path(out_dir) / f"{dataset_name}.pkl"
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    cfg = GENETIC_DATASETS[dataset_name]
    log.info(f"Processing {dataset_name} from {h5ad_path}")

    a = ad.read_h5ad(h5ad_path)
    X = np.array(a.obsm["X_hvg"]).astype(np.float32)
    genes = a.obs[cfg["gene_col"]].astype(str).values
    if X.ndim != 2 or len(genes) != X.shape[0] or not np.isfinite(X).all():
        raise ValueError("X_hvg must be a finite cell-by-gene matrix matching observation labels")
    names = aligned_feature_names(a, X.shape[1], hvg_names)
    ctrl_mask = genes == cfg["ctrl_label"]
    if not ctrl_mask.any():
        raise ValueError(f"No control cells with label {cfg['ctrl_label']!r}")
    ctrl_mean = X[ctrl_mask].mean(0)

    log.info(f"  Shape: {a.shape}, HVGs: {X.shape[1]}, ctrl: {ctrl_mask.sum()}")

    # Pseudobulk deltas
    deltas = {}
    cell_counts = {}
    for pn in sorted(set(genes) - {cfg["ctrl_label"]}):
        mask = genes == pn
        n = mask.sum()
        if n >= min_cells:
            deltas[pn] = X[mask].mean(0) - ctrl_mean
            cell_counts[pn] = n

    log.info(f"  Perturbations: {len(deltas)}")

    # Save
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "deltas": deltas,
        "ctrl_mean": ctrl_mean,
        "cell_counts": cell_counts,
        "n_hvgs": X.shape[1],
        "var_names": list(a.var_names) if hasattr(a, "var_names") else None,
        "hvg_names": names,
        "dataset": dataset_name,
    }

    with output_path.open("xb") as f:
        pickle.dump(out, f)

    log.info(f"  Saved: {output_path}")
    del a
    return out


def build_depmap_embeddings(depmap_csv_path, gene_list, out_path):
    """Extract DepMap dependency profiles for a list of perturbation genes.

    Args:
        depmap_csv_path: path to CRISPRGeneEffect.csv
        gene_list: list of gene symbols to extract
        out_path: output pickle path
    """
    log.info(f"Building DepMap embeddings from {depmap_csv_path}")
    import pandas as pd
    df = pd.read_csv(depmap_csv_path, index_col=0)

    # Parse column headers: "GENE (ENTREZ_ID)" -> "GENE"
    col_to_symbol = {}
    for col in df.columns:
        if " (" in col:
            symbol = col.split(" (")[0]
            col_to_symbol[symbol] = col
        else:
            col_to_symbol[col] = col

    emb = {}
    for gene in gene_list:
        if gene in col_to_symbol:
            vec = df[col_to_symbol[gene]].values.astype(np.float32)
            vec = np.nan_to_num(vec, 0.0)
            emb[gene] = vec

    log.info(f"  Coverage: {len(emb)}/{len(gene_list)}")
    log.info(f"  Embedding dim: {len(list(emb.values())[0])}")

    with open(out_path, "wb") as f:
        pickle.dump(emb, f)

    log.info(f"  Saved: {out_path}")
    return emb


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(GENETIC_DATASETS.keys()) + ["all"])
    parser.add_argument("--h5ad", type=str, default=None,
                        help="Path to input h5ad (required for single dataset)")
    parser.add_argument("--data_dir", type=str, default="./data/raw")
    parser.add_argument("--out_dir", type=str, default="./data/processed/genetic")
    parser.add_argument("--hvg_names", type=Path, help="JSON list of gene names in X_hvg column order")
    parser.add_argument("--depmap_csv", type=str, default=None,
                        help="Path to CRISPRGeneEffect.csv")
    args = parser.parse_args()

    datasets = list(GENETIC_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    if args.h5ad and len(datasets) != 1:
        parser.error("--h5ad requires a single --dataset")
    input_paths = {ds: Path(args.h5ad) if args.h5ad else Path(args.data_dir) / ds / "replogle.h5ad"
                   for ds in datasets}
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        parser.error("Missing raw inputs: " + ", ".join(missing))
    from benchmarks._runtime import require_compute
    require_compute("Raw-data processing")
    import json
    supplied = json.loads(args.hvg_names.read_text()) if args.hvg_names else None

    all_genes = set()
    for ds in datasets:
        if args.h5ad:
            h5ad_path = args.h5ad
        else:
            h5ad_path = Path(args.data_dir) / f"{ds}" / "replogle.h5ad"
        if not Path(h5ad_path).exists():
            log.warning(f"  {h5ad_path} not found, skipping")
            continue
        out = process_genetic(h5ad_path, args.out_dir, ds, hvg_names=supplied)
        all_genes |= set(out["deltas"].keys())

    if args.depmap_csv and all_genes:
        build_depmap_embeddings(
            args.depmap_csv, sorted(all_genes),
            Path(args.out_dir) / "depmap_embeddings.pkl")


if __name__ == "__main__":
    main()
