"""Prepare release data files from cluster artifacts.

Run ON THE CLUSTER to generate the small data files that ship with the repo.
These are pseudobulk deltas (dict of mean vectors), not raw h5ad.

Usage (on cluster):
    python -m data.prepare_release --cluster_dir /mnt/storage01/home/amolina/benchmark_2026

Outputs to data/processed/:
    genetic/{k562,rpe1,hepg2,jurkat}.pkl     (~2 MB each)
    genetic/depmap_embeddings.pkl             (~10 MB)
    norman/norman.pkl                         (~1 MB)
    norman/depmap_embeddings.pkl              (~1 MB)
    norman/split_info.json
    tahoe/pseudobulk_deltas.pkl               (~30 MB)
    tahoe/morgan_fingerprints.pkl             (~2 MB)
    tahoe/cc_features.pkl                     (~3 MB)
    tahoe/drug_moa.json
    tahoe/smiles.json
    depmap/CRISPRGeneEffect.csv               (~100 MB, or download separately)
"""
import argparse, pickle, json, logging, shutil
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()


def export_genetic(cluster_dir, out_dir):
    """Export genetic perturbation pseudobulk data."""
    import anndata as ad

    genetic_out = out_dir / "genetic"
    genetic_out.mkdir(parents=True, exist_ok=True)

    CLS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
           "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

    all_genes = set()
    for cl, ds in CLS.items():
        h5ad_path = cluster_dir / "data" / cl / "replogle.h5ad"
        if not h5ad_path.exists():
            log.warning(f"  {h5ad_path} not found")
            continue

        log.info(f"  Processing {cl}...")
        a = ad.read_h5ad(h5ad_path)
        X = np.array(a.obsm["X_hvg"]).astype(np.float32)
        genes = a.obs["gene"].astype(str).values
        ctrl_mean = X[genes == "non-targeting"].mean(0)

        deltas = {}
        cell_counts = {}
        for pn in sorted(set(genes) - {"non-targeting"}):
            mask = genes == pn
            n = mask.sum()
            if n >= 5:
                deltas[pn] = X[mask].mean(0) - ctrl_mean
                cell_counts[pn] = int(n)
                all_genes.add(pn)

        # Try to get HVG var names
        hvg_names = None
        if "highly_variable" in a.var.columns:
            hvg_names = list(a.var_names[a.var["highly_variable"]])
        else:
            hvg_names = list(a.var_names[:X.shape[1]])

        out_data = {
            "deltas": deltas,
            "ctrl_mean": ctrl_mean,
            "cell_counts": cell_counts,
            "n_hvgs": X.shape[1],
            "hvg_names": hvg_names,
            "dataset": ds,
        }
        with open(genetic_out / f"{cl}.pkl", "wb") as f:
            pickle.dump(out_data, f)
        log.info(f"    {cl}: {len(deltas)} perts, saved {cl}.pkl")
        del a

    # DepMap embeddings for genetic perturbation genes
    morph_base = cluster_dir.parent / "agop-perturbation/morph_pipeline"
    emb_dict = {}
    for ds in CLS.values():
        ep = morph_base / f"outputs/{ds}/pert_embeddings.pkl"
        if ep.exists():
            with open(ep, "rb") as f:
                raw = pickle.load(f)
            for k, v in raw.items():
                gene = k.replace("+ctrl", "")
                if gene in all_genes:
                    emb_dict[gene] = v.astype(np.float32)

    with open(genetic_out / "depmap_embeddings.pkl", "wb") as f:
        pickle.dump(emb_dict, f)
    log.info(f"  DepMap: {len(emb_dict)} genes, saved depmap_embeddings.pkl")


def export_norman(cluster_dir, out_dir):
    """Export Norman combinatorial perturbation data."""
    import anndata as ad

    norman_out = out_dir / "norman"
    norman_out.mkdir(parents=True, exist_ok=True)

    st_dir = cluster_dir.parent / "agop-perturbation/st_agop_pipeline/data/st/norman"
    if not st_dir.exists():
        log.warning(f"  Norman ST data not found at {st_dir}")
        return

    log.info("  Processing Norman...")
    a_train = ad.read_h5ad(st_dir / "train.h5ad")
    a_test = ad.read_h5ad(st_dir / "test.h5ad")

    X_all = np.vstack([np.array(a_train.obsm["X_hvg"]),
                       np.array(a_test.obsm["X_hvg"])]).astype(np.float32)
    conds_all = np.concatenate([a_train.obs["condition"].astype(str).values,
                                 a_test.obs["condition"].astype(str).values])
    ctrl_mean = X_all[conds_all == "ctrl"].mean(0)

    deltas = {}
    cell_counts = {}
    for c in sorted(set(conds_all) - {"ctrl"}):
        mask = conds_all == c
        n = mask.sum()
        if n >= 5:
            deltas[c] = X_all[mask].mean(0) - ctrl_mean
            cell_counts[c] = int(n)

    # Split info
    si = json.load(open(st_dir / "split_info.json"))

    out_data = {
        "deltas": deltas,
        "ctrl_mean": ctrl_mean,
        "cell_counts": cell_counts,
        "n_hvgs": X_all.shape[1],
    }
    with open(norman_out / "norman.pkl", "wb") as f:
        pickle.dump(out_data, f)
    with open(norman_out / "split_info.json", "w") as f:
        json.dump(si, f, indent=2)

    # Norman DepMap embeddings
    depmap_path = cluster_dir / "data/norman/norman_depmap_embeddings.pkl"
    if depmap_path.exists():
        shutil.copy(depmap_path, norman_out / "depmap_embeddings.pkl")
        log.info("    Copied norman depmap_embeddings.pkl")

    log.info(f"    Norman: {len(deltas)} conditions, saved")
    del a_train, a_test


def export_tahoe(cluster_dir, out_dir):
    """Export Tahoe drug perturbation data."""
    tahoe_out = out_dir / "tahoe"
    tahoe_out.mkdir(parents=True, exist_ok=True)

    tahoe_data = cluster_dir / "data/tahoe_cc"

    for fname in ["pseudobulk_deltas.pkl", "morgan_fingerprints.pkl",
                  "cc_features.pkl", "drug_moa.json", "smiles.json",
                  "drug_inchikeys.json"]:
        src = tahoe_data / fname
        if src.exists():
            shutil.copy(src, tahoe_out / fname)
            log.info(f"    Copied {fname}")
        else:
            log.warning(f"    {fname} not found")


def export_depmap_raw(cluster_dir, out_dir):
    """Copy raw DepMap CSV (optional, large)."""
    depmap_out = out_dir / "depmap"
    depmap_out.mkdir(parents=True, exist_ok=True)

    src = cluster_dir.parent / "agop-perturbation/data/depmap/CRISPRGeneEffect.csv"
    if src.exists():
        shutil.copy(src, depmap_out / "CRISPRGeneEffect.csv")
        log.info(f"    Copied CRISPRGeneEffect.csv ({src.stat().st_size / 1e6:.0f} MB)")
    else:
        log.info("    CRISPRGeneEffect.csv not found (download from DepMap portal)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_dir", type=Path,
                        default=Path("/mnt/storage01/home/amolina/benchmark_2026"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--include_depmap_csv", action="store_true",
                        help="Include raw CRISPRGeneEffect.csv (~100 MB)")
    args = parser.parse_args()

    log.info("=== Preparing release data ===")
    export_genetic(args.cluster_dir, args.out_dir)
    export_norman(args.cluster_dir, args.out_dir)
    export_tahoe(args.cluster_dir, args.out_dir)
    if args.include_depmap_csv:
        export_depmap_raw(args.cluster_dir, args.out_dir)

    # Report sizes
    log.info("\n=== Output sizes ===")
    for p in sorted(args.out_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            log.info(f"  {p.relative_to(args.out_dir)}: {size_mb:.1f} MB")

    log.info(f"\nDone. Total: {sum(p.stat().st_size for p in args.out_dir.rglob('*') if p.is_file()) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
