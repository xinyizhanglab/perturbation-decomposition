"""Pathway-level prediction analysis.

For each pathway × model, computes correlation of predicted vs true
pathway activity across held-out perturbations. Uses MSigDB Hallmark.

Usage:
    python -m benchmarks.pathway_prediction --data_dir data/processed/genetic
    python -m benchmarks.pathway_prediction --setting cross_cl
"""
import argparse, json, pickle, logging
from pathlib import Path
import numpy as np

from evaluation.metrics import pearson as corr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
PAIRED = {"k562": "rpe1", "rpe1": "k562", "hepg2": "jurkat", "jurkat": "hepg2"}
MIN_PATHWAY_GENES = 5
N_FOLDS = 5

HALLMARK_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"


def load_hallmark(gmt_path):
    if not gmt_path.exists():
        log.info(f"  Downloading Hallmark GMT...")
        import urllib.request
        urllib.request.urlretrieve(HALLMARK_URL, str(gmt_path))
    gene_sets = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3: continue
            name = parts[0].replace("HALLMARK_", "").replace("_", " ").title()
            gene_sets[name] = set(g.upper() for g in parts[2:])
    return gene_sets


def build_pathway_vectors(gene_sets, hvg_symbols, G):
    symbol_to_idx = {s.upper(): i for i, s in enumerate(hvg_symbols)
                     if s and not s.startswith("ENSG")}
    pvecs = {}
    for pname, pgenes in gene_sets.items():
        overlap = [g for g in pgenes if g in symbol_to_idx]
        if len(overlap) >= MIN_PATHWAY_GENES:
            vec = np.zeros(G, dtype=np.float32)
            for g in overlap:
                vec[symbol_to_idx[g]] = 1.0
            vec /= np.linalg.norm(vec) + 1e-10
            pvecs[pname] = vec
    return pvecs


def pathway_prediction_crosscl(data_dir, depmap_path, out_dir, gmt_path, save_checkpoints=False):
    """Cross-CL pathway prediction: Ridge/MLP trained on sources, predict target."""
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    log.info("\n=== Cross-CL pathway prediction ===")

    # Load data
    cl_data = {}
    hvg_names = None
    for cl in CLS:
        with open(Path(data_dir) / f"{cl}.pkl", "rb") as f:
            d = pickle.load(f)
        cl_data[cl] = d["deltas"]
        if hvg_names is None and "hvg_names" in d:
            hvg_names = d["hvg_names"]
        log.info(f"  {cl}: {len(cl_data[cl])} perts")

    with open(depmap_path, "rb") as f:
        depmap = pickle.load(f)
    def get_emb(p):
        return depmap.get(p, depmap.get(p.replace("+ctrl", ""), None))

    G = len(list(cl_data[CLS[0]].values())[0])
    gene_sets = load_hallmark(gmt_path)

    if hvg_names:
        pvecs = build_pathway_vectors(gene_sets, hvg_names, G)
    else:
        log.warning("  No HVG names available, trying Ensembl mapping...")
        try:
            import mygene
            # Would need var_names, skip if unavailable
            pvecs = {}
        except ImportError:
            pvecs = {}

    if not pvecs:
        log.error("  No pathway vectors could be built")
        return {}

    log.info(f"  Pathway vectors: {len(pvecs)}")

    results = {}
    for tgt in CLS:
        srcs = [c for c in CLS if c != tgt]
        tgt_perts = sorted(p for p in cl_data[tgt] if get_emb(p) is not None)
        tD = np.array([cl_data[tgt][p] for p in tgt_perts])
        tV = np.array([get_emb(p) for p in tgt_perts])
        T_tgt = tD.mean(0)

        # Source training
        sV, sD = [], []
        for c in srcs:
            for pn, d in cl_data[c].items():
                e = get_emb(pn)
                if e is not None:
                    sV.append(e); sD.append(d)
        sV = np.array(sV); sD = np.array(sD)

        # Ridge
        pca = PCA(50).fit(sV)
        ridge = Ridge(alpha=10).fit(pca.transform(sV), sD)
        yp_ridge = ridge.predict(pca.transform(tV))
        if save_checkpoints:
            if hvg_names is None or len(hvg_names) != G or len(set(hvg_names)) != G:
                raise ValueError("Checkpoint export requires verified ordered hvg_names")
            from models.checkpoints import save_estimator, save_predictions_with_axes
            from reproduction.io import sha256
            directory = Path(out_dir) / "checkpoints" / tgt
            save_estimator(pca, ridge, directory, hvg_names, "sha256:" + sha256(depmap_path),
                           {"experiment": "pathway_cross_cl", "target": tgt, "sources": srcs})
            save_predictions_with_axes(directory / "predictions.npz", tgt_perts, yp_ridge, tD, T_tgt, hvg_names)

        # Direct transfer
        yp_direct = []
        direct_perts = []
        for p in tgt_perts:
            src_resps = [cl_data[c][p] for c in srcs if p in cl_data[c]]
            if src_resps:
                yp_direct.append(np.mean(src_resps, axis=0))
                direct_perts.append(p)
        yp_direct = np.array(yp_direct) if yp_direct else np.zeros((0, G))

        for pname, pvec in pvecs.items():
            # Ridge
            true_act = (tD - T_tgt) @ pvec
            pred_act = (yp_ridge - T_tgt) @ pvec
            results[(tgt, pname, "Ridge")] = corr(true_act, pred_act)

            # Direct
            if len(direct_perts) >= 10:
                d_idx = [tgt_perts.index(p) for p in direct_perts]
                true_d = (tD[d_idx] - T_tgt) @ pvec
                pred_d = (yp_direct - T_tgt) @ pvec
                results[(tgt, pname, "Direct")] = corr(true_d, pred_d)

    # Summary
    all_pathways = sorted(set(pw for _, pw, _ in results))
    models = sorted(set(m for _, _, m in results))

    log.info(f"\n  Pathways: {len(all_pathways)}, Models: {models}")
    for pw in all_pathways[:10]:
        vals = {m: np.mean([results.get((cl, pw, m), 0) for cl in CLS]) for m in models}
        log.info(f"    {pw[:35]:<37} " + "  ".join(f"{m}={v:.3f}" for m, v in vals.items()))

    # Save
    results_save = {f"{cl}||{pw}||{m}": float(v) for (cl, pw, m), v in results.items()}
    with open(out_dir / "pathway_results.json", "w") as f:
        json.dump(results_save, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/genetic"))
    parser.add_argument("--depmap", type=Path, default=Path("data/processed/genetic/depmap_embeddings.pkl"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/pathway"))
    parser.add_argument("--gmt", type=Path, default=Path("data/processed/hallmark.gmt"))
    parser.add_argument("--setting", default="cross_cl", choices=["cross_cl", "within_cl"])
    parser.add_argument("--dry_run", action="store_true", help="Print inputs/settings without running")
    parser.add_argument("--save_checkpoints", action="store_true")
    args = parser.parse_args()
    if args.setting != "cross_cl":
        parser.error("within_cl pathway analysis is not implemented; use --setting cross_cl")
    from benchmarks._runtime import prepare_run
    try:
        if not prepare_run(args, [args.data_dir / f"{cl}.pkl" for cl in CLS] + [args.depmap, args.gmt]):
            return
    except (OSError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    if args.setting == "cross_cl":
        pathway_prediction_crosscl(args.data_dir, args.depmap, args.out_dir, args.gmt, args.save_checkpoints)


if __name__ == "__main__":
    main()
