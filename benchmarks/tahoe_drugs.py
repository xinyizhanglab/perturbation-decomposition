"""Tahoe drug perturbation benchmark.

Compares compound priors (Morgan, Chemical Checker) against
source-response transfer across 5 cell lines.

Usage:
    python -m benchmarks.tahoe_drugs --data_dir data/processed/tahoe
    python -m benchmarks.tahoe_drugs --features source --rank_sweep
    python -m benchmarks.tahoe_drugs --moa_holdout
"""
import argparse, json, pickle, logging
from pathlib import Path
from collections import defaultdict
import numpy as np
from urllib.parse import quote

from evaluation.metrics import pearson as corr, pref

log = logging.getLogger(__name__)

N_FOLDS = 5
CLS_TAHOE = ["C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1"]


def load_tahoe(data_dir):
    data_dir = Path(data_dir)
    with open(data_dir / "pseudobulk_deltas.pkl", "rb") as f:
        pdata = pickle.load(f)

    feats = {}
    for name, fname in [("Morgan", "morgan_fingerprints.pkl"),
                        ("CC", "cc_features.pkl")]:
        p = data_dir / fname
        if p.exists():
            with open(p, "rb") as f:
                feats[name] = pickle.load(f)
            log.info(f"  {name}: {len(feats[name])} drugs")

    cl_responses = {}
    for cl in CLS_TAHOE:
        cl_responses[cl] = {}
        for (c, drug, conc), delta in pdata["deltas"].items():
            if c == cl:
                cl_responses[cl][(drug, conc)] = delta

    return pdata, cl_responses, feats


def _export_fit(options, group, pca, ridge, labels, predictions, truth, reference, axis_id, metadata):
    if options is None:
        return
    from models.checkpoints import save_estimator, save_predictions_with_axes
    directory = Path(options["root"]) / "checkpoints" / group
    save_estimator(pca, ridge, directory, options["genes"], axis_id, metadata)
    save_predictions_with_axes(directory / "predictions.npz", labels, predictions, truth,
                               reference, options["genes"])


def _condition(drug, dose):
    return json.dumps([str(drug), str(dose)], separators=(",", ":"))


def run_compound_cv(cl_responses, feats, out_dir, checkpoint_options=None):
    """5-fold CV with compound priors (Morgan, CC)."""
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    log.info("\n=== Compound-prior benchmark (5-fold CV) ===")
    results = {}

    for tgt in CLS_TAHOE:
        for feat_name, feat_dict in feats.items():
            obs = [{"drug": dc[0], "dose": dc[1], "feat": feat_dict[dc[0]], "tgt": cl_responses[tgt][dc]}
                   for dc in sorted(cl_responses[tgt].keys())
                   if dc[0] in feat_dict]
            if len(obs) < 30: continue

            drugs = sorted(set(o["drug"] for o in obs))
            drug_labels = np.array([o["drug"] for o in obs])
            unique_drugs = np.array(drugs)
            V = np.array([o["feat"] for o in obs], dtype=np.float32)
            D = np.array([o["tgt"] for o in obs], dtype=np.float32)

            rng = np.random.RandomState(42)
            perm = rng.permutation(len(unique_drugs))
            folds = np.array_split(perm, N_FOLDS)

            fold_prefs = []
            for fold in range(N_FOLDS):
                test_drugs = set(unique_drugs[i] for i in folds[fold])
                tr = np.array([d not in test_drugs for d in drug_labels])
                te = ~tr
                if tr.sum() < 10 or te.sum() < 5: continue

                nc = min(50, tr.sum() - 1, V.shape[1])
                pca = PCA(nc).fit(V[tr])
                ridge = Ridge(alpha=10).fit(pca.transform(V[tr]), D[tr])
                yp = ridge.predict(pca.transform(V[te]))
                fold_prefs.append(pref(D[te], yp, D[tr].mean(0)))
                _export_fit(checkpoint_options, f"compound/{quote(tgt, safe='')}/{feat_name}/fold{fold}",
                    pca, ridge, [_condition(o["drug"], o["dose"]) for o, keep in zip(obs, te) if keep],
                    yp, D[te], D[tr].mean(0),
                    checkpoint_options["axis_ids"][feat_name] if checkpoint_options else "",
                    {"experiment": "tahoe_compound_cv", "target": tgt, "fold": fold,
                     "feature_builder": feat_name, "test_drugs": sorted(test_drugs)})

            if fold_prefs:
                results[f"{tgt}__{feat_name}"] = {
                    "mean": float(np.mean(fold_prefs)),
                    "std": float(np.std(fold_prefs)),
                }
                log.info(f"  {tgt} {feat_name}: {np.mean(fold_prefs):.4f} ± {np.std(fold_prefs):.4f}")

    return results


def run_source_transfer(cl_responses, out_dir, ranks=(10, 50, 100), checkpoint_options=None):
    """Source-response transfer with SVD rank sweep."""
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    log.info("\n=== Source-response transfer ===")
    results = {}

    for tgt in CLS_TAHOE:
        srcs = [c for c in CLS_TAHOE if c != tgt]
        obs = []
        for dc in sorted(cl_responses[tgt].keys()):
            src_resps = [cl_responses[s][dc] for s in srcs if dc in cl_responses[s]]
            if src_resps:
                obs.append({"drug": dc[0], "dose": dc[1], "dc": dc,
                            "src": np.mean(src_resps, axis=0).astype(np.float32),
                            "tgt": cl_responses[tgt][dc].astype(np.float32)})
        if len(obs) < 30: continue

        drugs = sorted(set(o["drug"] for o in obs))
        drug_labels = np.array([o["drug"] for o in obs])
        unique_drugs = np.array(drugs)
        V_src = np.array([o["src"] for o in obs])
        D_tgt = np.array([o["tgt"] for o in obs])

        rng = np.random.RandomState(42)
        perm = rng.permutation(len(unique_drugs))
        folds = np.array_split(perm, N_FOLDS)

        # Direct (zero-param)
        direct_prefs = []
        for fold in range(N_FOLDS):
            test_drugs = set(unique_drugs[i] for i in folds[fold])
            tr = np.array([d not in test_drugs for d in drug_labels])
            te = ~tr
            if tr.sum() < 10 or te.sum() < 5: continue
            direct_prefs.append(pref(D_tgt[te], V_src[te], D_tgt[tr].mean(0)))
        results[f"{tgt}__Direct"] = {"mean": float(np.mean(direct_prefs))}

        # Ridge at different ranks
        for k in ranks:
            fold_prefs = []
            for fold in range(N_FOLDS):
                test_drugs = set(unique_drugs[i] for i in folds[fold])
                tr = np.array([d not in test_drugs for d in drug_labels])
                te = ~tr
                if tr.sum() < 10 or te.sum() < 5: continue

                nc = min(k, tr.sum() - 1, V_src.shape[1])
                pca = PCA(nc).fit(V_src[tr])
                ridge = Ridge(alpha=10).fit(pca.transform(V_src[tr]), D_tgt[tr])
                yp = ridge.predict(pca.transform(V_src[te]))
                fold_prefs.append(pref(D_tgt[te], yp, D_tgt[tr].mean(0)))
                from models.checkpoints import response_axis_id
                _export_fit(checkpoint_options, f"source/{quote(tgt, safe='')}/rank{k}/fold{fold}",
                    pca, ridge, [_condition(o["drug"], o["dose"]) for o, keep in zip(obs, te) if keep],
                    yp, D_tgt[te], D_tgt[tr].mean(0),
                    response_axis_id(checkpoint_options["genes"], srcs) if checkpoint_options else "",
                    {"experiment": "tahoe_source_cv", "target": tgt, "fold": fold, "rank": k,
                     "feature_builder": "mean_available_source_response", "source_contexts": srcs})

            results[f"{tgt}__Source_k{k}"] = {
                "mean": float(np.mean(fold_prefs)),
                "std": float(np.std(fold_prefs)),
            }

        log.info(f"  {tgt}: Direct={results[f'{tgt}__Direct']['mean']:.4f}  " +
                 "  ".join(f"k={k}:{results[f'{tgt}__Source_k{k}']['mean']:.3f}" for k in ranks))

    return results


def run_moa_holdout(cl_responses, feats, data_dir, checkpoint_options=None):
    """Leave-one-MoA-out evaluation."""
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    log.info("\n=== MoA-held-out ===")

    moa_path = Path(data_dir) / "drug_moa.json"
    if not moa_path.exists():
        log.warning("  drug_moa.json not found")
        return {}

    with open(moa_path) as f:
        drug_moa = json.load(f)

    moa_drugs = defaultdict(set)
    for drug, moa in drug_moa.items():
        if moa != "unclear":
            moa_drugs[moa].add(drug)
    valid_moas = {m: d for m, d in moa_drugs.items() if len(d) >= 3}
    all_moa_drugs = set()
    for d in valid_moas.values():
        all_moa_drugs |= d

    results = {}

    for tgt in CLS_TAHOE:
        srcs = [c for c in CLS_TAHOE if c != tgt]

        # Source transfer MoA-held-out
        obs = []
        for dc in sorted(cl_responses[tgt].keys()):
            drug = dc[0]
            if drug not in all_moa_drugs: continue
            moa = drug_moa.get(drug, "unclear")
            if moa == "unclear": continue
            src_resps = [cl_responses[s][dc] for s in srcs if dc in cl_responses[s]]
            if src_resps:
                obs.append({"drug": drug, "dose": dc[1], "moa": moa,
                            "src": np.mean(src_resps, axis=0).astype(np.float32),
                            "tgt": cl_responses[tgt][dc].astype(np.float32)})

        if len(obs) < 20: continue

        moa_prefs_source = []
        moa_prefs_morgan = []

        for held_moa in valid_moas:
            train = [o for o in obs if o["moa"] != held_moa]
            test = [o for o in obs if o["moa"] == held_moa]
            if len(train) < 10 or len(test) < 2: continue

            V_tr = np.array([o["src"] for o in train])
            D_tr = np.array([o["tgt"] for o in train])
            V_te = np.array([o["src"] for o in test])
            D_te = np.array([o["tgt"] for o in test])
            T_tr = D_tr.mean(0)

            nc = min(50, V_tr.shape[0] - 1)
            pca = PCA(nc).fit(V_tr)
            ridge = Ridge(alpha=10).fit(pca.transform(V_tr), D_tr)
            yp = ridge.predict(pca.transform(V_te))
            moa_prefs_source.append(pref(D_te, yp, T_tr))
            from models.checkpoints import response_axis_id
            _export_fit(checkpoint_options, f"moa-source/{quote(tgt, safe='')}/{quote(held_moa, safe='')}",
                pca, ridge, [_condition(o["drug"], o["dose"]) for o in test], yp, D_te, T_tr,
                response_axis_id(checkpoint_options["genes"], srcs) if checkpoint_options else "",
                {"experiment": "tahoe_moa_source", "target": tgt, "held_moa": held_moa,
                 "feature_builder": "mean_available_source_response", "source_contexts": srcs})

            # Morgan MoA-held-out
            if "Morgan" in feats:
                train_m = [o for o in train if o["drug"] in feats["Morgan"]]
                test_m = [o for o in test if o["drug"] in feats["Morgan"]]
                if len(train_m) >= 5 and len(test_m) >= 2:
                    Vm_tr = np.array([feats["Morgan"][o["drug"]] for o in train_m])
                    Dm_tr = np.array([o["tgt"] for o in train_m])
                    Vm_te = np.array([feats["Morgan"][o["drug"]] for o in test_m])
                    Dm_te = np.array([o["tgt"] for o in test_m])
                    nc_m = min(50, Vm_tr.shape[0] - 1, Vm_tr.shape[1])
                    pca_m = PCA(nc_m).fit(Vm_tr)
                    ridge_m = Ridge(alpha=10).fit(pca_m.transform(Vm_tr), Dm_tr)
                    yp_m = ridge_m.predict(pca_m.transform(Vm_te))
                    moa_prefs_morgan.append(pref(Dm_te, yp_m, Dm_tr.mean(0)))
                    _export_fit(checkpoint_options, f"moa-Morgan/{quote(tgt, safe='')}/{quote(held_moa, safe='')}",
                        pca_m, ridge_m, [_condition(o["drug"], o["dose"]) for o in test_m],
                        yp_m, Dm_te, Dm_tr.mean(0),
                        checkpoint_options["axis_ids"]["Morgan"] if checkpoint_options else "",
                        {"experiment": "tahoe_moa_compound", "target": tgt, "held_moa": held_moa,
                         "feature_builder": "Morgan"})

        results[f"{tgt}__Source_MoA"] = float(np.mean(moa_prefs_source)) if moa_prefs_source else 0
        results[f"{tgt}__Morgan_MoA"] = float(np.mean(moa_prefs_morgan)) if moa_prefs_morgan else 0
        log.info(f"  {tgt}: Source MoA-out={results[f'{tgt}__Source_MoA']:.4f}  "
                 f"Morgan MoA-out={results[f'{tgt}__Morgan_MoA']:.4f}")

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/tahoe"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/tahoe"))
    parser.add_argument("--features", default="all", choices=["morgan", "cc", "source", "all"])
    parser.add_argument("--rank_sweep", action="store_true")
    parser.add_argument("--moa_holdout", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print inputs/settings without running")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--gene_names", type=Path, help="Verified response-gene ordering JSON")
    args = parser.parse_args()
    if args.save_checkpoints and not args.gene_names:
        parser.error("--save_checkpoints requires --gene_names")
    paths = [args.data_dir / "pseudobulk_deltas.pkl"]
    if args.features in ("morgan", "all"):
        paths.append(args.data_dir / "morgan_fingerprints.pkl")
    if args.features in ("cc", "all"):
        paths.append(args.data_dir / "cc_features.pkl")
    if args.moa_holdout:
        paths.append(args.data_dir / "drug_moa.json")
    from benchmarks._runtime import prepare_run
    try:
        if not prepare_run(args, paths):
            return
    except (OSError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    pdata, cl_responses, feats = load_tahoe(args.data_dir)
    checkpoint_options = None
    if args.save_checkpoints:
        from reproduction.io import sha256
        genes = json.loads(args.gene_names.read_text())
        width = len(next(iter(pdata["deltas"].values())))
        if len(genes) != width or len(set(genes)) != len(genes):
            parser.error("Response-gene ordering does not match Tahoe deltas")
        checkpoint_options = {"root": args.out_dir, "genes": genes, "axis_ids": {
            name: "sha256:" + sha256(args.data_dir / filename)
            for name, filename in (("Morgan", "morgan_fingerprints.pkl"), ("CC", "cc_features.pkl"))
            if (args.data_dir / filename).is_file()}}
    log.info(f"  Cell lines: {list(cl_responses.keys())}")
    log.info(f"  Total observations: {sum(len(v) for v in cl_responses.values())}")

    all_results = {}

    if args.features in ("morgan", "cc", "all"):
        all_results["compound_cv"] = run_compound_cv(cl_responses, feats, args.out_dir, checkpoint_options)

    if args.features in ("source", "all"):
        ranks = (1, 2, 5, 10, 20, 50, 100, 150) if args.rank_sweep else (10, 50)
        all_results["source_transfer"] = run_source_transfer(cl_responses, args.out_dir, ranks, checkpoint_options)

    if args.moa_holdout:
        all_results["moa_holdout"] = run_moa_holdout(cl_responses, feats, args.data_dir, checkpoint_options)

    with open(args.out_dir / "tahoe_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    log.info(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
