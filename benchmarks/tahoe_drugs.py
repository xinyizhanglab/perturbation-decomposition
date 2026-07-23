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
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from evaluation.metrics import pearson as corr, pref

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

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


def run_compound_cv(cl_responses, feats, out_dir):
    """5-fold CV with compound priors (Morgan, CC)."""
    log.info("\n=== Compound-prior benchmark (5-fold CV) ===")
    results = {}

    for tgt in CLS_TAHOE:
        for feat_name, feat_dict in feats.items():
            obs = [{"drug": dc[0], "feat": feat_dict[dc[0]], "tgt": cl_responses[tgt][dc]}
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

            if fold_prefs:
                results[f"{tgt}__{feat_name}"] = {
                    "mean": float(np.mean(fold_prefs)),
                    "std": float(np.std(fold_prefs)),
                }
                log.info(f"  {tgt} {feat_name}: {np.mean(fold_prefs):.4f} ± {np.std(fold_prefs):.4f}")

    return results


def run_source_transfer(cl_responses, out_dir, ranks=(10, 50, 100)):
    """Source-response transfer with SVD rank sweep."""
    log.info("\n=== Source-response transfer ===")
    results = {}

    for tgt in CLS_TAHOE:
        srcs = [c for c in CLS_TAHOE if c != tgt]
        obs = []
        for dc in sorted(cl_responses[tgt].keys()):
            src_resps = [cl_responses[s][dc] for s in srcs if dc in cl_responses[s]]
            if src_resps:
                obs.append({"drug": dc[0], "dc": dc,
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

            results[f"{tgt}__Source_k{k}"] = {
                "mean": float(np.mean(fold_prefs)),
                "std": float(np.std(fold_prefs)),
            }

        log.info(f"  {tgt}: Direct={results[f'{tgt}__Direct']['mean']:.4f}  " +
                 "  ".join(f"k={k}:{results[f'{tgt}__Source_k{k}']['mean']:.3f}" for k in ranks))

    return results


def run_moa_holdout(cl_responses, feats, data_dir):
    """Leave-one-MoA-out evaluation."""
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
                obs.append({"drug": drug, "moa": moa,
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

        results[f"{tgt}__Source_MoA"] = float(np.mean(moa_prefs_source)) if moa_prefs_source else 0
        results[f"{tgt}__Morgan_MoA"] = float(np.mean(moa_prefs_morgan)) if moa_prefs_morgan else 0
        log.info(f"  {tgt}: Source MoA-out={results[f'{tgt}__Source_MoA']:.4f}  "
                 f"Morgan MoA-out={results[f'{tgt}__Morgan_MoA']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/tahoe"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/tahoe"))
    parser.add_argument("--features", default="all", choices=["morgan", "cc", "source", "all"])
    parser.add_argument("--rank_sweep", action="store_true")
    parser.add_argument("--moa_holdout", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pdata, cl_responses, feats = load_tahoe(args.data_dir)
    log.info(f"  Cell lines: {list(cl_responses.keys())}")
    log.info(f"  Total observations: {sum(len(v) for v in cl_responses.values())}")

    all_results = {}

    if args.features in ("morgan", "cc", "all"):
        all_results["compound_cv"] = run_compound_cv(cl_responses, feats, args.out_dir)

    if args.features in ("source", "all"):
        ranks = (1, 2, 5, 10, 20, 50, 100, 150) if args.rank_sweep else (10, 50)
        all_results["source_transfer"] = run_source_transfer(cl_responses, args.out_dir, ranks)

    if args.moa_holdout:
        all_results["moa_holdout"] = run_moa_holdout(cl_responses, feats, args.data_dir)

    with open(args.out_dir / "tahoe_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    log.info(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
