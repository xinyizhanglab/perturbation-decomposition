"""Within-cell-line 5-fold CV benchmark.

Evaluates Ridge, MLP, MORPH, and STATE on the same held-out perturbations.

Usage:
    python -m benchmarks.within_cl --model ridge --cell_line k562
    python -m benchmarks.within_cl --model all --cell_line all
"""
import argparse, json, logging, pickle
from pathlib import Path
import numpy as np

from evaluation.metrics import pref, std_r, save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

N_FOLDS = 5
MODELS = ["ridge", "mlp"]


def load_dataset(data_dir, cell_line, depmap_path):
    """Load pseudobulk deltas and DepMap embeddings."""
    with open(Path(data_dir) / f"{cell_line}.pkl", "rb") as f:
        data = pickle.load(f)
    with open(depmap_path, "rb") as f:
        depmap = pickle.load(f)

    # Filter to perturbations with DepMap
    deltas = data["deltas"]
    perts = sorted(p for p in deltas if p in depmap)

    V = np.array([depmap[p] for p in perts], dtype=np.float32)
    D = np.array([deltas[p] for p in perts], dtype=np.float32)

    log.info(f"  {cell_line}: {len(perts)} perts with DepMap (of {len(deltas)} total)")
    return perts, V, D


def run_cv(perts, V, D, model_name, out_dir, cell_line,
           n_folds=N_FOLDS, device="cuda:0", n_mlp_seeds=3):
    """Run 5-fold CV and save per-fold predictions."""
    from models.ridge import train_ridge, train_mlp

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(perts))
    folds = np.array_split(perm, n_folds)

    all_prefs = []

    for fold in range(n_folds):
        te_idx = folds[fold]
        tr_idx = np.concatenate([folds[f] for f in range(n_folds) if f != fold])

        V_tr, V_te = V[tr_idx], V[te_idx]
        D_tr, D_te = D[tr_idx], D[te_idx]
        test_perts = [perts[i] for i in te_idx]

        if model_name == "ridge":
            predict_fn, T_train = train_ridge(V_tr, D_tr)
            yp = predict_fn(V_te)

        elif model_name == "mlp":
            # Average over seeds
            yp = np.zeros_like(D_te)
            for seed in range(n_mlp_seeds):
                predict_fn, T_train = train_mlp(V_tr, D_tr, seed=seed, device=device)
                yp += predict_fn(V_te) / n_mlp_seeds
            T_train = D_tr.mean(0)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Save fold predictions
        fold_dir = Path(out_dir) / cell_line / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_predictions(fold_dir / "predictions.npz",
                         test_perts, yp, D_te, T_train)

        pr = pref(D_te, yp, T_train)
        all_prefs.append(pr)
        log.info(f"    fold{fold}: pref={pr:.4f}, n={len(te_idx)}")

    log.info(f"  {model_name} {cell_line}: mean pref={np.mean(all_prefs):.4f} "
             f"± {np.std(all_prefs):.4f}")

    return {"mean": float(np.mean(all_prefs)),
            "std": float(np.std(all_prefs)),
            "per_fold": [float(p) for p in all_prefs]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ridge", choices=MODELS + ["all"])
    parser.add_argument("--cell_line", default="k562",
                        choices=["k562", "rpe1", "hepg2", "jurkat", "all"])
    parser.add_argument("--data_dir", default="./data/processed")
    parser.add_argument("--depmap", default="./data/processed/depmap_embeddings.pkl")
    parser.add_argument("--out_dir", default="./results/within_cl")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    cell_lines = ["k562", "rpe1", "hepg2", "jurkat"] if args.cell_line == "all" else [args.cell_line]

    results = {}
    for cl in cell_lines:
        perts, V, D = load_dataset(args.data_dir, cl, args.depmap)
        for model in models:
            out = Path(args.out_dir) / model
            out.mkdir(parents=True, exist_ok=True)
            res = run_cv(perts, V, D, model, out, cl, device=args.device)
            results[f"{cl}_{model}"] = res

    # Save summary
    summary_path = Path(args.out_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
