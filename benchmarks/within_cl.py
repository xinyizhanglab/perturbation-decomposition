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


def load_dataset(data_dir, cell_line, depmap_path, return_metadata=False):
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
    if return_metadata:
        names = data.get("hvg_names", data.get("var_names"))
        if names is None or len(names) != D.shape[1] or len(set(names)) != len(names):
            raise ValueError("Processed data need unique ordered hvg_names matching response columns")
        return perts, V, D, np.asarray(names, dtype=str)
    return perts, V, D


def run_cv(perts, V, D, model_name, out_dir, cell_line,
           n_folds=N_FOLDS, device="cuda:0", n_mlp_seeds=3,
           feature_names=None, descriptor_axis_id=None, pca_seed=None):
    """Run 5-fold CV and save per-fold predictions."""
    from models.ridge import train_ridge, train_mlp

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(perts))
    folds = np.array_split(perm, n_folds)

    all_prefs = []

    for fold in range(n_folds):
        fitted = []
        te_idx = folds[fold]
        tr_idx = np.concatenate([folds[f] for f in range(n_folds) if f != fold])

        V_tr, V_te = V[tr_idx], V[te_idx]
        D_tr, D_te = D[tr_idx], D[te_idx]
        test_perts = [perts[i] for i in te_idx]

        if model_name == "ridge":
            predict_fn, T_train = train_ridge(V_tr, D_tr, pca_seed=pca_seed)
            yp = predict_fn(V_te)
            fitted.append(predict_fn)

        elif model_name == "mlp":
            # Average over seeds
            yp = np.zeros_like(D_te)
            for seed in range(n_mlp_seeds):
                predict_fn, T_train = train_mlp(V_tr, D_tr, seed=seed, device=device, pca_seed=pca_seed)
                yp += predict_fn(V_te) / n_mlp_seeds
                fitted.append(predict_fn)
            T_train = D_tr.mean(0)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Save fold predictions
        fold_dir = Path(out_dir) / cell_line / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        from models.checkpoints import save_fitted, save_ensemble, save_predictions_with_axes
        save_predictions_with_axes(fold_dir / "predictions.npz", test_perts, yp, D_te,
                                   T_train, feature_names)
        if feature_names is not None and descriptor_axis_id is not None:
            metadata = {"experiment": "within_cl", "cell_line": cell_line,
                        "fold": fold, "split_seed": 42,
                        "train_perts": [perts[i] for i in tr_idx], "test_perts": test_perts}
            if model_name == "ridge":
                save_fitted(fitted[0], fold_dir / "checkpoint", feature_names, descriptor_axis_id, metadata)
            else:
                save_ensemble(fitted, fold_dir / "checkpoint", feature_names, descriptor_axis_id, metadata)

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
    parser.add_argument("--data_dir", default="./data/processed/genetic")
    parser.add_argument("--depmap", default="./data/processed/genetic/depmap_embeddings.pkl")
    parser.add_argument("--out_dir", default="./results/within_cl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_mlp_seeds", type=int, default=3)
    parser.add_argument("--pca_seed", type=int, help="Optional fixed PCA seed; omitted preserves the original behavior")
    parser.add_argument("--dry_run", action="store_true", help="Print inputs/settings without training")
    args = parser.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    cell_lines = ["k562", "rpe1", "hepg2", "jurkat"] if args.cell_line == "all" else [args.cell_line]
    if args.n_mlp_seeds < 1:
        parser.error("--n_mlp_seeds must be positive")
    from benchmarks._runtime import prepare_run
    try:
        if not prepare_run(args, [Path(args.data_dir) / f"{cl}.pkl" for cl in cell_lines] + [args.depmap]):
            return
    except (OSError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    results = {}
    from reproduction.io import sha256
    descriptor_axis_id = "sha256:" + sha256(args.depmap)
    for cl in cell_lines:
        perts, V, D, feature_names = load_dataset(args.data_dir, cl, args.depmap, return_metadata=True)
        for model in models:
            out = Path(args.out_dir) / model
            out.mkdir(parents=True, exist_ok=True)
            res = run_cv(perts, V, D, model, out, cl, device=args.device,
                         n_mlp_seeds=args.n_mlp_seeds, feature_names=feature_names,
                         descriptor_axis_id=descriptor_axis_id, pca_seed=args.pca_seed)
            results[f"{cl}_{model}"] = res

    # Save summary
    summary_path = Path(args.out_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
