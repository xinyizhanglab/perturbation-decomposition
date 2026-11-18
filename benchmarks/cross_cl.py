"""Cross-cell-line zero-shot transfer benchmark.

Trains on source CL(s), predicts target CL perturbations.
Evaluates Ridge, MLP, MORPH, STATE.

Usage:
    python -m benchmarks.cross_cl --model ridge --target k562 --n_sources 3
    python -m benchmarks.cross_cl --model all --target all
"""
import argparse, json, logging, pickle
from pathlib import Path
import numpy as np

from evaluation.metrics import pref, std_r, save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
PAIRED = {"k562": "rpe1", "rpe1": "k562", "hepg2": "jurkat", "jurkat": "hepg2"}


def load_depmap(path):
    with open(path, "rb") as f:
        emb = pickle.load(f)
    def get(p):
        for k in [p, p + "+ctrl"]:
            if k in emb:
                return emb[k].astype(np.float32)
        return None
    return get


def load_cl_data(data_dir, get_emb):
    cl_data = {}
    for cl in CLS:
        pkl = Path(data_dir) / f"{cl}.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        deltas = {p: d for p, d in data["deltas"].items() if get_emb(p) is not None}
        cl_data[cl] = deltas
        log.info(f"  {cl}: {len(deltas)} perts")
    return cl_data


def run_transfer(cl_data, get_emb, target, sources, model_name,
                 out_dir, device="cuda:0", n_mlp_seeds=3,
                 feature_names=None, descriptor_axis_id=None, pca_seed=None):
    """Run zero-shot transfer from source(s) to target."""
    from models.ridge import train_ridge, train_mlp
    # Source training data
    sV, sD, membership = [], [], []
    for src in sources:
        for pn, d in cl_data[src].items():
            e = get_emb(pn)
            if e is not None:
                sV.append(e); sD.append(d)
                membership.append({"cell_line": src, "perturbation": pn})
    sV = np.array(sV, dtype=np.float32)
    sD = np.array(sD, dtype=np.float32)

    # Target test data
    tgt_perts = sorted(cl_data[target].keys())
    tV = np.array([get_emb(p) for p in tgt_perts], dtype=np.float32)
    tD = np.array([cl_data[target][p] for p in tgt_perts], dtype=np.float32)
    T_tgt = tD.mean(0)

    fitted = []
    if model_name == "ridge":
        predict_fn, _ = train_ridge(sV, sD, pca_seed=pca_seed)
        yp = predict_fn(tV)
        fitted.append(predict_fn)
    elif model_name == "mlp":
        yp = np.zeros_like(tD)
        for seed in range(n_mlp_seeds):
            predict_fn, _ = train_mlp(sV, sD, seed=seed, device=device, pca_seed=pca_seed)
            yp += predict_fn(tV) / n_mlp_seeds
            fitted.append(predict_fn)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pr = pref(tD, yp, T_tgt)
    sr = std_r(tD, yp)

    src_label = "+".join(sources)
    log.info(f"  {src_label}→{target} {model_name}: pref={pr:.4f}, std_r={sr:.4f}")

    # Save
    out_path = Path(out_dir) / model_name / f"{src_label}_to_{target}"
    out_path.mkdir(parents=True, exist_ok=True)
    from models.checkpoints import save_fitted, save_ensemble, save_predictions_with_axes
    save_predictions_with_axes(out_path / "predictions.npz", tgt_perts, yp, tD, T_tgt, feature_names)
    if feature_names is not None and descriptor_axis_id is not None:
        metadata = {"experiment": "cross_cl", "target": target, "sources": list(sources),
                    "membership": membership, "test_perts": tgt_perts}
        if model_name == "ridge":
            save_fitted(fitted[0], out_path / "checkpoint", feature_names, descriptor_axis_id, metadata)
        else:
            save_ensemble(fitted, out_path / "checkpoint", feature_names, descriptor_axis_id, metadata)

    return {"pref": pr, "std_r": sr, "n_perts": len(tgt_perts),
            "source": src_label, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ridge", choices=["ridge", "mlp", "all"])
    parser.add_argument("--target", default="k562",
                        choices=CLS + ["all"])
    parser.add_argument("--n_sources", type=int, default=3, choices=[1, 3],
                        help="1 = paired source, 3 = all other CLs")
    parser.add_argument("--data_dir", default="./data/processed/genetic")
    parser.add_argument("--depmap", default="./data/processed/genetic/depmap_embeddings.pkl")
    parser.add_argument("--out_dir", default="./results/cross_cl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n_mlp_seeds", type=int, default=3)
    parser.add_argument("--pca_seed", type=int, help="Optional fixed PCA seed")
    parser.add_argument("--dry_run", action="store_true", help="Print inputs/settings without training")
    args = parser.parse_args()
    if args.n_mlp_seeds < 1:
        parser.error("--n_mlp_seeds must be positive")
    from benchmarks._runtime import prepare_run
    try:
        if not prepare_run(args, [Path(args.data_dir) / f"{cl}.pkl" for cl in CLS] + [args.depmap]):
            return
    except (OSError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    get_emb = load_depmap(args.depmap)
    cl_data = load_cl_data(args.data_dir, get_emb)
    from reproduction.io import sha256
    descriptor_axis_id = "sha256:" + sha256(args.depmap)
    feature_names = None
    for cell in CLS:
        with (Path(args.data_dir) / f"{cell}.pkl").open("rb") as stream:
            dataset = pickle.load(stream)
        names = dataset.get("hvg_names", dataset.get("var_names"))
        if names is None or len(set(names)) != len(names):
            raise ValueError("Processed data need unique ordered gene names")
        names = np.asarray(names, dtype=str)
        if names.shape != (len(next(iter(cl_data[cell].values()))),):
            raise ValueError("Gene names do not match response width")
        if feature_names is not None and not np.array_equal(feature_names, names):
            raise ValueError("Cross-cell-line gene identities/order differ")
        feature_names = names

    models = ["ridge", "mlp"] if args.model == "all" else [args.model]
    targets = CLS if args.target == "all" else [args.target]

    results = {}
    for target in targets:
        if args.n_sources == 1:
            sources_list = [[PAIRED[target]]]
        else:
            sources_list = [[c for c in CLS if c != target]]

        for sources in sources_list:
            for model in models:
                res = run_transfer(cl_data, get_emb, target, sources, model,
                                   args.out_dir, device=args.device, n_mlp_seeds=args.n_mlp_seeds,
                                   feature_names=feature_names, descriptor_axis_id=descriptor_axis_id,
                                   pca_seed=args.pca_seed)
                key = f"{'+'.join(sources)}_to_{target}_{model}"
                results[key] = res

    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
