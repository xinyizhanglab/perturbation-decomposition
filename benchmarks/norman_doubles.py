"""Norman double-gene combinatorial perturbation benchmark.

Predicts double-gene perturbation responses from DepMap pair features.
Stratifies by unseen-gene count (0/2, 1/2, 2/2).
Compares SALT, Ridge, MLP against MORPH.

Usage:
    python -m benchmarks.norman_doubles --data_dir data/processed/norman
    python -m benchmarks.norman_doubles --include_morph --morph_dir results/morph/norman
"""
import argparse, json, pickle, logging
from pathlib import Path
import numpy as np

from evaluation.metrics import pearson as corr, pref, save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

try:
    import torch, torch.nn as nn
    HAS_TORCH = True
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    dev = "cpu"


class MLP(nn.Module if HAS_TORCH else object):
    def __init__(self, d_in, G):
        if not HAS_TORCH:
            raise ImportError("Install MLP dependencies with uv sync --locked --extra gpu")
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, G))
    def forward(self, v): return self.net(v)


def train_mlp(V_tr, D_tr, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(V_tr.shape[1], D_tr.shape[1]).to(dev).float()
    opt = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
    Vt = torch.tensor(V_tr, dtype=torch.float32, device=dev)
    Dt = torch.tensor(D_tr, dtype=torch.float32, device=dev)
    n = len(V_tr); m.train()
    for ep in range(400):
        pm = np.random.permutation(n)
        for s in range(0, n, 64):
            idx = pm[s:s+64]
            loss = nn.MSELoss()(m(Vt[idx]), Dt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
    m.eval(); return m


def parse_condition(c):
    parts = c.split("+")
    if len(parts) == 2:
        g1, g2 = parts
        if g1 == "ctrl": return g2, None
        if g2 == "ctrl": return g1, None
        return g1, g2
    return c, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed/norman"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/norman"))
    parser.add_argument("--include_morph", action="store_true")
    parser.add_argument("--morph_dir", type=Path, default=None)
    parser.add_argument("--n_mlp_seeds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry_run", action="store_true", help="Print inputs/settings without training")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--gene_names", type=Path, help="Verified ordered output-gene JSON for checkpoint export")
    args = parser.parse_args()
    if args.save_checkpoints and not args.gene_names:
        parser.error("--save_checkpoints requires --gene_names (the legacy bundle omits this axis)")

    if args.device == "auto":
        args.device = "cuda:0" if HAS_TORCH and torch.cuda.is_available() else "cpu"
    global dev; dev = args.device

    from benchmarks._runtime import prepare_run
    try:
        if not prepare_run(args, [args.data_dir / name for name in
                                 ("norman.pkl", "split_info.json", "depmap_embeddings.pkl")]):
            return
    except (OSError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")

    # Load data
    log.info("=== Loading Norman data ===")
    with open(args.data_dir / "norman.pkl", "rb") as f:
        data = pickle.load(f)
    with open(args.data_dir / "split_info.json") as f:
        si = json.load(f)
    with open(args.data_dir / "depmap_embeddings.pkl", "rb") as f:
        depmap = pickle.load(f)

    deltas = data["deltas"]
    test_perts_raw = set(si["test_perts"])
    val_perts_raw = set(si.get("val_perts", []))

    # Train/test split
    train_conds = [c for c in deltas if c not in test_perts_raw and c not in val_perts_raw]
    T_train = np.mean([deltas[c] for c in train_conds], axis=0)

    # Singles and train genes
    train_singles = {}
    for c in train_conds:
        g1, g2 = parse_condition(c)
        if g2 is None:
            train_singles[g1] = deltas[c]
    train_single_genes = set(train_singles.keys())

    # Test doubles
    test_doubles = {}
    unseen_class = {}
    for c in deltas:
        if c in test_perts_raw:
            g1, g2 = parse_condition(c)
            if g2 is not None:
                test_doubles[c] = deltas[c]
                u = (0 if g1 in train_single_genes else 1) + (0 if g2 in train_single_genes else 1)
                unseen_class[c] = u

    def get_emb(g):
        return depmap.get(g, None)

    test_ok = {c: d for c, d in test_doubles.items()
               if get_emb(c.split("+")[0]) is not None and get_emb(c.split("+")[1]) is not None}
    test_list = sorted(test_ok.keys())

    log.info(f"  Train: {len(train_conds)} ({len(train_singles)} singles)")
    log.info(f"  Test doubles: {len(test_ok)} (of {len(test_doubles)})")
    for u in [0, 1, 2]:
        n = sum(1 for c in test_ok if unseen_class.get(c) == u)
        log.info(f"    {u}/2 unseen: {n}")

    # Train doubles with DepMap
    train_double_conds = [(c, deltas[c]) for c in train_conds
                          if "+" in c and "ctrl" not in c
                          and get_emb(c.split("+")[0]) is not None
                          and get_emb(c.split("+")[1]) is not None]

    D_tr = np.array([d for _, d in train_double_conds], dtype=np.float32)
    D_te = np.array([test_ok[c] for c in test_list], dtype=np.float32)

    V_tr = np.array([get_emb(c.split("+")[0]) + get_emb(c.split("+")[1])
                     for c, _ in train_double_conds], dtype=np.float32)
    V_te = np.array([get_emb(c.split("+")[0]) + get_emb(c.split("+")[1])
                     for c in test_list], dtype=np.float32)

    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    nc = min(50, V_tr.shape[0] - 1, V_tr.shape[1])
    pca = PCA(nc).fit(V_tr)

    models = {}
    checkpoint_genes = json.loads(args.gene_names.read_text()) if args.save_checkpoints else None
    if checkpoint_genes is not None:
        if len(checkpoint_genes) != D_tr.shape[1] or len(set(checkpoint_genes)) != len(checkpoint_genes):
            parser.error("Gene-name axis does not match Norman responses")
        from models.checkpoints import save_estimator, save_predictions_with_axes
        from reproduction.io import sha256
        descriptor_axis_id = "sha256:" + sha256(args.data_dir / "depmap_embeddings.pkl") + ":pair_sum"

    # SALT
    salt = {}
    for c in test_list:
        g1, g2 = c.split("+")
        if g1 in train_singles and g2 in train_singles:
            salt[c] = train_singles[g1] + train_singles[g2]
    models["SALT"] = salt

    # Ridge
    ridge = Ridge(alpha=10).fit(pca.transform(V_tr), D_tr)
    models["Ridge"] = {c: ridge.predict(pca.transform(V_te[i:i+1]))[0] for i, c in enumerate(test_list)}
    if args.save_checkpoints:
        save_estimator(pca, ridge, args.out_dir / "checkpoints/ridge", checkpoint_genes,
                       descriptor_axis_id, {"experiment": "norman_doubles", "feature_builder": "pair_sum",
                                            "test_perts": test_list})
        save_predictions_with_axes(args.out_dir / "ridge_predictions.npz", test_list,
            np.stack([models["Ridge"][c] for c in test_list]), D_te, T_train, checkpoint_genes)

    # MLP
    if HAS_TORCH:
        mlp_preds = {c: np.zeros(D_te.shape[1], dtype=np.float32) for c in test_list}
        checkpoint_members = []
        for seed in range(args.n_mlp_seeds):
            m = train_mlp(pca.transform(V_tr), D_tr, seed=seed)
            with torch.no_grad():
                yp = m(torch.tensor(pca.transform(V_te), dtype=torch.float32, device=dev)).cpu().numpy()
            for i, c in enumerate(test_list):
                mlp_preds[c] += yp[i] / args.n_mlp_seeds
            if args.save_checkpoints:
                member = args.out_dir / f"checkpoints/mlp/member{seed}"
                save_estimator(pca, m, member, checkpoint_genes, descriptor_axis_id,
                               {"experiment": "norman_doubles", "seed": seed, "feature_builder": "pair_sum"})
                checkpoint_members.append({"path": f"member{seed}", "model_sha256": sha256(member / "model.json")})
            del m
            if "cuda" in dev: torch.cuda.empty_cache()
        models["MLP"] = mlp_preds
        if args.save_checkpoints:
            from reproduction.io import write_json
            write_json(args.out_dir / "checkpoints/mlp/model.json", {
                "schema_version": 1, "kind": "ensemble", "members": checkpoint_members,
                "descriptor_axis_id": descriptor_axis_id, "feature_names": checkpoint_genes,
                "experiment": "norman_doubles", "feature_builder": "pair_sum"})
            save_predictions_with_axes(args.out_dir / "mlp_predictions.npz", test_list,
                np.stack([mlp_preds[c] for c in test_list]), D_te, T_train, checkpoint_genes)

    # MORPH (load precomputed predictions)
    if args.include_morph and args.morph_dir:
        morph_preds = {}
        n_seeds = 0
        for seed in range(5):
            f = args.morph_dir / f"seed{seed}" / "predictions.npz"
            if f.exists():
                d = np.load(f, allow_pickle=True)
                perts = list(d["test_perts"])
                yp = d["predictions"]
                for i, p in enumerate(perts):
                    if p in test_ok:
                        if p not in morph_preds: morph_preds[p] = []
                        morph_preds[p].append(yp[i])
                n_seeds += 1
        if morph_preds:
            models["MORPH"] = {c: np.mean(v, axis=0) for c, v in morph_preds.items()
                               if len(v) > 0}
            log.info(f"  MORPH: {len(models['MORPH'])} perts from {n_seeds} seeds")

    # Evaluate
    log.info(f"\n{'='*75}")
    log.info(f"{'Model':<25} {'Overall':>8} {'0/2':>8} {'1/2':>8} {'2/2':>8}")
    log.info("-" * 55)

    results = {}
    for name, preds in models.items():
        row = {}
        for label, u_val in [("Overall", None), ("0/2", 0), ("1/2", 1), ("2/2", 2)]:
            if u_val is None:
                conds = [c for c in test_list if c in preds]
            else:
                conds = [c for c in test_list if c in preds and unseen_class.get(c) == u_val]
            if conds:
                yt = np.array([test_ok[c] for c in conds])
                yp = np.array([preds[c] for c in conds])
                pr = pref(yt, yp, T_train)
                row[label] = f"{pr:.4f}"
            else:
                row[label] = "—"
        log.info(f"{name:<25} {row['Overall']:>8} {row['0/2']:>8} {row['1/2']:>8} {row['2/2']:>8}")
        results[name] = row

    with open(args.out_dir / "norman_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
