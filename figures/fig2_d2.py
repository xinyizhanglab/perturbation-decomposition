"""
Figure 2d: Predictive information is distributed across DepMap axes.

Line plot:
- Ridge and MLP perturbed-reference Pearson as a function of DepMap PCA rank.
- Raw unreduced DepMap performance shown as a separated reference column.
- MORPH 5-fold CV shown as a final reference column.

Run:
    python3 fig2_rank_ablation.py
"""

import json
import pickle
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────────

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
MORPH_BASE = Path("/mnt/storage01/home/amolina/agop-perturbation/morph_pipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure2"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]

CL_LABELS = {
    "k562": "K562",
    "rpe1": "RPE1",
    "hepg2": "HepG2",
    "jurkat": "Jurkat",
}

CL_COLORS = {
    "k562": "#4A90D9",
    "rpe1": "#D9534F",
    "hepg2": "#5CB85C",
    "jurkat": "#F0AD4E",
}

DS = {
    "k562": "replogle_k562_essential",
    "rpe1": "replogle_rpe1_essential",
    "hepg2": "nadig_hepg2",
    "jurkat": "nadig_jurkat",
}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

RANKS = [1, 5, 10, 20, 50, 100, 200]

N_MLP_SEEDS = 1
MLP_EPOCHS = 400
MLP_BATCH_SIZE = 64
MLP_LR = 5e-4
MLP_WEIGHT_DECAY = 1e-5

dev = "cuda:0" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def corr(a, b):
    """Pearson correlation with safe handling of near-constant vectors."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def perturbed_reference_pearson(y_true, y_pred, reference):
    """Mean per-perturbation Pearson after subtracting training perturbed centroid."""
    return float(np.mean([
        corr(y_true[i] - reference, y_pred[i] - reference)
        for i in range(len(y_true))
    ]))


class MLP(nn.Module):
    def __init__(self, d_in, n_genes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_genes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(V_tr, D_tr, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(V_tr.shape[1], D_tr.shape[1]).to(dev).float()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MLP_LR,
        weight_decay=MLP_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MLP_EPOCHS,
    )

    Vt = torch.tensor(V_tr, dtype=torch.float32, device=dev)
    Dt = torch.tensor(D_tr, dtype=torch.float32, device=dev)

    n = len(V_tr)
    model.train()

    for _ in range(MLP_EPOCHS):
        perm = np.random.permutation(n)

        for start in range(0, n, MLP_BATCH_SIZE):
            idx = perm[start:start + MLP_BATCH_SIZE]
            loss = nn.MSELoss()(model(Vt[idx]), Dt[idx])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Load DepMap embeddings
# ─────────────────────────────────────────────────────────────────────────────

emb_dict = {}

for ds in DS.values():
    emb_path = MORPH_BASE / f"outputs/{ds}/pert_embeddings.pkl"
    if emb_path.exists():
        with open(emb_path, "rb") as f:
            emb_dict.update(pickle.load(f))
    else:
        print(f"[warning] Missing embedding file: {emb_path}")


def get_emb(perturbation):
    for key in [perturbation + "+ctrl", perturbation]:
        if key in emb_dict:
            return emb_dict[key].astype(np.float32)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Load MORPH 5-fold CV results
# ─────────────────────────────────────────────────────────────────────────────

morph_pref = {}

for cl in CLS:
    result_path = BENCH / f"runs/MORPH_5foldCV/{DS[cl]}/fold_results.json"
    if result_path.exists():
        with open(result_path, "r") as f:
            fold_results = json.load(f)
        morph_pref[cl] = float(np.mean([r["pref"] for r in fold_results]))
    else:
        print(f"[warning] Missing MORPH CV file: {result_path}")

print("\nMORPH 5-fold CV perturbed-reference Pearson:")
for cl in CLS:
    if cl in morph_pref:
        print(f"  {CL_LABELS[cl]}: {morph_pref[cl]:.4f}")
    else:
        print(f"  {CL_LABELS[cl]}: N/A")


# ─────────────────────────────────────────────────────────────────────────────
# Rank ablation
# ─────────────────────────────────────────────────────────────────────────────

ridge_results = {}
mlp_results = {}

for cl in CLS:
    print(f"\n{CL_LABELS[cl]}")

    adata_path = BENCH / "data" / cl / "replogle.h5ad"
    split_path = BENCH / "data" / cl / "split_info.json"

    a = ad.read_h5ad(adata_path)
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values

    ctrl_mask = genes == "non-targeting"
    ctrl_mean = X[ctrl_mask].mean(axis=0)

    with open(split_path, "r") as f:
        split_info = json.load(f)

    deltas = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        mask = genes == pn
        if mask.sum() >= 5 and get_emb(pn) is not None:
            deltas[pn] = X[mask].mean(axis=0) - ctrl_mean

    train_perts = [p for p in split_info["train_perts"] if p in deltas]
    test_perts = [p for p in split_info["test_perts"] if p in deltas]

    V_tr_raw = np.array([get_emb(p) for p in train_perts]).astype(np.float32)
    D_tr = np.array([deltas[p] for p in train_perts]).astype(np.float32)

    V_te_raw = np.array([get_emb(p) for p in test_perts]).astype(np.float32)
    D_te = np.array([deltas[p] for p in test_perts]).astype(np.float32)

    T_tr = D_tr.mean(axis=0)

    n_train = V_tr_raw.shape[0]
    d_full = V_tr_raw.shape[1]
    max_rank = min(max(RANKS), n_train - 1, d_full)

    pca = PCA(n_components=max_rank).fit(V_tr_raw)
    V_tr_pca = pca.transform(V_tr_raw).astype(np.float32)
    V_te_pca = pca.transform(V_te_raw).astype(np.float32)

    ridge_results[cl] = {}
    mlp_results[cl] = {}

    for rank in RANKS:
        rank_use = min(rank, max_rank)

        V_tr = V_tr_pca[:, :rank_use]
        V_te = V_te_pca[:, :rank_use]

        # Ridge on PCA rank
        ridge = Ridge(alpha=10)
        ridge.fit(V_tr, D_tr)
        yp_ridge = ridge.predict(V_te)
        ridge_pref = perturbed_reference_pearson(D_te, yp_ridge, T_tr)
        ridge_results[cl][rank] = float(ridge_pref)

        # MLP on PCA rank
        seed_scores = []
        for seed in range(N_MLP_SEEDS):
            model = train_mlp(V_tr, D_tr, seed=seed)
            with torch.no_grad():
                yp_mlp = model(
                    torch.tensor(V_te, dtype=torch.float32, device=dev)
                ).cpu().numpy()

            score = perturbed_reference_pearson(D_te, yp_mlp, T_tr)
            seed_scores.append(float(score))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mlp_results[cl][rank] = {
            "mean": float(np.mean(seed_scores)),
            "std": float(np.std(seed_scores)),
            "values": seed_scores,
        }

        print(
            f"  rank={rank:>4} "
            f"(used {rank_use:>3}): "
            f"Ridge={ridge_pref:.4f}  "
            f"MLP={np.mean(seed_scores):.4f}"
        )

    # Raw unreduced DepMap Ridge
    ridge_raw = Ridge(alpha=10)
    ridge_raw.fit(V_tr_raw, D_tr)
    yp_ridge_raw = ridge_raw.predict(V_te_raw)
    ridge_raw_pref = perturbed_reference_pearson(D_te, yp_ridge_raw, T_tr)
    ridge_results[cl]["raw"] = float(ridge_raw_pref)

    # Raw unreduced DepMap MLP
    raw_seed_scores = []
    for seed in range(N_MLP_SEEDS):
        model = train_mlp(V_tr_raw, D_tr, seed=seed)
        with torch.no_grad():
            yp_mlp_raw = model(
                torch.tensor(V_te_raw, dtype=torch.float32, device=dev)
            ).cpu().numpy()

        score = perturbed_reference_pearson(D_te, yp_mlp_raw, T_tr)
        raw_seed_scores.append(float(score))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mlp_results[cl]["raw"] = {
        "mean": float(np.mean(raw_seed_scores)),
        "std": float(np.std(raw_seed_scores)),
        "values": raw_seed_scores,
    }

    print(
        f"  raw ({d_full}): "
        f"Ridge={ridge_raw_pref:.4f}  "
        f"MLP={np.mean(raw_seed_scores):.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "text.color": C_TEXT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(4.5, 3.15))

rank_labels = [str(r) for r in RANKS]
x_rank = np.arange(len(rank_labels))
x_raw = len(rank_labels) + 0.35
x_morph = len(rank_labels) + 1.15

all_y_values = []

for cl in CLS:
    color = CL_COLORS[cl]

    y_ridge = np.array([ridge_results[cl][r] for r in RANKS])
    y_mlp = np.array([mlp_results[cl][r]["mean"] for r in RANKS])

    all_y_values.extend(y_ridge.tolist())
    all_y_values.extend(y_mlp.tolist())
    all_y_values.append(ridge_results[cl]["raw"])
    all_y_values.append(mlp_results[cl]["raw"]["mean"])

    if cl in morph_pref:
        all_y_values.append(morph_pref[cl])

    # Ridge PCA-rank curve
    ax.plot(
        x_rank,
        y_ridge,
        color=color,
        linewidth=1.7,
        marker="o",
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=3,
    )

    # MLP PCA-rank curve
    ax.plot(
        x_rank,
        y_mlp,
        color=color,
        linewidth=1.35,
        linestyle=(0, (3, 2)),
        marker="^",
        markersize=4.3,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=3,
        alpha=0.95,
    )

    # Raw unreduced DepMap reference points
    ax.scatter(
        x_raw,
        ridge_results[cl]["raw"],
        color=color,
        marker="o",
        s=28,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )

    ax.scatter(
        x_raw,
        mlp_results[cl]["raw"]["mean"],
        color=color,
        marker="^",
        s=32,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )

    # MORPH reference marker
    if cl in morph_pref:
        ax.scatter(
            x_morph,
            morph_pref[cl],
            color=color,
            marker="D",
            s=35,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )

# Separators for raw and MORPH reference columns
ax.axvline(len(rank_labels) - 0.45, color=C_GRID, linewidth=0.8, zorder=0)
ax.axvline(x_morph - 0.45, color=C_GRID, linewidth=0.8, zorder=0)
ax.axhline(0, color=C_TEXT, linewidth=0.6, alpha=0.35, zorder=0)

xticks = list(x_rank) + [x_raw, x_morph]
xticklabels = rank_labels + ["Ridge/MLP", "MORPH"]

ax.set_xticks(xticks)
ax.set_xticklabels(
    xticklabels,
    fontsize=8,
    rotation=35,
    ha="right",
    rotation_mode="anchor",
)

ax.set_xlabel("DepMap PCA rank", fontsize=9.5)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=9.5)

# Stable but data-adaptive y-axis
y_min = min(all_y_values)
y_max = max(all_y_values)
bottom = 0.0 if y_min > -0.02 else np.floor((y_min - 0.03) / 0.05) * 0.05
top = np.ceil((y_max + 0.03) / 0.05) * 0.05
ax.set_ylim(bottom, top)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.55, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Optional internal title
ax.set_title("DepMap rank ablation", fontsize=9.5, pad=5)

# Legends
model_handles = [
    Line2D(
        [0], [0],
        color=C_TEXT,
        linewidth=1.7,
        marker="o",
        markersize=4.2,
        markeredgecolor="white",
        label="Ridge",
    ),
    Line2D(
        [0], [0],
        color=C_TEXT,
        linewidth=1.35,
        linestyle=(0, (3, 2)),
        marker="^",
        markersize=4.3,
        markeredgecolor="white",
        label="MLP",
    ),
    Line2D(
        [0], [0],
        color=C_TEXT,
        linestyle="None",
        marker="D",
        markersize=4.6,
        markeredgecolor="white",
        label="MORPH",
    ),
]

cell_handles = [
    Line2D(
        [0], [0],
        color=CL_COLORS[cl],
        linewidth=2.3,
        label=CL_LABELS[cl],
    )
    for cl in CLS
]

leg1 = ax.legend(
    handles=model_handles,
    frameon=False,
    fontsize=7.2,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    title="Model",
    title_fontsize=7.5,
    handlelength=1.8,
    labelspacing=0.45,
)
ax.add_artist(leg1)

ax.legend(
    handles=cell_handles,
    frameon=False,
    fontsize=7.2,
    loc="upper left",
    bbox_to_anchor=(1.02, 0.55),
    title="Cell line",
    title_fontsize=7.5,
    handlelength=1.8,
    labelspacing=0.45,
)

plt.tight_layout(rect=[0, 0, 0.78, 1])

fig.savefig(OUT / "fig2d_rank_ablation.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2d_rank_ablation.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2d_rank_ablation.png", dpi=300, bbox_inches="tight")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Print and save results
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'=' * 80}")
print("DepMap rank ablation with raw DepMap and MORPH reference")
print(f"{'=' * 80}")

header = f"{'Rank':<8}"
for cl in CLS:
    header += f"  {CL_LABELS[cl] + ' Ridge':>13}  {CL_LABELS[cl] + ' MLP':>13}"
print(header)
print("-" * len(header))

for rank in RANKS:
    row = f"{str(rank):<8}"
    for cl in CLS:
        row += (
            f"  {ridge_results[cl][rank]:>13.4f}"
            f"  {mlp_results[cl][rank]['mean']:>13.4f}"
        )
    print(row)

row = f"{'Raw':<8}"
for cl in CLS:
    row += (
        f"  {ridge_results[cl]['raw']:>13.4f}"
        f"  {mlp_results[cl]['raw']['mean']:>13.4f}"
    )
print(row)

row = f"{'MORPH':<8}"
for cl in CLS:
    if cl in morph_pref:
        row += f"  {morph_pref[cl]:>13.4f}  {'':>13}"
    else:
        row += f"  {'N/A':>13}  {'':>13}"
print(row)

save_data = {
    "ranks": RANKS,
    "ridge": ridge_results,
    "mlp": mlp_results,
    "morph_5foldcv": {cl: float(v) for cl, v in morph_pref.items()},
    "settings": {
        "n_mlp_seeds": N_MLP_SEEDS,
        "mlp_epochs": MLP_EPOCHS,
        "mlp_batch_size": MLP_BATCH_SIZE,
        "mlp_lr": MLP_LR,
        "mlp_weight_decay": MLP_WEIGHT_DECAY,
        "ridge_alpha": 10,
    },
}

with open(OUT / "fig2d_rank_ablation_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved outputs to: {OUT}")