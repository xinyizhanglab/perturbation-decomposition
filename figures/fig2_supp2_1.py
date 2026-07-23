"""
Supplementary Figure 2: Supporting analyses for DepMap response geometry.

S2a: Pairwise decile plots — DepMap similarity bins vs residual response similarity
S2b: Co-essentiality cluster coherence — within vs between cluster residual correlation
S2c: Full pairwise scatter — DepMap cosine vs response similarity

All saved to supplemetary_figure_2_2/

Run: python3 fig2_supp.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import anndata as ad
import pickle, json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
MORPH_BASE = Path("/mnt/storage01/home/amolina/agop-perturbation/morph_pipeline")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "supplemetary_figure_2_2"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_CTRL = "#8A94A6"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.linewidth": 0.6,
    "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "text.color": C_TEXT,
})

# ── Load DepMap embeddings ──
emb_dict = {}
for ds in DS.values():
    ep = MORPH_BASE / f"outputs/{ds}/pert_embeddings.pkl"
    if ep.exists():
        with open(ep, "rb") as f:
            emb_dict.update(pickle.load(f))

def get_emb(p):
    for k in [p + "+ctrl", p]:
        if k in emb_dict:
            return emb_dict[k].astype(np.float32)
    return None

def cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X_n = X / norms
    return X_n @ X_n.T

# ── Load all CL data once ──
print("Loading data...")
cl_data = {}
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)

    perts, deltas_list, depmap_list = [], [], []
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5:
            emb = get_emb(pn)
            if emb is not None:
                perts.append(pn)
                deltas_list.append(X[genes == pn].mean(0) - ctrl_mean)
                depmap_list.append(emb)

    D = np.array(deltas_list)
    V = np.array(depmap_list)
    T = D.mean(0)
    Tn = T / (np.linalg.norm(T) + 1e-30)
    alpha = D @ Tn
    R = D - np.outer(alpha, Tn)

    cl_data[cl] = {"D": D, "V": V, "R": R, "perts": perts, "n": len(perts)}
    print(f"  {cl}: {len(perts)} perturbations")


# ═══════════════════════════════════════════════
# S2a: Pairwise decile plots
# ═══════════════════════════════════════════════
print("\n  S2a: Pairwise decile plots...")

N_BINS = 10

fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), sharex=True, sharey=True)

for ci, cl in enumerate(CLS):
    ax = axes[ci // 2, ci % 2]
    V = cl_data[cl]["V"]
    D = cl_data[cl]["D"]
    R = cl_data[cl]["R"]
    n = cl_data[cl]["n"]

    dep_sim = cosine_sim_matrix(V)
    full_sim = cosine_sim_matrix(D)
    resid_sim = cosine_sim_matrix(R)

    triu = np.triu_indices(n, k=1)
    dep_flat = dep_sim[triu]
    full_flat = full_sim[triu]
    resid_flat = resid_sim[triu]

    # Bin by DepMap similarity deciles
    percentiles = np.percentile(dep_flat, np.linspace(0, 100, N_BINS + 1))
    bin_centers = []
    full_means = []
    resid_means = []

    for bi in range(N_BINS):
        lo, hi = percentiles[bi], percentiles[bi + 1]
        if bi == N_BINS - 1:
            mask = (dep_flat >= lo) & (dep_flat <= hi)
        else:
            mask = (dep_flat >= lo) & (dep_flat < hi)

        if mask.sum() == 0:
            continue

        bin_centers.append((lo + hi) / 2)
        full_means.append(np.mean(full_flat[mask]))
        resid_means.append(np.mean(resid_flat[mask]))

    ax.plot(bin_centers, full_means, color=C_TEMPLATE, linewidth=1.5,
            marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.3,
            label="Full response" if ci == 0 else None)
    ax.plot(bin_centers, resid_means, color=C_RESIDUAL, linewidth=1.5,
            marker="s", markersize=4, markeredgecolor="white", markeredgewidth=0.3,
            label="Residual" if ci == 0 else None)

    ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.3)
    ax.set_title(CL_LABELS[cl], fontsize=11, color=C_TEXT, pad=4)

    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[1, 0].set_xlabel("DepMap cosine similarity", fontsize=10, color=C_TEXT)
axes[1, 1].set_xlabel("DepMap cosine similarity", fontsize=10, color=C_TEXT)
axes[0, 0].set_ylabel("Mean response\nsimilarity", fontsize=10, color=C_TEXT)
axes[1, 0].set_ylabel("Mean response\nsimilarity", fontsize=10, color=C_TEXT)

axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")

plt.tight_layout()
fig.savefig(OUT / "s2a_decile_plots.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s2a_decile_plots.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s2a")


# ═══════════════════════════════════════════════
# S2b: Co-essentiality cluster coherence
# ═══════════════════════════════════════════════
print("\n  S2b: Co-essentiality cluster coherence...")

K_VALUES = [10, 20, 50, 100, 200]

fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), sharex=True, sharey=True)

for ci, cl in enumerate(CLS):
    ax = axes[ci // 2, ci % 2]
    V = cl_data[cl]["V"]
    R = cl_data[cl]["R"]
    n = cl_data[cl]["n"]

    resid_sim = cosine_sim_matrix(R)

    within_means = []
    between_means = []

    for k in K_VALUES:
        k_use = min(k, n - 1)
        km = KMeans(n_clusters=k_use, random_state=0, n_init=5).fit(V)
        labels = km.labels_

        within_corrs = []
        between_corrs = []

        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    within_corrs.append(resid_sim[i, j])
                else:
                    # Sample to avoid O(n^2) for between
                    if np.random.random() < min(1.0, 5000 / (n * (n - 1) / 2)):
                        between_corrs.append(resid_sim[i, j])

        within_means.append(np.mean(within_corrs) if within_corrs else 0)
        between_means.append(np.mean(between_corrs) if between_corrs else 0)

    ax.plot(range(len(K_VALUES)), within_means, color=C_RESIDUAL, linewidth=1.5,
            marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.5,
            label="Within cluster" if ci == 0 else None)
    ax.plot(range(len(K_VALUES)), between_means, color=C_CTRL, linewidth=1.5,
            marker="s", markersize=5, markeredgecolor="white", markeredgewidth=0.5,
            label="Between cluster" if ci == 0 else None)

    ax.set_title(CL_LABELS[cl], fontsize=11, color=C_TEXT, pad=4)
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=9, color=C_TEXT)

    ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.3)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[1, 0].set_xlabel("Number of DepMap clusters", fontsize=10, color=C_TEXT)
axes[1, 1].set_xlabel("Number of DepMap clusters", fontsize=10, color=C_TEXT)
axes[0, 0].set_ylabel("Mean residual\nsimilarity", fontsize=10, color=C_TEXT)
axes[1, 0].set_ylabel("Mean residual\nsimilarity", fontsize=10, color=C_TEXT)

axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(OUT / "s2b_cluster_coherence.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s2b_cluster_coherence.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s2b")


# ═══════════════════════════════════════════════
# S2c: Full pairwise scatter plots
# ═══════════════════════════════════════════════
print("\n  S2c: Full pairwise scatter plots...")

fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.5))

for ci, cl in enumerate(CLS):
    V = cl_data[cl]["V"]
    D = cl_data[cl]["D"]
    R = cl_data[cl]["R"]
    n = cl_data[cl]["n"]

    dep_sim = cosine_sim_matrix(V)
    full_sim = cosine_sim_matrix(D)
    resid_sim = cosine_sim_matrix(R)

    triu = np.triu_indices(n, k=1)
    dep_flat = dep_sim[triu]
    full_flat = full_sim[triu]
    resid_flat = resid_sim[triu]

    # Subsample for plotting (max 20k points)
    n_pairs = len(dep_flat)
    if n_pairs > 20000:
        idx = np.random.RandomState(0).choice(n_pairs, 20000, replace=False)
    else:
        idx = np.arange(n_pairs)

    # Top row: full delta
    ax = axes[0, ci]
    ax.scatter(dep_flat[idx], full_flat[idx], s=0.5, alpha=0.15, color=C_TEMPLATE,
               rasterized=True)
    rho, _ = spearmanr(dep_flat, full_flat)
    ax.set_title(f"{CL_LABELS[cl]} (ρ={rho:.3f})", fontsize=9, color=C_TEXT, pad=3)
    ax.axhline(y=0, color=C_TEXT, linewidth=0.4, alpha=0.3)
    ax.axvline(x=0, color=C_TEXT, linewidth=0.4, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ci == 0:
        ax.set_ylabel("Full response\nsimilarity", fontsize=9, color=C_TEXT)

    # Bottom row: residual
    ax = axes[1, ci]
    ax.scatter(dep_flat[idx], resid_flat[idx], s=0.5, alpha=0.15, color=C_RESIDUAL,
               rasterized=True)
    rho, _ = spearmanr(dep_flat, resid_flat)
    ax.set_title(f"ρ={rho:.3f}", fontsize=9, color=C_TEXT, pad=3)
    ax.set_xlabel("DepMap cosine", fontsize=9, color=C_TEXT)
    ax.axhline(y=0, color=C_TEXT, linewidth=0.4, alpha=0.3)
    ax.axvline(x=0, color=C_TEXT, linewidth=0.4, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ci == 0:
        ax.set_ylabel("Residual\nsimilarity", fontsize=9, color=C_TEXT)

plt.tight_layout()
fig.savefig(OUT / "s2c_pairwise_scatter.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s2c_pairwise_scatter.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s2c")

print(f"\nAll supplementary panels saved to {OUT}")