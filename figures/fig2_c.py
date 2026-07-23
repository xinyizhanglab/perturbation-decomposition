"""
Figure 2c: Response-aligned projection rotates DepMap into response geometry.
Paired dot plot: raw DepMap vs Ridge-projected DepMap similarity alignment
with template-removed residual response.

For each CL:
  1. Train Ridge (DepMap PCA50 → delta)
  2. Raw DepMap cosine sim → Spearman ρ with residual sim
  3. Projected DepMap (W @ v) cosine sim → Spearman ρ with residual sim

Run: python3 fig2_ridge_projection.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import anndata as ad
import pickle, json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
MORPH_BASE = Path("/mnt/storage01/home/amolina/agop-perturbation/morph_pipeline")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure2"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

C_RAW = "#8A94A6"
C_PROJ = "#149E9A"
C_LINE = "#C9CED6"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

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

# ── Compute ──
results = {}

for cl in CLS:
    print(f"\n  {cl.upper()}:")
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)
    si = json.load(open(BENCH / "data" / cl / "split_info.json"))

    # All perts with DepMap
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
    n = len(perts)

    # Template removal
    T = D.mean(0)
    Tn = T / (np.linalg.norm(T) + 1e-30)
    alpha = D @ Tn
    R = D - np.outer(alpha, Tn)

    # Train Ridge on training split
    tr_set = set(si["train_perts"])
    tr_idx = [i for i, p in enumerate(perts) if p in tr_set]
    V_tr = V[tr_idx]
    D_tr = D[tr_idx]

    pca = PCA(50).fit(V_tr)
    V_pca = pca.transform(V)
    V_tr_pca = pca.transform(V_tr)

    ridge = Ridge(alpha=10).fit(V_tr_pca, D_tr)
    W = ridge.coef_  # (G, 50)

    # Project all DepMap embeddings through Ridge weights
    V_projected = V_pca @ W.T  # (n, G) — DepMap in response space

    # Pairwise similarities
    triu = np.triu_indices(n, k=1)

    # Raw DepMap
    raw_sim = cosine_sim_matrix(V)[triu]

    # Projected DepMap
    proj_sim = cosine_sim_matrix(V_projected)[triu]

    # Residual response
    resid_sim = cosine_sim_matrix(R)[triu]

    # Spearman
    rho_raw, p_raw = spearmanr(raw_sim, resid_sim)
    rho_proj, p_proj = spearmanr(proj_sim, resid_sim)

    results[cl] = {
        "rho_raw": float(rho_raw),
        "p_raw": float(p_raw),
        "rho_proj": float(rho_proj),
        "p_proj": float(p_proj),
        "n_perts": n,
    }
    print(f"    Raw DepMap:       ρ = {rho_raw:.4f}")
    print(f"    Ridge-projected:  ρ = {rho_proj:.4f}")
    print(f"    Δρ = {rho_proj - rho_raw:.4f}")

# ── Plot ──
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

fig, ax = plt.subplots(figsize=(3.8, 3.2))

x = np.arange(len(CLS))
y_raw = [results[cl]["rho_raw"] for cl in CLS]
y_proj = [results[cl]["rho_proj"] for cl in CLS]

# Connecting lines
for i in range(len(CLS)):
    ax.plot([x[i], x[i]], [y_raw[i], y_proj[i]],
            color=C_LINE, linewidth=1.5, zorder=1)

# Dots
ax.scatter(x, y_raw, s=55, color=C_RAW, edgecolor="white", linewidth=0.8,
           zorder=2, label="Raw DepMap")
ax.scatter(x, y_proj, s=55, color=C_PROJ, edgecolor="white", linewidth=0.8,
           zorder=2, label="Response-aligned projection")

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Cell line", fontsize=10, color=C_TEXT)
ax.set_ylabel("Spearman ρ with residual similarity", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, fontsize=8, loc="center left",
          bbox_to_anchor=(1.02, 0.5), labelspacing=0.6)

plt.tight_layout(rect=[0, 0, 0.68, 1])
fig.savefig(OUT / "fig2c_ridge_projection.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2c_ridge_projection.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print ──
print(f"\n{'='*60}")
print("Raw vs Ridge-projected DepMap alignment with residual")
print(f"{'='*60}")
print(f"{'CL':<8} {'Raw ρ':>10} {'Projected ρ':>14} {'Δρ':>8}")
print("-" * 45)
for cl in CLS:
    r = results[cl]
    print(f"{CL_LABELS[cl]:<8} {r['rho_raw']:>10.4f} {r['rho_proj']:>14.4f} "
          f"{r['rho_proj'] - r['rho_raw']:>8.4f}")

with open(OUT / "fig2c_ridge_projection_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT}")