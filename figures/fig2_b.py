"""
Figure 2b: Raw co-essentiality is not a response neighbor graph.
Paired bars: Spearman ρ between DepMap pairwise cosine and response pairwise
similarity, for full delta and template-removed residual.

Run: python3 fig2_depmap_pairwise.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
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

C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
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
    """Pairwise cosine similarity from rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    X_normed = X / norms
    return X_normed @ X_normed.T

# ── Compute correlations from data ──
results = {}

for cl in CLS:
    print(f"\n  {cl.upper()}:")
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)

    # Perturbation deltas for perts with DepMap embeddings
    perts = []
    deltas = []
    depmap_vecs = []
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5:
            emb = get_emb(pn)
            if emb is not None:
                perts.append(pn)
                deltas.append(X[genes == pn].mean(0) - ctrl_mean)
                depmap_vecs.append(emb)

    D = np.array(deltas)
    V = np.array(depmap_vecs)
    n = len(perts)
    print(f"    {n} perturbations with DepMap embeddings")

    # Template removal
    T = D.mean(0)
    Tn = T / (np.linalg.norm(T) + 1e-30)
    alpha = D @ Tn
    R = D - np.outer(alpha, Tn)

    # Pairwise similarities
    depmap_sim = cosine_sim_matrix(V)
    delta_sim = cosine_sim_matrix(D)
    resid_sim = cosine_sim_matrix(R)

    # Extract upper triangle (exclude diagonal)
    triu_idx = np.triu_indices(n, k=1)
    dep_flat = depmap_sim[triu_idx]
    delta_flat = delta_sim[triu_idx]
    resid_flat = resid_sim[triu_idx]

    # Spearman correlations
    rho_full, p_full = spearmanr(dep_flat, delta_flat)
    rho_resid, p_resid = spearmanr(dep_flat, resid_flat)

    results[cl] = {
        "rho_full": float(rho_full),
        "p_full": float(p_full),
        "rho_resid": float(rho_resid),
        "p_resid": float(p_resid),
        "n_perts": n,
        "n_pairs": len(dep_flat),
    }
    print(f"    Full delta:  ρ = {rho_full:.4f} (p = {p_full:.2e})")
    print(f"    Residual:    ρ = {rho_resid:.4f} (p = {p_resid:.2e})")

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

fig, ax = plt.subplots(figsize=(3.8, 4.0))

x = np.arange(len(CLS))
width = 0.32

vals_full = [results[cl]["rho_full"] for cl in CLS]
vals_resid = [results[cl]["rho_resid"] for cl in CLS]

ax.bar(x - width / 2, vals_full, width, color=C_TEMPLATE, edgecolor="white",
       linewidth=0.5, label="Full response", zorder=2)
ax.bar(x + width / 2, vals_resid, width, color=C_RESIDUAL, edgecolor="white",
       linewidth=0.5, label="Template-removed residual", zorder=2)

ax.axhline(y=0, color=C_TEXT, linewidth=0.6, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Cell line", fontsize=10, color=C_TEXT)
ax.set_ylabel("Spearman ρ with response similarity", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, fontsize=8, loc="center left",
          bbox_to_anchor=(1.02, 0.5), labelspacing=0.6)

plt.tight_layout(rect=[0, 0, 0.72, 1])
fig.savefig(OUT / "fig2b_depmap_pairwise.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2b_depmap_pairwise.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print summary ──
print(f"\n{'='*70}")
print("DepMap pairwise cosine vs response similarity (Spearman ρ)")
print(f"{'='*70}")
print(f"{'CL':<8} {'Full delta':>12} {'Residual':>12} {'n_perts':>10} {'n_pairs':>10}")
print("-" * 55)
for cl in CLS:
    r = results[cl]
    print(f"{CL_LABELS[cl]:<8} {r['rho_full']:>12.4f} {r['rho_resid']:>12.4f} "
          f"{r['n_perts']:>10} {r['n_pairs']:>10}")

with open(OUT / "fig2b_depmap_pairwise_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT}")