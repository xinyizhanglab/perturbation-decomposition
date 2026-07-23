"""
Figure 1: Template removal equalizes residual dimensionality.
Paired dot / slope plot: full δ effective rank → residual effective rank.
Run: python3 fig1_erank.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import anndata as ad
from sklearn.decomposition import PCA

# ── Paths ──
BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure1"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}

# ── Colors ──
C_FULL = "#8A94A6"      # slate grey — full response
C_RESIDUAL = "#149E9A"   # teal — perturbation-specific residual
C_LINE = "#C9CED6"       # light grey — connecting line
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

def effective_rank(M):
    """Shannon-entropy effective rank of matrix M (rows = samples)."""
    n_comp = min(100, M.shape[0] - 1, M.shape[1])
    pca = PCA(n_components=n_comp).fit(M)
    ev = pca.explained_variance_
    ev = ev[ev > 1e-12]
    p = ev / ev.sum()
    return np.exp(-np.sum(p * np.log(p)))

# ── Compute effective ranks from data ──
erank_full = {}
erank_resid = {}

for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)

    deltas = []
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5:
            deltas.append(X[genes == pn].mean(0) - ctrl_mean)
    D = np.array(deltas)

    # Full response effective rank
    erank_full[cl] = effective_rank(D)

    # Template removal
    T = D.mean(0)
    Tn = T / (np.linalg.norm(T) + 1e-30)
    alpha = D @ Tn
    R = D - np.outer(alpha, Tn)

    # Residual effective rank
    erank_resid[cl] = effective_rank(R)

# ── Sort by full effective rank ──
sorted_cls = sorted(CLS, key=lambda c: erank_full[c])

# ── Plot ──
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 0.6,
    "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "text.color": C_TEXT,
})

fig, ax = plt.subplots(figsize=(3.5, 3.2))

x = np.arange(len(sorted_cls))
y_full = [erank_full[cl] for cl in sorted_cls]
y_resid = [erank_resid[cl] for cl in sorted_cls]
labels = [CL_LABELS[cl] for cl in sorted_cls]

# Connecting lines
for i in range(len(sorted_cls)):
    ax.plot([x[i], x[i]], [y_full[i], y_resid[i]],
            color=C_LINE, linewidth=1.5, zorder=1)

# Dots
ax.scatter(x, y_full, s=60, color=C_FULL, edgecolor="white", linewidth=0.8,
           zorder=2, label="Full response")
ax.scatter(x, y_resid, s=60, color=C_RESIDUAL, edgecolor="white", linewidth=0.8,
           zorder=2, label="After template removal")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, color=C_TEXT)
ax.set_ylabel("Effective rank", fontsize=11, color=C_TEXT)
ax.set_ylim(0, max(y_resid) * 1.2)

# Light grid on y only
ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.0),
          handletextpad=0.4, borderpad=0.2)

plt.tight_layout()
fig.savefig(OUT / "fig1_erank_slope.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1_erank_slope.pdf", dpi=300, bbox_inches="tight")
plt.close()

print("Saved to", OUT)
for cl in sorted_cls:
    print(f"  {CL_LABELS[cl]}: full={erank_full[cl]:.1f}, residual={erank_resid[cl]:.1f}")