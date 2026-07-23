"""
Figure 1e: Template and residual in control expression space.
Two side-by-side panels: first 5 control PCs and first 100 control PCs.
Each shows fraction of template/residual variance captured.

Run: python3 fig1_ctrl_space.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import anndata as ad
import json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure1"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}

C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

K_VALUES = [5, 100]
PANEL_TITLES = ["First 5 control PCs", "First 100 control PCs"]

# ── Compute projections from data ──
results = {k: {"template": {}, "residual": {}} for k in K_VALUES}

for cl in CLS:
    print(f"\n  {cl.upper()}: loading...")
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_cells = X[genes == "non-targeting"]
    ctrl_mean = ctrl_cells.mean(0)

    # Control PCA
    ctrl_centered = ctrl_cells - ctrl_cells.mean(0)
    n_ctrl_pcs = min(100, ctrl_centered.shape[0] - 1)
    pca_ctrl = PCA(n_components=n_ctrl_pcs).fit(ctrl_centered)
    U_ctrl = pca_ctrl.components_  # (n_pcs, G)

    # Perturbation deltas
    deltas = []
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5:
            deltas.append(X[genes == pn].mean(0) - ctrl_mean)
    D = np.array(deltas)

    # Template
    T = D.mean(0)
    Tn = T / (np.linalg.norm(T) + 1e-30)
    T_norm2 = np.linalg.norm(T) ** 2

    # Residual
    alpha = D @ Tn
    D_resid = D - np.outer(alpha, Tn)
    n_perts = len(D)

    for k in K_VALUES:
        k_use = min(k, n_ctrl_pcs)
        U_k = U_ctrl[:k_use]  # (k, G)

        # Template projection
        T_in = U_k.T @ (U_k @ T)
        t_frac = np.linalg.norm(T_in) ** 2 / T_norm2 * 100

        # Residual projection (mean across perturbations)
        r_fracs = []
        for i in range(n_perts):
            e = D_resid[i]
            e_norm2 = np.linalg.norm(e) ** 2
            if e_norm2 < 1e-12:
                continue
            e_in = U_k.T @ (U_k @ e)
            r_fracs.append(np.linalg.norm(e_in) ** 2 / e_norm2 * 100)
        r_frac = np.mean(r_fracs)

        results[k]["template"][cl] = float(t_frac)
        results[k]["residual"][cl] = float(r_frac)
        print(f"    k={k}: template={t_frac:.1f}%, residual={r_frac:.1f}%")

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

fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0), sharey=True)

width = 0.32
x = np.arange(len(CLS))

for pi, (k, title) in enumerate(zip(K_VALUES, PANEL_TITLES)):
    ax = axes[pi]

    t_vals = [results[k]["template"][cl] for cl in CLS]
    r_vals = [results[k]["residual"][cl] for cl in CLS]

    ax.bar(x - width / 2, t_vals, width, color=C_TEMPLATE, edgecolor="white",
           linewidth=0.5, label="Template" if pi == 0 else None, zorder=2)
    ax.bar(x + width / 2, r_vals, width, color=C_RESIDUAL, edgecolor="white",
           linewidth=0.5, label="Residual" if pi == 0 else None, zorder=2)

    ax.axhline(y=50, color=C_GRID, linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9, color=C_TEXT,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=10, color=C_TEXT, pad=6)

    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Variance captured (%)", fontsize=10, color=C_TEXT)

# Shared legend outside
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, loc="center right",
           bbox_to_anchor=(1.12, 0.5), labelspacing=0.6)

plt.tight_layout(rect=[0, 0, 0.88, 1])
fig.savefig(OUT / "fig1e_ctrl_space.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1e_ctrl_space.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print ──
print(f"\n{'='*70}")
print("Variance captured by control PCs (%)")
print(f"{'='*70}")
for k in K_VALUES:
    print(f"\n  k={k} control PCs:")
    print(f"  {'CL':<8} {'Template':>10} {'Residual':>10}")
    print(f"  {'-'*30}")
    for cl in CLS:
        print(f"  {CL_LABELS[cl]:<8} {results[k]['template'][cl]:>9.1f}% {results[k]['residual'][cl]:>9.1f}%")

# Save
with open(OUT / "fig1e_ctrl_space_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT}")