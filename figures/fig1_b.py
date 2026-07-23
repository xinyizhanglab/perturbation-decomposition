"""
Figure 1b: Template variance explained across cell lines.
Run: python3 fig1_b.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import anndata as ad

# ── Paths ──
BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure1"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}

# ── Colors ──
C_TEMPLATE = "#3B5BA9"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

# ── Compute template variance from data ──
template_var = {}
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

    T = D.mean(0)
    total_var = np.mean(np.sum(D ** 2, axis=1))
    template_component = np.mean(np.sum((np.outer(D @ T / (np.dot(T, T) + 1e-30), T)) ** 2, axis=1))
    template_var[cl] = template_component / total_var * 100

# ── Sort by value ──
sorted_cls = sorted(CLS, key=lambda c: template_var[c])

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

fig, ax = plt.subplots(figsize=(3.2, 3.0))

x = np.arange(len(sorted_cls))
vals = [template_var[cl] for cl in sorted_cls]
labels = [CL_LABELS[cl] for cl in sorted_cls]

bars = ax.bar(x, vals, width=0.6, color=C_TEMPLATE, edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, color=C_TEXT)
ax.set_ylabel("Template variance explained (%)", fontsize=11, color=C_TEXT)
ax.set_ylim(0, max(vals) * 1.15)

# Light grid on y only
ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "fig1b_template_variance.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1b_template_variance.pdf", dpi=300, bbox_inches="tight")
plt.close()

print("Saved to", OUT)
for cl in sorted_cls:
    print(f"  {CL_LABELS[cl]}: {template_var[cl]:.1f}%")


