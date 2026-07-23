"""
Figure 1: Template variance explained by control identity axis.

Identity axis: I_c(g) = mean_ctrl_c(g) - mean_ctrl_not_c(g)
Metric: cos²(T_c, I_c) = fraction of template direction along identity axis.

Run: python3 fig1_identity_variance.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import anndata as ad
import json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure1"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}

C_IDENTITY = "#8A94A6"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

# ── Load controls ──
print("Loading control expression...")
ctrl_means = {}
ctrl_cells_all = {}

for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X_full = np.array(a.X)
    genes_obs = a.obs["gene"].astype(str).values
    ctrl = X_full[genes_obs == "non-targeting"]
    ctrl_means[cl] = ctrl.mean(0)
    ctrl_cells_all[cl] = ctrl
    print(f"  {cl}: {ctrl.shape[0]} controls")

# ── Identity axes ──
identity_axes = {}
for cl in CLS:
    other_ctrl = np.concatenate([ctrl_cells_all[c] for c in CLS if c != cl])
    identity_axes[cl] = ctrl_means[cl] - other_ctrl.mean(0)

# ── Templates ──
templates = {}
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X_full = np.array(a.X)
    genes_obs = a.obs["gene"].astype(str).values
    deltas = []
    for pn in sorted(set(genes_obs) - {"non-targeting"}):
        if (genes_obs == pn).sum() >= 5:
            deltas.append(X_full[genes_obs == pn].mean(0) - ctrl_means[cl])
    templates[cl] = np.array(deltas).mean(0)

# ── cos² ──
def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

cos2 = {}
for cl in CLS:
    c = cosine(templates[cl], identity_axes[cl])
    cos2[cl] = c ** 2
    print(f"  {cl}: cos={c:.4f}, cos²={cos2[cl]:.4f} ({cos2[cl]*100:.1f}%)")

# ── Sort by value ──
sorted_cls = sorted(CLS, key=lambda c: cos2[c], reverse=True)

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

fig, ax = plt.subplots(figsize=(3.0, 2.4))

y = np.arange(len(sorted_cls))
vals = [cos2[cl] * 100 for cl in sorted_cls]
labels = [CL_LABELS[cl] for cl in sorted_cls]

ax.barh(y, vals, height=0.55, color=C_IDENTITY, edgecolor="white", linewidth=0.5)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10, color=C_TEXT)
ax.set_xlabel("Template variance explained\nby identity axis (%)", fontsize=10, color=C_TEXT)
ax.set_xlim(0, max(vals) * 1.6)
ax.invert_yaxis()

ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Value labels
for i, v in enumerate(vals):
    ax.text(v + max(vals) * 0.04, i, f"{v:.1f}%", va="center", fontsize=9, color=C_TEXT)

plt.tight_layout()
fig.savefig(OUT / "fig1_identity_variance.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1_identity_variance.pdf", dpi=300, bbox_inches="tight")
plt.close()

# Save
results = {cl: {"cos": cosine(templates[cl], identity_axes[cl]),
                 "cos2": float(cos2[cl]),
                 "pct": float(cos2[cl] * 100)} for cl in CLS}
with open(OUT / "fig1_identity_variance_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT}")