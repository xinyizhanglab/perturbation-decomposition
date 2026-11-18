"""Original plotting statements, with input/output locations supplied by the launcher."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_DIAG = "#8A94A6"
with open(DATA) as f:
    results = json.load(f)

# ── Determine shared axis limits ──
all_r = []
for cl in CLS:
    for p in results[cl]["per_pert"]:
        all_r.append(p["r_source"])
        all_r.append(p["r_depmap"])
pad = 0.05
lo = min(all_r) - pad
hi = max(all_r) + pad
# Round to clean bounds
lo = max(lo, -0.5)
hi = min(hi, 1.0)

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

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

for ci, cl in enumerate(CLS):
    ax = axes[ci]
    res = results[cl]
    pp = res["per_pert"]

    x = np.array([p["r_source"] for p in pp])
    y = np.array([p["r_depmap"] for p in pp])

    # Points
    ax.scatter(x, y, s=10, alpha=0.35, color=CL_COLORS[cl], edgecolor="none", rasterized=True)

    # Diagonal
    ax.plot([lo, hi], [lo, hi], color=C_DIAG, linewidth=0.8, linestyle="--", alpha=0.6)

    # Zero lines
    ax.axhline(0, color=C_TEXT, linewidth=0.3, alpha=0.25)
    ax.axvline(0, color=C_TEXT, linewidth=0.3, alpha=0.25)

    # Limits
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    # Labels
    ax.set_title(CL_LABELS[cl], fontsize=11, fontweight="bold", color=C_TEXT, pad=8)
    ax.set_xlabel("Source-response transfer", fontsize=9, color=C_TEXT)
    if ci == 0:
        ax.set_ylabel("DepMap prediction", fontsize=10, color=C_TEXT)

    # Grid
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation block
    rho = res["complementarity"]
    frac = res["frac_depmap_wins"]
    ms = res["mean_source"]
    md = res["mean_depmap"]
    n = res["n"]

    ax.text(0.97, 0.03, f"\u03C1 = {rho:.2f}", transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right", color=C_TEXT)



plt.tight_layout(w_pad=1.5)
fig.savefig(OUT / "fig3_source_vs_depmap.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3_source_vs_depmap.pdf", dpi=300, bbox_inches="tight")
plt.close()
