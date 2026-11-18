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
DU_FRAC = 0.3
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
fig, ax = plt.subplots(figsize=(4.0, 3.5))

for cl in CLS:
    x_val = target_props[cl]["erank_resid"]

    # Seen perts (circle)
    ax.scatter(x_val, pref_seen[cl], s=60, marker="o",
               color=CL_COLORS[cl], edgecolor="white", linewidth=0.8,
               zorder=3)

    # Double-unseen (triangle)
    ax.scatter(x_val, pref_du[cl], s=60, marker="^",
               color=CL_COLORS[cl], edgecolor="white", linewidth=0.8,
               zorder=3)

    # Connecting line
    ax.plot([x_val, x_val], [pref_seen[cl], pref_du[cl]],
            color=CL_COLORS[cl], linewidth=1.0, alpha=0.5, zorder=2)

    # Label
    ax.annotate(CL_LABELS[cl], (x_val, pref_seen[cl]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=C_TEXT, ha="left", va="bottom")

ax.set_xlabel("Residual effective rank", fontsize=10, color=C_TEXT)
ax.set_ylabel("MLP perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend for marker shapes
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_TEXT,
           markeredgecolor="white", markersize=7, label="3→1 (seen perts)"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor=C_TEXT,
           markeredgecolor="white", markersize=7, label="Double unseen"),
]
ax.legend(handles=legend_elements, frameon=False, fontsize=8,
          loc="lower right")

plt.tight_layout()
fig.savefig(OUT / "fig3d_target_landscape.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3d_target_landscape.pdf", dpi=300, bbox_inches="tight")
plt.close()
