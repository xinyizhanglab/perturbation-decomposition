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
C_IDENTITY = "#8A94A6"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
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
