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
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}
C_RAW = "#8A94A6"
C_PROJ = "#149E9A"
C_LINE = "#C9CED6"
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
