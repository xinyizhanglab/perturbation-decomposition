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
C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
K_VALUES = [5, 100]
PANEL_TITLES = ["First 5 control PCs", "First 100 control PCs"]
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
