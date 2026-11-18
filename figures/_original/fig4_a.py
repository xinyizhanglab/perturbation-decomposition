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
C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
N_MLP_SEEDS = 3
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
fig, ax = plt.subplots(figsize=(6.0, 3.5))

x = np.arange(len(CLS))
group_width = 0.75
# 6 items: Ridge-α, Ridge-ε, MLP-α, MLP-ε, MORPH-α, MORPH-ε
n_items = 6
item_width = group_width / n_items
offsets = np.linspace(-group_width / 2 + item_width / 2,
                       group_width / 2 - item_width / 2, n_items)

configs = [
    ("Ridge", "alpha", "s",  C_TEMPLATE, "Ridge: template (α)"),
    ("Ridge", "eps",   "s",  C_RESIDUAL, "Ridge: residual (ε)"),
    ("MLP",   "alpha", "^",  C_TEMPLATE, "MLP: template (α)"),
    ("MLP",   "eps",   "^",  C_RESIDUAL, "MLP: residual (ε)"),
    ("MORPH", "alpha", "D",  C_TEMPLATE, "MORPH: template (α)"),
    ("MORPH", "eps",   "D",  C_RESIDUAL, "MORPH: residual (ε)"),
]

for ci, (model, comp, marker, color, label) in enumerate(configs):
    for xi, cl in enumerate(CLS):
        vals = results[model][cl][comp]
        if not vals:
            continue
        mean_v = np.mean(vals)

        if len(vals) > 1:
            std_v = np.std(vals)
            ax.errorbar(x[xi] + offsets[ci], mean_v, yerr=std_v,
                        fmt="none", ecolor=color, elinewidth=0.8,
                        capsize=1.5, capthick=0.6, zorder=2, alpha=0.7)

        # Use lighter fill for MORPH to distinguish from Ridge/MLP
        if model == "MORPH":
            ax.scatter(x[xi] + offsets[ci], mean_v, s=35, marker=marker,
                       facecolors=color, edgecolor="white", linewidth=0.5,
                       zorder=3, alpha=0.8,
                       label=label if xi == 0 else None)
        else:
            ax.scatter(x[xi] + offsets[ci], mean_v, s=40, marker=marker,
                       color=color, edgecolor="white", linewidth=0.5,
                       zorder=3, label=label if xi == 0 else None)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Cell line", fontsize=10, color=C_TEXT)
ax.set_ylabel("Correlation with ground truth", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend outside
handles, labels_leg = ax.get_legend_handles_labels()
seen = set(); uh, ul = [], []
for h, l in zip(handles, labels_leg):
    if l not in seen:
        seen.add(l)
        uh.append(h)
        ul.append(l)
ax.legend(uh, ul, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.5)

plt.tight_layout(rect=[0, 0, 0.68, 1])
fig.savefig(OUT / "fig4a_within_decomp.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4a_within_decomp.pdf", dpi=300, bbox_inches="tight")
plt.close()
