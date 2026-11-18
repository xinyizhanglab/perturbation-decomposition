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
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
MODEL_STYLE = {
    "scGPT":  {"color": "#8A94A6", "marker": "o", "label": "scGPT"},
    "MORPH":  {"color": "#6F5CC4", "marker": "D", "label": "MORPH"},
    "Ridge":  {"color": "#149E9A", "marker": "s", "label": "Ridge (DepMap)"},
    "MLP":    {"color": "#0D7377", "marker": "^", "label": "MLP (DepMap)"},
}
FOLDCV_DIRS = {"Ridge": "Ridge_5foldCV", "MLP": "MLP_5foldCV", "MORPH": "MORPH_5foldCV"}
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
fig, ax = plt.subplots(figsize=(4.5, 3.2))

n_models = len(MODEL_STYLE)
x = np.arange(len(CLS))
offsets = np.linspace(-0.2, 0.2, n_models)

for mi, (model, style) in enumerate(MODEL_STYLE.items()):
    for ci, cl in enumerate(CLS):
        vals = results[model][cl]
        if len(vals) == 0:
            continue

        mean_v = np.mean(vals)

        # Error bars
        if len(vals) > 1:
            std_v = np.std(vals)
            ax.errorbar(x[ci] + offsets[mi], mean_v, yerr=std_v,
                        fmt="none", ecolor=style["color"], elinewidth=1.0,
                        capsize=2, capthick=0.8, zorder=2)

        # Dot
        ax.scatter(x[ci] + offsets[mi], mean_v, s=40, marker=style["marker"],
                   color=style["color"], edgecolor="white", linewidth=0.5,
                   zorder=3, label=style["label"] if ci == 0 else None)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Cell line", fontsize=10, color=C_TEXT)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Deduplicated legend outside right
handles, labels = ax.get_legend_handles_labels()
seen = set()
unique_h, unique_l = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l)
        unique_h.append(h)
        unique_l.append(l)

ax.legend(unique_h, unique_l, frameon=False, fontsize=8,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.8)

plt.tight_layout(rect=[0, 0, 0.78, 1])
fig.savefig(OUT / "fig2a_within_cl_pref.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2a_within_cl_pref.pdf", dpi=300, bbox_inches="tight")
plt.close()
