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
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_RESIDUAL = "#149E9A"
C_CONSERVED = "#6F5CC4"
FEATURE_NAMES = [
    "DepMap PCA50",
    "Delta columns",
    "Co-perturbation",
    "Gene-gene corr",
    "Control expr",
    "DepMap + ctrl expr",
]
PLS_COMPONENTS = [5, 10, 20, 30, 50]
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
n_feats = len(FEATURE_NAMES)
width = 0.75 / n_feats
offsets = np.linspace(-(n_feats - 1) / 2 * width, (n_feats - 1) / 2 * width, n_feats)

feat_colors = ["#149E9A", "#8A94A6", "#8A94A6", "#8A94A6", "#8A94A6", "#6F5CC4"]

for fi, fn in enumerate(FEATURE_NAMES):
    vals = [feature_results[fn].get(cl, 0) for cl in CLS]
    ax.bar(x + offsets[fi], vals, width * 0.9, color=feat_colors[fi], edgecolor="white",
           linewidth=0.5, label=fn, zorder=2)

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

ax.legend(frameon=False, fontsize=7, loc="center left",
          bbox_to_anchor=(1.02, 0.5), labelspacing=0.5)

plt.tight_layout(rect=[0, 0, 0.72, 1])
fig.savefig(OUT / "s2_3a_feature_ablation.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s2_3a_feature_ablation.pdf", dpi=300, bbox_inches="tight")
plt.close()
fig, ax = plt.subplots(figsize=(5.0, 3.2))

x = np.arange(len(CLS))

for cl in CLS:
    ci = CLS.index(cl)
    pls_vals = [pls_results[nc][cl] for nc in PLS_COMPONENTS]
    ax.plot(range(len(PLS_COMPONENTS)), pls_vals, color=CL_COLORS[cl],
            linewidth=1.5, marker="o", markersize=4, markeredgecolor="white",
            markeredgewidth=0.3, label=CL_LABELS[cl], zorder=2)

    # Ridge baseline as horizontal dashed line
    ax.axhline(y=ridge_pca50[cl], color=CL_COLORS[cl], linewidth=1.0,
               linestyle="--", alpha=0.5, zorder=1)

ax.set_xticks(range(len(PLS_COMPONENTS)))
ax.set_xticklabels([str(nc) for nc in PLS_COMPONENTS], fontsize=9, color=C_TEXT)
ax.set_xlabel("PLS components", fontsize=10, color=C_TEXT)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Custom legend: lines = PLS, dashed = Ridge PCA50
from matplotlib.lines import Line2D
legend_cl = [Line2D([0], [0], color=CL_COLORS[cl], linewidth=1.5, label=CL_LABELS[cl])
             for cl in CLS]
legend_method = [Line2D([0], [0], color=C_TEXT, linewidth=1.0, linestyle="--",
                        alpha=0.5, label="Ridge PCA50")]
ax.legend(handles=legend_cl + legend_method, frameon=False, fontsize=7.5,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.5)

plt.tight_layout(rect=[0, 0, 0.78, 1])
fig.savefig(OUT / "s2_3b_pls_comparison.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s2_3b_pls_comparison.pdf", dpi=300, bbox_inches="tight")
plt.close()
