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
MORPH_SOURCE = {"k562": "rpe1", "rpe1": "k562", "hepg2": "jurkat", "jurkat": "hepg2"}
C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
N_MLP_SEEDS = 3
MODEL_ORDER = ["Ridge 1→1", "Ridge 3→1", "MLP 1→1", "MLP 3→1", "MORPH", "STATE"]
MODEL_MARKERS = {
    "Ridge 1→1": "s", "Ridge 3→1": "s",
    "MLP 1→1": "^", "MLP 3→1": "^",
    "MORPH": "D", "STATE": "o",
}
plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT, "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(7.0, 3.8))

x = np.arange(len(CLS))
n_models = len(MODEL_ORDER)
group_width = 0.80
offsets = np.linspace(-group_width / 2, group_width / 2, n_models)

for mi, model in enumerate(MODEL_ORDER):
    marker = MODEL_MARKERS[model]
    for xi, cl in enumerate(CLS):
        a_vals = results[model][cl]["alpha"]
        e_vals = results[model][cl]["eps"]
        if not a_vals:
            continue

        a_mean = np.mean(a_vals)
        e_mean = np.mean(e_vals)
        xpos = x[xi] + offsets[mi]

        # Template dot
        if len(a_vals) > 1:
            ax.errorbar(xpos - 0.02, a_mean, yerr=np.std(a_vals),
                        fmt="none", ecolor=C_TEMPLATE, elinewidth=0.8,
                        capsize=1.5, capthick=0.6, zorder=2, alpha=0.6)
        ax.scatter(xpos - 0.02, a_mean, s=35, marker=marker,
                   color=C_TEMPLATE, edgecolor="white", linewidth=0.5, zorder=3)

        # Residual dot
        if len(e_vals) > 1:
            ax.errorbar(xpos + 0.02, e_mean, yerr=np.std(e_vals),
                        fmt="none", ecolor=C_RESIDUAL, elinewidth=0.8,
                        capsize=1.5, capthick=0.6, zorder=2, alpha=0.6)
        ax.scatter(xpos + 0.02, e_mean, s=35, marker=marker,
                   color=C_RESIDUAL, edgecolor="white", linewidth=0.5, zorder=3)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)

# Model labels along top
for mi, model in enumerate(MODEL_ORDER):
    # Place at mean x position
    ax.text(np.mean(x) + offsets[mi], ax.get_ylim()[1] if ax.get_ylim()[1] else 1.0,
            model, ha="center", va="bottom", fontsize=6.5, color=C_TEXT,
            rotation=45, rotation_mode="anchor")

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT)
ax.set_ylabel("Correlation with ground truth", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend: components + model markers
leg_comp = [
    Line2D([0], [0], marker="o", color=C_TEMPLATE, markersize=6,
           markeredgecolor="white", linewidth=0, label="Template (α)"),
    Line2D([0], [0], marker="o", color=C_RESIDUAL, markersize=6,
           markeredgecolor="white", linewidth=0, label="Residual (ε)"),
]
leg_model = [
    Line2D([0], [0], marker="s", color=C_TEXT, markersize=5,
           markeredgecolor="white", linewidth=0, label="Ridge"),
    Line2D([0], [0], marker="^", color=C_TEXT, markersize=5,
           markeredgecolor="white", linewidth=0, label="MLP"),
    Line2D([0], [0], marker="D", color=C_TEXT, markersize=5,
           markeredgecolor="white", linewidth=0, label="MORPH"),
    Line2D([0], [0], marker="o", color=C_TEXT, markersize=5,
           markeredgecolor="white", linewidth=0, label="STATE"),
]

ax.legend(handles=leg_comp + leg_model, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.6)

plt.tight_layout(rect=[0, 0, 0.80, 0.92])
fig.savefig(OUT / "fig4_crosscl_decomp.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4_crosscl_decomp.pdf", dpi=300, bbox_inches="tight")
plt.close()
