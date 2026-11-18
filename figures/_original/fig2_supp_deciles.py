"""Draw a supplementary panel from saved numerical inputs."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}
C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_CTRL = "#8A94A6"
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
N_BINS = 10

fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), sharex=True, sharey=True)

for ci, cl in enumerate(CLS):
    ax = axes[ci // 2, ci % 2]
    bin_centers = curves[cl]["bin_centers"]
    full_means = curves[cl]["full_means"]
    resid_means = curves[cl]["resid_means"]

    ax.plot(bin_centers, full_means, color=C_TEMPLATE, linewidth=1.5,
            marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.3,
            label="Full response" if ci == 0 else None)
    ax.plot(bin_centers, resid_means, color=C_RESIDUAL, linewidth=1.5,
            marker="s", markersize=4, markeredgecolor="white", markeredgewidth=0.3,
            label="Residual" if ci == 0 else None)

    ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.3)
    ax.set_title(CL_LABELS[cl], fontsize=11, color=C_TEXT, pad=4)

    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[1, 0].set_xlabel("DepMap cosine similarity", fontsize=10, color=C_TEXT)
axes[1, 1].set_xlabel("DepMap cosine similarity", fontsize=10, color=C_TEXT)
axes[0, 0].set_ylabel("Mean response\nsimilarity", fontsize=10, color=C_TEXT)
axes[1, 0].set_ylabel("Mean response\nsimilarity", fontsize=10, color=C_TEXT)

axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")

plt.tight_layout()
fig.savefig(OUT / "s2a_decile_plots.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s2a_decile_plots.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s2a")


# ═══════════════════════════════════════════════
