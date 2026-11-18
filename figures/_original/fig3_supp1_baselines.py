"""Preserved supplementary plotting statements for saved numerical results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_LINE = "#C9CED6"
C_RIDGE = "#149E9A"
C_MLP = "#0D7377"
C_SOURCE = "#8A94A6"
C_FTCTRL = "#6F5CC4"
C_CALIB = "#D26A3A"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT,
})
CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)
settings = ["3→1 zero-shot", "Target-only 30%", "Within-CL full"]

for ax, model_name, title in [(ax_l, "Ridge", "Ridge (DepMap)"), (ax_r, "MLP", "MLP (DepMap)")]:
    x = np.arange(len(CLS))
    for ci, cl in enumerate(CLS):
        vals = [baseline_results[model_name][cl][s] for s in settings]
        ax.plot(range(len(settings)), vals, color=CL_COLORS[cl], linewidth=1.5,
                marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.5,
                label=CL_LABELS[cl] if model_name == "Ridge" else None, zorder=2)

    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(settings, fontsize=8, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax_l.set_ylabel("Perturbed-reference Pearson", fontsize=10)
ax_l.legend(frameon=False, fontsize=7.5, loc="upper left")

plt.tight_layout()
fig.savefig(OUT / "s3e_reference_baselines.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s3e_reference_baselines.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s3e")

