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
C_CONSERVED = "#6F5CC4"
C_INTERACTION = "#D26A3A"
C_NOISE = "#C9CED6"
C_CTRL = "#8A94A6"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_ZERO = "#FFFFFF"
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
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.5))
comp_names = ["alpha", "beta", "gamma"]
comp_colors = [C_TEMPLATE, C_CONSERVED, C_INTERACTION]
comp_labels = ["Cell line\ntemplate (α)", "Conserved\npert. (β)", "Interaction\n(γ)"]

for ai, (comp, col, lab) in enumerate(zip(comp_names, comp_colors, comp_labels)):
    ax = axes[ai]
    vals = np.array(split_results[comp])

    ax.hist(vals / ss_total * 100, bins=15, color=col, alpha=0.7, edgecolor="white")
    ax.axvline(x=uncorrected[comp] / ss_total * 100, color=C_TEXT, linewidth=1.2,
               linestyle="--", label="Uncorrected")
    ax.axvline(x=np.mean(vals) / ss_total * 100, color=col, linewidth=1.5,
               label="Signal (mean)")

    ax.set_xlabel("% of total variance", fontsize=9, color=C_TEXT)
    ax.set_title(lab, fontsize=10, color=C_TEXT, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if ai == 0:
        ax.set_ylabel("Count", fontsize=9, color=C_TEXT)
        ax.legend(frameon=False, fontsize=7)

plt.tight_layout()
fig.savefig(OUT / "s1e_split_half_noise.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s1e_split_half_noise.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s1e")


# ═══════════════════════════════════════════════
