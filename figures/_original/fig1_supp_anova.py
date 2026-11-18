"""Preserved supplementary plotting statements for saved numerical results."""
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
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))

# Left panel: orthogonality
ax = axes[0]
pairs = ["α · β", "α · γ", "β · γ"]
cross_vals = [cross_ab, cross_ag, cross_bg]
colors = [C_CONSERVED, C_INTERACTION, C_CTRL]

ax.barh(range(3), cross_vals, height=0.5, color=colors, edgecolor="white", linewidth=0.5)
ax.axvline(x=0, color=C_TEXT, linewidth=0.6)
ax.set_yticks(range(3))
ax.set_yticklabels(pairs, fontsize=10, color=C_TEXT)
ax.set_xlabel("Mean cross-component\ndot product", fontsize=9, color=C_TEXT)
ax.set_title("Orthogonality check", fontsize=11, color=C_TEXT, pad=6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, color=C_GRID, linewidth=0.5)
ax.set_axisbelow(True)

# Right panel: per-CL β/γ ratio
ax = axes[1]
x = np.arange(len(CLS))
width = 0.3

ax.bar(x - width / 2, beta_per_cl, width, color=C_CONSERVED, edgecolor="white",
       linewidth=0.5, label="β (conserved)", zorder=2)
ax.bar(x + width / 2, gamma_per_cl, width, color=C_INTERACTION, edgecolor="white",
       linewidth=0.5, label="γ (interaction)", zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel("Mean variance", fontsize=10, color=C_TEXT)
ax.set_title("Per-CL component variance", fontsize=11, color=C_TEXT, pad=6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(OUT / "s1f_anova_diagnostics.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s1f_anova_diagnostics.pdf", dpi=300, bbox_inches="tight")
plt.close()

print("    Saved s1f")
