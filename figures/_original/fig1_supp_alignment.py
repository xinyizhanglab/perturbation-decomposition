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
fig, ax = plt.subplots(figsize=(4.0, 2.8))

x = np.arange(len(CLS))
width = 0.32

ax.bar(x - width / 2, cos_delta_pc1, width, color=C_TEMPLATE, edgecolor="white",
       linewidth=0.5, label="Response PC1", zorder=2)
ax.bar(x + width / 2, cos_ctrl_pc1, width, color=C_CTRL, edgecolor="white",
       linewidth=0.5, label="Control PC1", zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel("|cos(template, PC1)|", fontsize=10, color=C_TEXT)
ax.set_ylim(0, 1.05)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, fontsize=8, loc="center left",
          bbox_to_anchor=(1.02, 0.5))

plt.tight_layout(rect=[0, 0, 0.82, 1])
fig.savefig(OUT / "s1b_template_pc1_alignment.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s1b_template_pc1_alignment.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s1b")
print(f"    Values: delta_PC1={cos_delta_pc1}, ctrl_PC1={cos_ctrl_pc1}")

