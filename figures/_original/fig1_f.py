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
C_TEMPLATE = "#3B5BA9"
C_CONSERVED = "#6F5CC4"
C_INTERACTION = "#D26A3A"
C_NOISE = "#C9CED6"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
N_SPLITS = 50
plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT,
})
fig, ax = plt.subplots(figsize=(5.0, 2.2))

bar_height = 0.45
y_positions = [1, 0]
y_labels = ["Full response", "Template-removed\nresidual"]

# Full response: template | conserved | interaction | noise
full_vals = [template_pct, beta_pct, gamma_pct, noise_pct]
full_colors = [C_TEMPLATE, C_CONSERVED, C_INTERACTION, C_NOISE]
full_labels = [r"Cell-line template ($\mu + \alpha$)",
               r"Conserved perturbation ($\beta$)",
               r"Interaction ($\gamma$)",
               "Noise"]

# Residual: conserved | interaction | noise
resid_vals = [beta_resid_pct, gamma_resid_pct, noise_resid_pct]
resid_colors = [C_CONSERVED, C_INTERACTION, C_NOISE]

# Draw full response bar
left = 0
for val, col, lab in zip(full_vals, full_colors, full_labels):
    ax.barh(y_positions[0], val, left=left, height=bar_height,
            color=col, edgecolor="white", linewidth=0.5, label=lab, zorder=2)
    if val > 6:
        tc = "white" if col != C_NOISE else C_TEXT
        ax.text(left + val / 2, y_positions[0], f"{val:.1f}%",
                ha="center", va="center", fontsize=8, color=tc)
    left += val

# Draw residual bar
left = 0
for val, col in zip(resid_vals, resid_colors):
    ax.barh(y_positions[1], val, left=left, height=bar_height,
            color=col, edgecolor="white", linewidth=0.5, zorder=2)
    if val > 6:
        tc = "white" if col != C_NOISE else C_TEXT
        ax.text(left + val / 2, y_positions[1], f"{val:.1f}%",
                ha="center", va="center", fontsize=8, color=tc)
    left += val

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=10, color=C_TEXT)
ax.set_xlabel("Variance (%)", fontsize=10, color=C_TEXT)
ax.set_xlim(0, 102)

ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center",
           bbox_to_anchor=(0.5, -0.15), ncol=2, columnspacing=1.0)

plt.tight_layout(rect=[0, 0.16, 1, 1])
fig.savefig(OUT / "fig1f_anova_stacked.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1f_anova_stacked.pdf", dpi=300, bbox_inches="tight")
plt.close()
