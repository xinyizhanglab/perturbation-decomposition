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
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_LINE = "#C9CED6"
C_ZEROSHOT = "#8A94A6"
C_CALIBRATED = "#D26A3A"
C_BETA = "#149E9A"
C_GAMMA = "#D26A3A"
CALIB_FRAC = 0.3
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
all_vals = []
for cl in CLS:
    for s in ["zeroshot", "target30"]:
        all_vals.append(results[cl][s]["beta"])
        all_vals.append(results[cl][s]["gamma"])
ymin = min(all_vals) - 0.03
ymax = max(all_vals) + 0.03
# Snap to nice bounds
ymin = max(ymin, -0.10)
ymax = min(ymax, 0.50)

# ── Plot ──
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.0, 3.2), sharey=True)

x = np.arange(len(CLS))
offset = 0.15

for ci, cl in enumerate(CLS):
    zs_b = results[cl]["zeroshot"]["beta"]
    t30_b = results[cl]["target30"]["beta"]
    zs_g = results[cl]["zeroshot"]["gamma"]
    t30_g = results[cl]["target30"]["gamma"]

    # Left: β (teal tones)
    ax_left.plot([x[ci] - offset, x[ci] + offset], [zs_b, t30_b],
                 color=C_LINE, linewidth=1.5, zorder=1)
    ax_left.scatter(x[ci] - offset, zs_b, s=50, marker="o", color=C_ZEROSHOT,
                    edgecolor="white", linewidth=0.5, zorder=3,
                    label="Zero-shot" if ci == 0 else None)
    ax_left.scatter(x[ci] + offset, t30_b, s=50, marker="o", color=C_BETA,
                    edgecolor="white", linewidth=0.5, zorder=3,
                    label="Target perturbations 30%" if ci == 0 else None)

    # Right: γ (terracotta tones)
    ax_right.plot([x[ci] - offset, x[ci] + offset], [zs_g, t30_g],
                  color=C_LINE, linewidth=1.5, zorder=1)
    ax_right.scatter(x[ci] - offset, zs_g, s=50, marker="o", color=C_ZEROSHOT,
                     edgecolor="white", linewidth=0.5, zorder=3)
    ax_right.scatter(x[ci] + offset, t30_g, s=50, marker="o", color=C_GAMMA,
                     edgecolor="white", linewidth=0.5, zorder=3)

# Shared y-limits
ax_left.set_ylim(ymin, ymax)

# Left panel
ax_left.axhline(y=0, color=C_TEXT, linewidth=0.8, alpha=0.5, zorder=1)
ax_left.set_xticks(x)
ax_left.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                        rotation=45, ha="right", rotation_mode="anchor")
ax_left.set_ylabel("corr(prediction, component)", fontsize=10, color=C_TEXT)
ax_left.set_title("Conserved component β$_p$(g)", fontsize=10, color=C_TEXT, pad=6)
ax_left.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax_left.xaxis.grid(False)
ax_left.set_axisbelow(True)
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)

# Right panel
ax_right.axhline(y=0, color=C_TEXT, linewidth=0.8, alpha=0.5, zorder=1)
ax_right.set_xticks(x)
ax_right.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                         rotation=45, ha="right", rotation_mode="anchor")
ax_right.set_title("Target interaction γ(c,p,g)", fontsize=10, color=C_TEXT, pad=6)
ax_right.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax_right.xaxis.grid(False)
ax_right.set_axisbelow(True)
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

# Legend below
handles, labels_leg = ax_left.get_legend_handles_labels()
fig.legend(handles, labels_leg, frameon=False, fontsize=8, loc="lower center",
           bbox_to_anchor=(0.5, -0.05), ncol=2, columnspacing=2.0)

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT / "fig4d_gamma_reveal.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4d_gamma_reveal.pdf", dpi=300, bbox_inches="tight")
plt.close()
