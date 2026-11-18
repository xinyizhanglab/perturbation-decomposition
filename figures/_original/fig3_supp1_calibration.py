"""Draw a supplementary panel from saved numerical inputs."""
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
CALIB_COUNTS = [1, 5, 10, 25, 50, 100, 150]
N_REPS = 3

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)

for tgt in CLS:
    valid_counts = calibration[tgt]["valid_counts"]
    target_only_means = calibration[tgt]["target_only_means"]
    pooled_means = calibration[tgt]["pooled_means"]
    within_pref = calibration[tgt]["within_pref"]

    # Plot target-only (left)
    means_to = [m[0] for m in target_only_means]
    stds_to = [m[1] for m in target_only_means]
    ax_l.plot(valid_counts, means_to, color=CL_COLORS[tgt], linewidth=1.5,
              marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.3,
              label=CL_LABELS[tgt], zorder=2)
    ax_l.fill_between(valid_counts,
                      [m - s for m, s in zip(means_to, stds_to)],
                      [m + s for m, s in zip(means_to, stds_to)],
                      color=CL_COLORS[tgt], alpha=0.1)

    # Plot pooled (right)
    means_po = [m[0] for m in pooled_means]
    stds_po = [m[1] for m in pooled_means]
    ax_r.plot(valid_counts, means_po, color=CL_COLORS[tgt], linewidth=1.5,
              marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.3,
              label=CL_LABELS[tgt], zorder=2)
    ax_r.fill_between(valid_counts,
                      [m - s for m, s in zip(means_po, stds_po)],
                      [m + s for m, s in zip(means_po, stds_po)],
                      color=CL_COLORS[tgt], alpha=0.1)

    # Within-CL baseline
    ax_l.axhline(y=within_pref, color=CL_COLORS[tgt], linewidth=0.7, linestyle=":", alpha=0.4)
    ax_r.axhline(y=within_pref, color=CL_COLORS[tgt], linewidth=0.7, linestyle=":", alpha=0.4)

for ax, title in [(ax_l, "Target-only calibration"), (ax_r, "Source + target pooled")]:
    ax.set_xlabel("Target perturbations used", fontsize=10)
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax_l.set_ylabel("Perturbed-reference Pearson", fontsize=10)
ax_l.legend(frameon=False, fontsize=7.5, loc="lower right")

plt.tight_layout()
fig.savefig(OUT / "s3c_saturation_curves.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s3c_saturation_curves.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s3c")


# ═══════════════════════════════════════════════
