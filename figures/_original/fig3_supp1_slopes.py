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
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 3.5))

# STATE slope plot
x_state = [0, 1]
for tgt in CLS:
    s_mean = np.mean(state_data[tgt]["source"])
    c_mean = np.mean(state_data[tgt]["calibrated"])
    ax_l.plot(x_state, [s_mean, c_mean], color=CL_COLORS[tgt], linewidth=1.5, zorder=1)
    ax_l.scatter(0, s_mean, s=45, color=CL_COLORS[tgt], edgecolor="white",
                 linewidth=0.5, zorder=3, label=CL_LABELS[tgt])
    ax_l.scatter(1, c_mean, s=45, color=CL_COLORS[tgt], edgecolor="white",
                 linewidth=0.5, zorder=3)
    # Error bars
    if len(state_data[tgt]["source"]) > 1:
        ax_l.errorbar(0, s_mean, yerr=np.std(state_data[tgt]["source"]),
                      fmt="none", ecolor=CL_COLORS[tgt], elinewidth=0.8, capsize=2, capthick=0.6)
    if len(state_data[tgt]["calibrated"]) > 1:
        ax_l.errorbar(1, c_mean, yerr=np.std(state_data[tgt]["calibrated"]),
                      fmt="none", ecolor=CL_COLORS[tgt], elinewidth=0.8, capsize=2, capthick=0.6)

ax_l.set_xticks(x_state)
ax_l.set_xticklabels(["Source only", "+30% target\nperturbations"], fontsize=9)
ax_l.set_ylabel("Perturbed-reference Pearson", fontsize=10)
ax_l.set_title("STATE", fontsize=11, color=C_TEXT, pad=6)
ax_l.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
ax_l.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax_l.xaxis.grid(False)
ax_l.set_axisbelow(True)
ax_l.spines["top"].set_visible(False); ax_l.spines["right"].set_visible(False)
ax_l.legend(frameon=False, fontsize=7.5, loc="upper left")

# MORPH slope plot
PAIR_COLORS = ["#4A90D9", "#D9534F", "#5CB85C", "#F0AD4E"]
x_morph = [0, 1, 2]
for pi, pk in enumerate(morph_data):
    vals = [morph_data[pk][s] for s in morph_settings]
    ax_r.plot(x_morph, vals, color=PAIR_COLORS[pi], linewidth=1.5, zorder=1)
    for xi, v in enumerate(vals):
        ax_r.scatter(xi, v, s=40, color=PAIR_COLORS[pi], edgecolor="white",
                     linewidth=0.5, zorder=3, label=pk if xi == 0 else None)

ax_r.set_xticks(x_morph)
ax_r.set_xticklabels(["Source\nonly", "Target\nctrls", "+30% target\nperturbations"], fontsize=9)
ax_r.set_ylabel("Perturbed-reference Pearson", fontsize=10)
ax_r.set_title("MORPH", fontsize=11, color=C_TEXT, pad=6)
ax_r.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
ax_r.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax_r.xaxis.grid(False)
ax_r.set_axisbelow(True)
ax_r.spines["top"].set_visible(False); ax_r.spines["right"].set_visible(False)
ax_r.legend(frameon=False, fontsize=7, loc="upper left")

plt.tight_layout()
fig.savefig(OUT / "s3b_slope_plots.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s3b_slope_plots.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s3b")


# ═══════════════════════════════════════════════
