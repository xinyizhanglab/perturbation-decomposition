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
CL_LABELS = {
    "k562": "K562",
    "rpe1": "RPE1",
    "hepg2": "HepG2",
    "jurkat": "Jurkat",
}
CL_COLORS = {
    "k562": "#4A90D9",
    "rpe1": "#D9534F",
    "hepg2": "#5CB85C",
    "jurkat": "#F0AD4E",
}
DS = {
    "k562": "replogle_k562_essential",
    "rpe1": "replogle_rpe1_essential",
    "hepg2": "nadig_hepg2",
    "jurkat": "nadig_jurkat",
}
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
RANKS = [1, 5, 10, 20, 50, 100, 200]
N_MLP_SEEDS = 1
MLP_EPOCHS = 400
MLP_BATCH_SIZE = 64
MLP_LR = 5e-4
MLP_WEIGHT_DECAY = 1e-5
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "text.color": C_TEXT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(4.5, 3.15))

rank_labels = [str(r) for r in RANKS]
x_rank = np.arange(len(rank_labels))
x_raw = len(rank_labels) + 0.35
x_morph = len(rank_labels) + 1.15

all_y_values = []

for cl in CLS:
    color = CL_COLORS[cl]

    y_ridge = np.array([ridge_results[cl][r] for r in RANKS])
    y_mlp = np.array([mlp_results[cl][r]["mean"] for r in RANKS])

    all_y_values.extend(y_ridge.tolist())
    all_y_values.extend(y_mlp.tolist())
    all_y_values.append(ridge_results[cl]["raw"])
    all_y_values.append(mlp_results[cl]["raw"]["mean"])

    if cl in morph_pref:
        all_y_values.append(morph_pref[cl])

    # Ridge PCA-rank curve
    ax.plot(
        x_rank,
        y_ridge,
        color=color,
        linewidth=1.7,
        marker="o",
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=3,
    )

    # MLP PCA-rank curve
    ax.plot(
        x_rank,
        y_mlp,
        color=color,
        linewidth=1.35,
        linestyle=(0, (3, 2)),
        marker="^",
        markersize=4.3,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=3,
        alpha=0.95,
    )

    # Raw unreduced DepMap reference points
    ax.scatter(
        x_raw,
        ridge_results[cl]["raw"],
        color=color,
        marker="o",
        s=28,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )

    ax.scatter(
        x_raw,
        mlp_results[cl]["raw"]["mean"],
        color=color,
        marker="^",
        s=32,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )

    # MORPH reference marker
    if cl in morph_pref:
        ax.scatter(
            x_morph,
            morph_pref[cl],
            color=color,
            marker="D",
            s=35,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )

# Separators for raw and MORPH reference columns
ax.axvline(len(rank_labels) - 0.45, color=C_GRID, linewidth=0.8, zorder=0)
ax.axvline(x_morph - 0.45, color=C_GRID, linewidth=0.8, zorder=0)
ax.axhline(0, color=C_TEXT, linewidth=0.6, alpha=0.35, zorder=0)

xticks = list(x_rank) + [x_raw, x_morph]
xticklabels = rank_labels + ["Ridge/MLP", "MORPH"]

ax.set_xticks(xticks)
ax.set_xticklabels(
    xticklabels,
    fontsize=8,
    rotation=35,
    ha="right",
    rotation_mode="anchor",
)

ax.set_xlabel("DepMap PCA rank", fontsize=9.5)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=9.5)

# Stable but data-adaptive y-axis
y_min = min(all_y_values)
y_max = max(all_y_values)
bottom = 0.0 if y_min > -0.02 else np.floor((y_min - 0.03) / 0.05) * 0.05
top = np.ceil((y_max + 0.03) / 0.05) * 0.05
ax.set_ylim(bottom, top)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.55, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Optional internal title
ax.set_title("DepMap rank ablation", fontsize=9.5, pad=5)

# Legends
model_handles = [
    Line2D(
        [0], [0],
        color=C_TEXT,
        linewidth=1.7,
        marker="o",
        markersize=4.2,
        markeredgecolor="white",
        label="Ridge",
    ),
    Line2D(
        [0], [0],
        color=C_TEXT,
        linewidth=1.35,
        linestyle=(0, (3, 2)),
        marker="^",
        markersize=4.3,
        markeredgecolor="white",
        label="MLP",
    ),
    Line2D(
        [0], [0],
        color=C_TEXT,
        linestyle="None",
        marker="D",
        markersize=4.6,
        markeredgecolor="white",
        label="MORPH",
    ),
]

cell_handles = [
    Line2D(
        [0], [0],
        color=CL_COLORS[cl],
        linewidth=2.3,
        label=CL_LABELS[cl],
    )
    for cl in CLS
]

leg1 = ax.legend(
    handles=model_handles,
    frameon=False,
    fontsize=7.2,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    title="Model",
    title_fontsize=7.5,
    handlelength=1.8,
    labelspacing=0.45,
)
ax.add_artist(leg1)

ax.legend(
    handles=cell_handles,
    frameon=False,
    fontsize=7.2,
    loc="upper left",
    bbox_to_anchor=(1.02, 0.55),
    title="Cell line",
    title_fontsize=7.5,
    handlelength=1.8,
    labelspacing=0.45,
)

plt.tight_layout(rect=[0, 0, 0.78, 1])

fig.savefig(OUT / "fig2d_rank_ablation.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2d_rank_ablation.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2d_rank_ablation.png", dpi=300, bbox_inches="tight")
plt.close()
