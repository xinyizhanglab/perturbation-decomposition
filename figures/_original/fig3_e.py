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
BASIS_RANK = 20
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
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.5, 3.2))

cl_labels = [CL_LABELS[cl] for cl in CLS]

# Colormaps
cmap_mlp = LinearSegmentedColormap.from_list("white_teal", ["#F3F4F6", "#0D7377"])
cmap_basis = LinearSegmentedColormap.from_list("white_grey", ["#F3F4F6", "#8A94A6"])

# Mask diagonal
mlp_plot = np.copy(mlp_matrix)
overlap_plot = np.copy(overlap_matrix)

# Left: MLP transfer
im_left = ax_left.imshow(mlp_plot, cmap=cmap_mlp, aspect="auto",
                          vmin=0, vmax=np.nanmax(mlp_plot) * 1.05)

for si in range(n_cl):
    for ti in range(n_cl):
        if si == ti:
            ax_left.add_patch(plt.Rectangle((ti - 0.5, si - 0.5), 1, 1,
                              facecolor="#F0F0F0", edgecolor="none"))
            ax_left.text(ti, si, "—", ha="center", va="center",
                        fontsize=9, color="#CCCCCC")
        else:
            v = mlp_plot[si, ti]
            tc = "white" if v > np.nanmax(mlp_plot) * 0.55 else C_TEXT
            ax_left.text(ti, si, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=tc)

ax_left.set_xticks(range(n_cl))
ax_left.set_xticklabels(cl_labels, fontsize=9, color=C_TEXT,
                        rotation=45, ha="right", rotation_mode="anchor")
ax_left.set_yticks(range(n_cl))
ax_left.set_yticklabels(cl_labels, fontsize=9, color=C_TEXT)
ax_left.set_xlabel("Target", fontsize=10, color=C_TEXT)
ax_left.set_ylabel("Source", fontsize=10, color=C_TEXT)
ax_left.set_title("MLP transfer (pref)", fontsize=11, color=C_TEXT, pad=6)

for spine in ax_left.spines.values():
    spine.set_visible(False)
ax_left.tick_params(length=0)

cb_left = fig.colorbar(im_left, ax=ax_left, shrink=0.8, pad=0.02)
cb_left.ax.tick_params(labelsize=7, colors=C_TEXT)
cb_left.outline.set_edgecolor(C_TEXT)
cb_left.outline.set_linewidth(0.5)

# Right: Basis coverage
im_right = ax_right.imshow(overlap_plot, cmap=cmap_basis, aspect="auto",
                            vmin=0, vmax=np.nanmax(overlap_plot) * 1.05)

for si in range(n_cl):
    for ti in range(n_cl):
        if si == ti:
            ax_right.add_patch(plt.Rectangle((ti - 0.5, si - 0.5), 1, 1,
                               facecolor="#F0F0F0", edgecolor="none"))
            ax_right.text(ti, si, "—", ha="center", va="center",
                         fontsize=9, color="#CCCCCC")
        else:
            v = overlap_plot[si, ti]
            tc = "white" if v > np.nanmax(overlap_plot) * 0.55 else C_TEXT
            ax_right.text(ti, si, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8, color=tc)

ax_right.set_xticks(range(n_cl))
ax_right.set_xticklabels(cl_labels, fontsize=9, color=C_TEXT,
                         rotation=45, ha="right", rotation_mode="anchor")
ax_right.set_yticks(range(n_cl))
ax_right.set_yticklabels(cl_labels, fontsize=9, color=C_TEXT)
ax_right.set_xlabel("Target", fontsize=10, color=C_TEXT)
ax_right.set_ylabel("Source", fontsize=10, color=C_TEXT)
ax_right.set_title("Basis coverage (%)", fontsize=11, color=C_TEXT, pad=6)

for spine in ax_right.spines.values():
    spine.set_visible(False)
ax_right.tick_params(length=0)

cb_right = fig.colorbar(im_right, ax=ax_right, shrink=0.8, pad=0.02)
cb_right.ax.tick_params(labelsize=7, colors=C_TEXT)
cb_right.outline.set_edgecolor(C_TEXT)
cb_right.outline.set_linewidth(0.5)

# Spearman annotation
fig.text(0.5, 0.01, f"Spearman ρ(basis coverage, MLP pref) = {rho:.2f}, p = {p_val:.2f}",
         ha="center", fontsize=9, color=C_TEXT)

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT / "fig3e_basis_vs_transfer.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3e_basis_vs_transfer.pdf", dpi=300, bbox_inches="tight")
plt.close()
