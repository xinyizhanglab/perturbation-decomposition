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
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 3.2))
cmap_basis = LinearSegmentedColormap.from_list("white_grey", ["#F3F4F6", "#8A94A6"])
cmap_perf = LinearSegmentedColormap.from_list("white_teal", ["#F3F4F6", C_MLP])

for ax, mat, cmap_use, title, fmt in [
    (ax_l, overlap_matrix, cmap_basis, f"Basis coverage (r={BASIS_RANK})", "{:.0f}%"),
    (ax_r, mlp_matrix, cmap_perf, "MLP 1→1 transfer", "{:.3f}"),
]:
    vmax = np.nanmax(mat) * 1.05
    im = ax.imshow(mat, cmap=cmap_use, aspect="auto", vmin=0, vmax=vmax)
    for i in range(n_cl):
        for j in range(n_cl):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor="#F0F0F0", edgecolor="none"))
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#CCC")
            else:
                v = mat[i, j]
                tc = "white" if v > vmax * 0.55 else C_TEXT
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=tc)
    ax.set_xticks(range(n_cl))
    ax.set_xticklabels(cl_labels, fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_cl))
    ax.set_yticklabels(cl_labels, fontsize=9)
    ax.set_xlabel("Target", fontsize=10); ax.set_ylabel("Source", fontsize=10)
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

fig.text(0.5, 0.01, f"Spearman ρ(basis coverage, MLP pref) = {rho:.2f}, p = {p_val:.2f}",
         ha="center", fontsize=9, color=C_TEXT)

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT / "s3d_basis_gap_heatmap.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s3d_basis_gap_heatmap.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"    Saved s3d (ρ={rho:.3f})")

