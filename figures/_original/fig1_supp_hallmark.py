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
fig, ax = plt.subplots(figsize=(14, 3.0))

cmap = LinearSegmentedColormap.from_list("blue_white_terra",
    [C_TEMPLATE, "#8FADD4", C_ZERO, "#E8A882", C_INTERACTION])

vmax = np.nanmax(np.abs(plot_nes)) * 1.05
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(plot_nes, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(range(len(valid)))
ax.set_xticklabels(plot_labels, fontsize=6, color=C_TEXT, rotation=90, ha="center")
ax.set_yticks(range(len(CLS)))
ax.set_yticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT)

# FDR dots
for ci in range(len(CLS)):
    for si in range(len(valid)):
        if np.isnan(plot_fdr[ci, si]):
            continue
        if plot_fdr[ci, si] < 0.05:
            dot_c = "white" if abs(plot_nes[ci, si]) > vmax * 0.5 else C_TEXT
            ax.plot(si + 0.35, ci - 0.35, ".", markersize=3, color=dot_c)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
cbar.set_label("Template enrichment NES", fontsize=9, color=C_TEXT)
cbar.ax.tick_params(labelsize=7, colors=C_TEXT)
cbar.outline.set_edgecolor(C_TEXT)
cbar.outline.set_linewidth(0.5)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
fig.savefig(OUT / "s1d_full_hallmark_gsea.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s1d_full_hallmark_gsea.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s1d")


# ═══════════════════════════════════════════════
