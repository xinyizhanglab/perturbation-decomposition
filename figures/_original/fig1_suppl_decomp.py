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
C_CONSERVED = "#6F5CC4"
C_INTERACTION = "#D26A3A"
C_NOISE = "#C9CED6"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}
N_SPLITS = 20
plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT,
})
fig, ax = plt.subplots(figsize=(5.5, 2.4))

bar_height = 0.5
method_keys = list(methods.keys())
y = np.arange(len(method_keys))[::-1]

for yi, mk in zip(y, method_keys):
    m = methods[mk]
    left = 0
    for val, col, lab in [(m["beta_pct"],  C_CONSERVED,   r"Conserved ($\beta$)"),
                           (m["gamma_pct"], C_INTERACTION, r"Interaction ($\gamma$)"),
                           (m["noise_pct"], C_NOISE,       "Noise")]:
        ax.barh(yi, val, left=left, height=bar_height,
                color=col, edgecolor="white", linewidth=0.5, zorder=2,
                label=lab if mk == method_keys[0] else None)
        if val > 8:
            tc = "white" if col != C_NOISE else C_TEXT
            ax.text(left + val / 2, yi, f"{val:.1f}%",
                    ha="center", va="center", fontsize=8, color=tc)
        left += val

ax.set_yticks(y)
ax.set_yticklabels(method_keys, fontsize=9, color=C_TEXT)
ax.set_xlabel("Residual variance (%)", fontsize=10, color=C_TEXT)
ax.set_xlim(0, 102)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center",
           bbox_to_anchor=(0.5, -0.12), ncol=3, columnspacing=1.5)

plt.tight_layout(rect=[0, 0.1, 1, 1])
fig.savefig(OUT / "decomp_supp_a_stacked.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "decomp_supp_a_stacked.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved panel a")


# ═══════════════════════════════════════════════════════════
# Panel b: Per-CL template geometry
# ═══════════════════════════════════════════════════════════
print("\n  Panel b: template geometry...")

fig, ax = plt.subplots(figsize=(5.5, 3.0))

x = np.arange(len(CLS))
width = 0.28

orig_vals = [geom[cl]["original"] for cl in CLS]
cent_vals = [geom[cl]["centroid"] for cl in CLS]
proj_vals = [geom[cl]["projected"] for cl in CLS]

bars_orig = ax.bar(x - width, orig_vals, width,
                   color=[CL_COLORS[cl] for cl in CLS],
                   edgecolor="white", linewidth=0.5, zorder=2)
bars_cent = ax.bar(x, cent_vals, width,
                   color=[CL_COLORS[cl] for cl in CLS],
                   edgecolor="white", linewidth=0.5, zorder=2, alpha=0.45)
bars_proj = ax.bar(x + width, proj_vals, width,
                   color=C_NOISE,
                   edgecolor="white", linewidth=0.5, zorder=2)

# Annotate ||T|| above each original bar
for ci, cl in enumerate(CLS):
    ax.text(x[ci] - width, orig_vals[ci] + 0.3,
            f"‖T‖ = {geom[cl]['T_norm']:.2f}",
            ha="center", va="bottom", fontsize=7, color=C_TEXT)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel("Mean variance along\ntemplate direction", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_elements = [
    Patch(facecolor=C_TEXT, alpha=0.7, label="Original"),
    Patch(facecolor=C_TEXT, alpha=0.3, label="After centroid removal"),
    Patch(facecolor=C_NOISE, alpha=1.0, label="After projection (≈ 0)"),
]
ax.legend(handles=legend_elements, frameon=False, fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(OUT / "decomp_supp_b_geometry.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "decomp_supp_b_geometry.pdf", dpi=300, bbox_inches="tight")
plt.close()
