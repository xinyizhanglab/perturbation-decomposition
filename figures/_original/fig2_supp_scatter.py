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
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_CTRL = "#8A94A6"
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
fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.5))

for ci, cl in enumerate(CLS):
    dep_flat = curves[cl]["dep_flat"]
    full_flat = curves[cl]["full_flat"]
    resid_flat = curves[cl]["resid_flat"]
    idx = np.arange(len(dep_flat))

    # Top row: full delta
    ax = axes[0, ci]
    ax.scatter(dep_flat[idx], full_flat[idx], s=0.5, alpha=0.15, color=C_TEMPLATE,
               rasterized=True)
    rho = curves[cl]["rho_full"]
    ax.set_title(f"{CL_LABELS[cl]} (ρ={rho:.3f})", fontsize=9, color=C_TEXT, pad=3)
    ax.axhline(y=0, color=C_TEXT, linewidth=0.4, alpha=0.3)
    ax.axvline(x=0, color=C_TEXT, linewidth=0.4, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ci == 0:
        ax.set_ylabel("Full response\nsimilarity", fontsize=9, color=C_TEXT)

    # Bottom row: residual
    ax = axes[1, ci]
    ax.scatter(dep_flat[idx], resid_flat[idx], s=0.5, alpha=0.15, color=C_RESIDUAL,
               rasterized=True)
    rho = curves[cl]["rho_resid"]
    ax.set_title(f"ρ={rho:.3f}", fontsize=9, color=C_TEXT, pad=3)
    ax.set_xlabel("DepMap cosine", fontsize=9, color=C_TEXT)
    ax.axhline(y=0, color=C_TEXT, linewidth=0.4, alpha=0.3)
    ax.axvline(x=0, color=C_TEXT, linewidth=0.4, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ci == 0:
        ax.set_ylabel("Residual\nsimilarity", fontsize=9, color=C_TEXT)

plt.tight_layout()
fig.savefig(OUT / "s2c_pairwise_scatter.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s2c_pairwise_scatter.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s2c")

print(f"\nAll supplementary panels saved to {OUT}")