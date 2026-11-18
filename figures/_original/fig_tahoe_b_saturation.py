"""Original Tahoe plotting statements on saved numerical results."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_DRUG = "#3572B0"
C_MOA = "#D26A3A"
C_NULL = "#C9CED6"
CLS = ["C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1"]
CL_COLORS = {"C32": "#4A90D9", "HOP62": "#D9534F", "HepG2/C3A": "#5CB85C",
             "Hs 766T": "#F0AD4E", "PANC-1": "#6F5CC4"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 10,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ══════════════════════════════════════════════════════════════

# PANEL B: Source basis saturation
# ══════════════════════════════════════════════════════════════
fig_b, ax = plt.subplots(figsize=(4.5, 3.2))

sat = battery["test2_saturation"]
RANKS = [1, 2, 5, 10, 20, 50, 100, 150]

for cl in CLS:
    y_vals = [sat[cl][str(k)]["mean"] for k in RANKS]
    ax.plot(RANKS, y_vals, marker="o", markersize=3.5, linewidth=1.3,
            color=CL_COLORS[cl], markeredgecolor="white", markeredgewidth=0.3)

ax.set_xscale("log")
ax.set_xlabel("Source SVD programs (k)", fontsize=10)
ax.set_ylabel("Residual pref", fontsize=10)
ax.set_xticks(RANKS)
ax.set_xticklabels([str(k) for k in RANKS], fontsize=8)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg_b = [Line2D([0], [0], color=CL_COLORS[cl], linewidth=2, marker="o", markersize=3,
                markeredgecolor="white", label=cl) for cl in CLS]
ax.legend(handles=leg_b, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
fig_b.savefig(OUT / "fig_tahoe_b_saturation.svg", dpi=300, bbox_inches="tight")
fig_b.savefig(OUT / "fig_tahoe_b_saturation.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_b)
print("Saved Panel B")

# ══════════════════════════════════════════════════════════════
