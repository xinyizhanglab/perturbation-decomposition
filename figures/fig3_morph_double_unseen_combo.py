#!/usr/bin/env python3
"""
Figure: Norman double-gene prediction stratified by unseen-gene count.

Clean version:
- no annotation
- no highlighted background
- x tick labels at 45 degrees
- neutral wording for strata

Usage: python3 fig_morph_double_unseen.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure3"
OUT.mkdir(exist_ok=True)

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

C_RIDGE = "#4B5563"
C_MLP   = "#111827"
C_MORPH = "#3572B0"
C_SALT  = "#D26A3A"
C_ADD   = "#7C3AED"

# Neutral stratum labels
strata = ["0/2 unseen\nn=13", "1/2 unseen\nn=44", "2/2 unseen\nn=15"]
x = np.arange(len(strata))

# Updated values
ridge_sum = np.array([0.5735, 0.5572, 0.2822])
mlp_sum   = np.array([0.5162, 0.5603, 0.2031])
morph     = np.array([0.8760, 0.6206, 0.1173])
salt      = np.array([0.8285, np.nan, np.nan])
add_resid = np.array([0.8348, np.nan, np.nan])

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "text.color": C_TEXT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(4.2, 2.9))

# Main models
ax.plot(
    x, ridge_sum,
    marker="o", markersize=4.5, linewidth=1.4,
    color=C_RIDGE, markeredgecolor="white", markeredgewidth=0.4,
    label="Ridge (Σ DepMap)"
)

ax.plot(
    x, mlp_sum,
    marker="^", markersize=4.8, linewidth=1.4, linestyle="--",
    color=C_MLP, markeredgecolor="white", markeredgewidth=0.4,
    label="MLP (Σ DepMap)"
)

ax.plot(
    x, morph,
    marker="s", markersize=4.8, linewidth=1.8,
    color=C_MORPH, markeredgecolor="white", markeredgewidth=0.4,
    label="MORPH"
)

# 0/2-only reference points
ax.scatter(
    [x[0]], [salt[0]],
    marker="D", s=34,
    facecolor="white", edgecolor=C_SALT, linewidth=1.1,
    zorder=6, label="SALT"
)

ax.scatter(
    [x[0]], [add_resid[0]],
    marker="P", s=40,
    facecolor="white", edgecolor=C_ADD, linewidth=1.1,
    zorder=6, label="Add+Resid"
)

ax.set_xticks(x)
ax.set_xticklabels(strata, rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Unseen-gene stratum")
ax.set_ylabel("Pert-ref pearson")

ax.set_ylim(0.0, 0.95)
ax.set_xlim(-0.2, 2.2)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.45, alpha=0.8)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    fontsize=6.8,
    loc="upper right",
    handlelength=1.8,
    borderaxespad=0.3
)

fig.tight_layout()
fig.savefig(OUT / "fig3_morph_double_unseen_clean.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3_morph_double_unseen_clean.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3_morph_double_unseen_clean.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved to {OUT / 'fig3_morph_double_unseen_clean.svg'}")
print(f"       + {OUT / 'fig3_morph_double_unseen_clean.pdf'}")
print(f"       + {OUT / 'fig3_morph_double_unseen_clean.png'}")