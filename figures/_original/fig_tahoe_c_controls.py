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

# PANEL C: Controls
# ══════════════════════════════════════════════════════════════
fig_c, ax = plt.subplots(figsize=(3.5, 3.2))

ctrl_labels = ["Source CL\npert decomp", "Template\nremoved", "Permuted"]
ctrl_colors = [C_DRUG, "#149E9A", C_NULL]

for cl in CLS:
    source_val = battery["test9_template"][cl]["with_template"]
    notemplate_val = battery["test9_template"][cl]["without_template"]
    null_cl = battery["test3_leakage"].get(cl, {}).get("null_mean", None)

    x_pts = [0, 1]
    y_pts = [source_val, notemplate_val]
    if null_cl is not None:
        x_pts.append(2)
        y_pts.append(null_cl)

    ax.scatter(x_pts, y_pts, s=25, color=CL_COLORS[cl],
               edgecolor="white", linewidth=0.4, zorder=3)

mean_source = np.mean([battery["test9_template"][cl]["with_template"] for cl in CLS])
mean_noT = np.mean([battery["test9_template"][cl]["without_template"] for cl in CLS])
null_cls = [cl for cl in CLS if cl in battery["test3_leakage"]]
mean_null = np.mean([battery["test3_leakage"][cl]["null_mean"] for cl in null_cls])

ax.bar([0, 1, 2], [mean_source, mean_noT, mean_null],
       color=ctrl_colors, alpha=0.25, width=0.6, edgecolor="none")

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(ctrl_labels, fontsize=8, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylabel("Residual pref", fontsize=10)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(-0.08, 0.92)

leg_c = [Line2D([0], [0], marker="o", color=CL_COLORS[cl], markersize=4,
                markeredgecolor="white", linewidth=0, label=cl) for cl in CLS]
ax.legend(handles=leg_c, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
fig_c.savefig(OUT / "fig_tahoe_c_controls.svg", dpi=300, bbox_inches="tight")
fig_c.savefig(OUT / "fig_tahoe_c_controls.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_c)
print("Saved Panel C")

print(f"\nAll saved to {OUT}")