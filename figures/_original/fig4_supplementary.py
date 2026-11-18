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
C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_CONSERVED = "#6F5CC4"
C_INTERACTION = "#D26A3A"
C_SOURCE = "#8A94A6"
C_LINE = "#C9CED6"
C_ZERO = "#FFFFFF"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT,
})
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.0, 2.8))
cmap_div = LinearSegmentedColormap.from_list("grey_teal", [C_INTERACTION, C_LINE, "#F3F4F6", "#A8D8D5", C_RESIDUAL])

for ax, mat, col_labels, title in [
    (ax_l, within_results, WITHIN_FEATS, "Within-CL"),
    (ax_r, cross_results, CROSS_FEATS, "Cross-CL 3→1"),
]:
    vmax = np.nanmax(np.abs(mat)) * 1.05
    norm = TwoSlopeNorm(vmin=-vmax * 0.3, vcenter=0, vmax=vmax)
    im = ax.imshow(mat, cmap=cmap_div, norm=norm, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v): ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#999"); continue
            tc = "white" if abs(v) > vmax * 0.5 else C_TEXT
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=tc)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.5, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(CLS)))
    ax.set_yticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9)
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

plt.tight_layout()
fig.savefig(OUT / "s4a_feature_ablation.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4a_feature_ablation.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4a")



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.0, 2.8))

models = ["Ridge", "MLP"]
x = np.arange(len(CLS))
width = 0.35

# Left: α correlation
for mi, model in enumerate(models):
    vals = [within_decomp[model][cl]["alpha_corr"] for cl in CLS]
    ax1.bar(x + mi * width - width / 2, vals, width, color=C_TEMPLATE if mi == 0 else C_RESIDUAL,
            edgecolor="white", linewidth=0.5, label=model, zorder=2)
ax1.set_xticks(x); ax1.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
ax1.set_ylabel("corr(α̂, α)", fontsize=10)
ax1.set_title("Template coefficient", fontsize=10, color=C_TEXT, pad=6)
ax1.legend(frameon=False, fontsize=8); ax1.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
ax1.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# Middle: ε correlation
for mi, model in enumerate(models):
    vals = [within_decomp[model][cl]["eps_corr"] for cl in CLS]
    ax2.bar(x + mi * width - width / 2, vals, width, color=C_TEMPLATE if mi == 0 else C_RESIDUAL,
            edgecolor="white", linewidth=0.5, label=model, zorder=2)
ax2.set_xticks(x); ax2.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
ax2.set_ylabel("corr(ε̂, ε)", fontsize=10)
ax2.set_title("Residual recovery", fontsize=10, color=C_TEXT, pad=6)
ax2.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
ax2.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# Right: variance attribution stacked bars
bar_width = 0.18
for mi, model in enumerate(models):
    for ci, cl in enumerate(CLS):
        tf = within_decomp[model][cl]["t_frac"]
        ef = within_decomp[model][cl]["e_frac"]
        xp = ci + mi * bar_width * 2.5 - bar_width
        ax3.bar(xp, tf, bar_width * 2, color=C_TEMPLATE, edgecolor="white", linewidth=0.3, zorder=2)
        ax3.bar(xp, ef, bar_width * 2, bottom=tf, color=C_RESIDUAL, edgecolor="white", linewidth=0.3, zorder=2)
        ax3.text(xp, 0.5, model[0], ha="center", va="center", fontsize=6, color="white")

ax3.set_xticks(x); ax3.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
ax3.set_ylabel("Predicted variance fraction", fontsize=10)
ax3.set_title("Variance attribution", fontsize=10, color=C_TEXT, pad=6)
ax3.set_ylim(0, 1.05)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

from matplotlib.patches import Patch
ax3.legend(handles=[Patch(facecolor=C_TEMPLATE, label="Template"),
                     Patch(facecolor=C_RESIDUAL, label="Residual")],
           frameon=False, fontsize=7, loc="upper right")

plt.tight_layout()
fig.savefig(OUT / "s4b_within_decomp.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4b_within_decomp.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4b")



fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.0))
metrics = [("corr_beta", "corr(δ̂, β)"), ("corr_gamma", "corr(δ̂, γ)"),
           ("r2_beta", "R²(β)"), ("r2_gamma", "R²(γ)")]

for mi, (metric, label) in enumerate(metrics):
    ax = axes[mi // 2, mi % 2]
    mat = np.array([[cross_decomp[model][cl][metric] for model in models] for cl in CLS])
    vmax = max(abs(np.nanmax(mat)), abs(np.nanmin(mat))) * 1.05
    if "gamma" in metric:
        cmap_use = LinearSegmentedColormap.from_list("b_w_t", [C_TEMPLATE, C_ZERO, C_INTERACTION])
    else:
        cmap_use = LinearSegmentedColormap.from_list("b_w_t", [C_TEMPLATE, C_ZERO, C_RESIDUAL])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(mat, cmap=cmap_use, norm=norm, aspect="auto")

    for i in range(len(CLS)):
        for j in range(len(models)):
            v = mat[i, j]
            tc = "white" if abs(v) > vmax * 0.5 else C_TEXT
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=tc)

    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=9)
    ax.set_yticks(range(len(CLS))); ax.set_yticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9)
    ax.set_title(label, fontsize=10, color=C_TEXT, pad=4)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

plt.tight_layout()
fig.savefig(OUT / "s4c_crosscl_decomp.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4c_crosscl_decomp.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4c")



metric_labels = {"pref": "Pert-ref Pearson", "eps_corr": "corr(ε̂, ε)",
                 "beta_corr": "corr(δ̂, β)", "gamma_corr": "corr(δ̂, γ)"}

fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.5))
for mi, metric in enumerate(state_metrics):
    ax = axes[mi]
    mat = np.full((len(CLS), 2), np.nan)
    for ci, cl in enumerate(CLS):
        for si, s in enumerate(state_settings):
            if state_data[cl][s] is not None:
                mat[ci, si] = state_data[cl][s][metric]

    vmax = np.nanmax(np.abs(mat)) * 1.1 if not np.all(np.isnan(mat)) else 1
    if metric == "gamma_corr":
        cmap_use = LinearSegmentedColormap.from_list("t", [C_TEMPLATE, C_ZERO, C_INTERACTION])
    else:
        cmap_use = LinearSegmentedColormap.from_list("t", ["#F3F4F6", C_RESIDUAL])
        vmax = max(vmax, 0.01)

    if metric == "gamma_corr":
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = None
        im = ax.imshow(mat, cmap=cmap_use, aspect="auto", vmin=0, vmax=vmax)

    if metric == "gamma_corr":
        im = ax.imshow(mat, cmap=cmap_use, norm=norm, aspect="auto")

    for i in range(len(CLS)):
        for j in range(2):
            v = mat[i, j]
            if np.isnan(v): ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#999"); continue
            tc = "white" if abs(v) > vmax * 0.5 else C_TEXT
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=tc)

    ax.set_xticks([0, 1]); ax.set_xticklabels(["Source\nonly", "+30%\ntarget"], fontsize=7)
    ax.set_yticks(range(len(CLS))); ax.set_yticklabels([CL_LABELS[cl] for cl in CLS] if mi == 0 else [], fontsize=9)
    ax.set_title(metric_labels[metric], fontsize=9, color=C_TEXT, pad=4)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

fig.suptitle("STATE", fontsize=12, color=C_TEXT, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "s4d_state_components.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4d_state_components.pdf", dpi=300, bbox_inches="tight")
plt.close()

# Plot MORPH: 4 small heatmaps
morph_pair_labels = list(morph_data.keys())
fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.5))
for mi, metric in enumerate(state_metrics):
    ax = axes[mi]
    mat = np.full((len(morph_pair_labels), 3), np.nan)
    for pi, pk in enumerate(morph_pair_labels):
        for si, s in enumerate(morph_settings_list):
            if morph_data[pk][s] is not None:
                mat[pi, si] = morph_data[pk][s][metric]

    vmax = np.nanmax(np.abs(mat)) * 1.1 if not np.all(np.isnan(mat)) else 1
    if metric == "gamma_corr":
        cmap_use = LinearSegmentedColormap.from_list("t", [C_TEMPLATE, C_ZERO, C_INTERACTION])
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(mat, cmap=cmap_use, norm=norm, aspect="auto")
    else:
        cmap_use = LinearSegmentedColormap.from_list("t", ["#F3F4F6", C_RESIDUAL])
        im = ax.imshow(mat, cmap=cmap_use, aspect="auto", vmin=0, vmax=max(vmax, 0.01))

    for i in range(len(morph_pair_labels)):
        for j in range(3):
            v = mat[i, j]
            if np.isnan(v): ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#999"); continue
            tc = "white" if abs(v) > vmax * 0.5 else C_TEXT
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=tc)

    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Source", "Ctrls", "+30%"], fontsize=7)
    ax.set_yticks(range(len(morph_pair_labels)))
    ax.set_yticklabels(morph_pair_labels if mi == 0 else [], fontsize=8)
    ax.set_title(metric_labels[metric], fontsize=9, color=C_TEXT, pad=4)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

fig.suptitle("MORPH", fontsize=12, color=C_TEXT, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "s4d_morph_components.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4d_morph_components.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4d")



fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)

x = np.arange(len(CLS))
s_colors = [C_SOURCE, C_CONSERVED, C_INTERACTION]
s_markers = ["o", "D", "o"]
offsets_3 = [-0.2, 0, 0.2]

for ci, cl in enumerate(CLS):
    # β panel
    for si, (s, col, mk) in enumerate(zip(calib_settings, s_colors, s_markers)):
        v = calib_results[cl][s]["beta"]
        ax_l.scatter(x[ci] + offsets_3[si], v, s=40, marker=mk, color=col,
                     edgecolor="white", linewidth=0.5, zorder=3,
                     label=s if ci == 0 else None)
    vals = [calib_results[cl][s]["beta"] for s in calib_settings]
    ax_l.plot([x[ci] + o for o in offsets_3], vals, color=C_LINE, linewidth=1.0, zorder=1)

    # γ panel
    for si, (s, col, mk) in enumerate(zip(calib_settings, s_colors, s_markers)):
        v = calib_results[cl][s]["gamma"]
        ax_r.scatter(x[ci] + offsets_3[si], v, s=40, marker=mk, color=col,
                     edgecolor="white", linewidth=0.5, zorder=3)
    vals = [calib_results[cl][s]["gamma"] for s in calib_settings]
    ax_r.plot([x[ci] + o for o in offsets_3], vals, color=C_LINE, linewidth=1.0, zorder=1)

for ax, title in [(ax_l, "Conserved component β"), (ax_r, "Target interaction γ")]:
    ax.axhline(y=0, color=C_TEXT, linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontsize=10, color=C_TEXT, pad=6)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax_l.set_ylabel("corr(prediction, component)", fontsize=10)
ax_l.legend(frameon=False, fontsize=7.5, loc="lower left")

plt.tight_layout()
fig.savefig(OUT / "s4e_pooled_calibration.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4e_pooled_calibration.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4e")


# ═══════════════════════════════════════════════
