"""
Figure 1 Supplementary panels.
All panels save to figure_1_supplementary/.

Panels:
  S1a: PC variance curves for full response and residual
  S1b: Template alignment with full-response PC1 and control PC1
  S1c: Control PCs vs residual PCs energy captured (k=1..50)
  S1d: Full signature heatmap (all Hallmark + all lineage × all CLs)
  S1e: Split-half noise correction details
  S1f: ANOVA orthogonality diagnostics

Run: python3 fig1_supp.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from pathlib import Path
from sklearn.decomposition import PCA
import anndata as ad
import gseapy as gp
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure_1_supplementary"
OUT.mkdir(exist_ok=True)

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

# ═══════════════════════════════════════════════
# LOAD ALL DATA ONCE
# ═══════════════════════════════════════════════
print("Loading all data...")
cl_data = {}
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X_hvg = np.array(a.obsm["X_hvg"])
    X_full = np.array(a.X)
    G_hvg = X_hvg.shape[1]
    G_full = X_full.shape[1]
    genes = a.obs["gene"].astype(str).values
    ctrl_cells_hvg = X_hvg[genes == "non-targeting"]
    ctrl_mean_hvg = ctrl_cells_hvg.mean(0)

    hvg_mask = a.var["highly_variable"]
    hvg_names = list(a.var_names[hvg_mask])
    full_names = list(a.var_names)

    deltas = []
    pert_names = []
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5:
            deltas.append(X_hvg[genes == pn].mean(0) - ctrl_mean_hvg)
            pert_names.append(pn)
    D = np.array(deltas)
    T = D.mean(0)
    Tn = T / (np.linalg.norm(T) + 1e-30)
    alpha = D @ Tn
    R = D - np.outer(alpha, Tn)

    # PCA on full delta, residual, control
    n_pca = min(100, D.shape[0] - 1)
    pca_full = PCA(n_components=n_pca).fit(D)
    pca_resid = PCA(n_components=n_pca).fit(R)
    ctrl_centered = ctrl_cells_hvg - ctrl_cells_hvg.mean(0)
    pca_ctrl = PCA(n_components=min(n_pca, ctrl_centered.shape[0] - 1)).fit(ctrl_centered)

    # Full gene template for GSEA
    ctrl_mean_full = X_full[genes == "non-targeting"].mean(0)
    deltas_full = []
    for pn in pert_names:
        deltas_full.append(X_full[genes == pn].mean(0) - ctrl_mean_full)
    D_full = np.array(deltas_full)
    T_full = D_full.mean(0)

    cl_data[cl] = {
        "D": D, "R": R, "T": T, "Tn": Tn,
        "pca_full": pca_full, "pca_resid": pca_resid, "pca_ctrl": pca_ctrl,
        "ctrl_cells": ctrl_cells_hvg, "ctrl_mean": ctrl_mean_hvg,
        "pert_names": pert_names, "n_pca": n_pca,
        "G_hvg": G_hvg, "G_full": G_full,
        "hvg_names": hvg_names, "full_names": full_names,
        "T_full": T_full, "X_full": X_full, "genes": genes,
    }
    print(f"  {cl}: {len(pert_names)} perts, {G_hvg} HVG, {G_full} full genes")


# ═══════════════════════════════════════════════
# S1a: PC variance curves
# ═══════════════════════════════════════════════
print("\n  S1a: PC variance curves...")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

for cl in CLS:
    ev_full = cl_data[cl]["pca_full"].explained_variance_ratio_[:50]
    ev_resid = cl_data[cl]["pca_resid"].explained_variance_ratio_[:50]
    x = np.arange(1, len(ev_full) + 1)

    axes[0].plot(x, np.cumsum(ev_full) * 100, color=CL_COLORS[cl],
                 linewidth=1.5, label=CL_LABELS[cl])
    axes[1].plot(x, np.cumsum(ev_resid) * 100, color=CL_COLORS[cl],
                 linewidth=1.5, label=CL_LABELS[cl])

for ax, title in zip(axes, ["Full response", "After template removal"]):
    ax.set_xlabel("Principal component", fontsize=10, color=C_TEXT)
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    ax.axhline(y=50, color=C_GRID, linewidth=0.8, linestyle="--")
    ax.axhline(y=80, color=C_GRID, linewidth=0.8, linestyle="--")
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Cumulative variance (%)", fontsize=10, color=C_TEXT)
axes[0].legend(frameon=False, fontsize=8, loc="lower right")

plt.tight_layout()
fig.savefig(OUT / "s1a_pc_variance_curves.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s1a_pc_variance_curves.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s1a")


# ═══════════════════════════════════════════════
# S1b: Template alignment with PC1s
# ═══════════════════════════════════════════════
print("\n  S1b: Template alignment...")

fig, ax = plt.subplots(figsize=(4.0, 2.8))

x = np.arange(len(CLS))
width = 0.32

cos_delta_pc1 = []
cos_ctrl_pc1 = []

for cl in CLS:
    Tn = cl_data[cl]["Tn"]
    delta_pc1 = cl_data[cl]["pca_full"].components_[0]
    ctrl_pc1 = cl_data[cl]["pca_ctrl"].components_[0]
    cos_d = abs(float(np.dot(Tn, delta_pc1 / (np.linalg.norm(delta_pc1) + 1e-12))))
    cos_c = abs(float(np.dot(Tn, ctrl_pc1 / (np.linalg.norm(ctrl_pc1) + 1e-12))))
    cos_delta_pc1.append(cos_d)
    cos_ctrl_pc1.append(cos_c)

ax.bar(x - width / 2, cos_delta_pc1, width, color=C_TEMPLATE, edgecolor="white",
       linewidth=0.5, label="Response PC1", zorder=2)
ax.bar(x + width / 2, cos_ctrl_pc1, width, color=C_CTRL, edgecolor="white",
       linewidth=0.5, label="Control PC1", zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel("|cos(template, PC1)|", fontsize=10, color=C_TEXT)
ax.set_ylim(0, 1.05)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, fontsize=8, loc="center left",
          bbox_to_anchor=(1.02, 0.5))

plt.tight_layout(rect=[0, 0, 0.82, 1])
fig.savefig(OUT / "s1b_template_pc1_alignment.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s1b_template_pc1_alignment.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s1b")
print(f"    Values: delta_PC1={cos_delta_pc1}, ctrl_PC1={cos_ctrl_pc1}")


# ═══════════════════════════════════════════════
# S1c: Control PCs vs Residual PCs energy captured
# ═══════════════════════════════════════════════
print("\n  S1c: Control PCs vs residual PCs energy...")

fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5), sharex=True, sharey=True)

k_range = np.arange(1, 51)

for ci, cl in enumerate(CLS):
    ax = axes[ci // 2, ci % 2]
    R = cl_data[cl]["R"]
    U_ctrl = cl_data[cl]["pca_ctrl"].components_
    U_resid = cl_data[cl]["pca_resid"].components_
    n_perts = R.shape[0]

    ctrl_energy = []
    resid_energy = []

    for k in k_range:
        k_ctrl = min(k, U_ctrl.shape[0])
        k_resid = min(k, U_resid.shape[0])

        # Control PCs → residual
        Uk_c = U_ctrl[:k_ctrl]
        fracs_c = []
        for i in range(n_perts):
            e = R[i]
            en2 = np.linalg.norm(e) ** 2
            if en2 < 1e-12:
                continue
            e_in = Uk_c.T @ (Uk_c @ e)
            fracs_c.append(np.linalg.norm(e_in) ** 2 / en2)
        ctrl_energy.append(np.mean(fracs_c) * 100)

        # Residual PCs → residual
        Uk_r = U_resid[:k_resid]
        fracs_r = []
        for i in range(n_perts):
            e = R[i]
            en2 = np.linalg.norm(e) ** 2
            if en2 < 1e-12:
                continue
            e_in = Uk_r.T @ (Uk_r @ e)
            fracs_r.append(np.linalg.norm(e_in) ** 2 / en2)
        resid_energy.append(np.mean(fracs_r) * 100)

    ax.plot(k_range, ctrl_energy, color=C_CTRL, linewidth=1.5, label="Control PCs")
    ax.plot(k_range, resid_energy, color=C_RESIDUAL, linewidth=1.5, label="Residual PCs")
    ax.fill_between(k_range, ctrl_energy, resid_energy, alpha=0.1, color=C_RESIDUAL)

    ax.set_title(CL_LABELS[cl], fontsize=11, color=C_TEXT, pad=4)
    ax.set_ylim(0, 80)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ci == 0:
        ax.legend(frameon=False, fontsize=8, loc="lower right")

axes[1, 0].set_xlabel("Number of PCs", fontsize=10, color=C_TEXT)
axes[1, 1].set_xlabel("Number of PCs", fontsize=10, color=C_TEXT)
axes[0, 0].set_ylabel("Residual variance\ncaptured (%)", fontsize=10, color=C_TEXT)
axes[1, 0].set_ylabel("Residual variance\ncaptured (%)", fontsize=10, color=C_TEXT)

plt.tight_layout()
fig.savefig(OUT / "s1c_ctrl_vs_resid_pcs.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s1c_ctrl_vs_resid_pcs.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s1c")


# ═══════════════════════════════════════════════
# S1d: Full signature heatmap (all Hallmark × all CLs)
# ═══════════════════════════════════════════════
print("\n  S1d: Full Hallmark GSEA heatmap...")

hallmark_lib = gp.get_library("MSigDB_Hallmark_2020")
hallmark_keys = sorted(hallmark_lib.keys())
print(f"    {len(hallmark_keys)} Hallmark sets")

nes_full = np.full((len(CLS), len(hallmark_keys)), np.nan)
fdr_full = np.full((len(CLS), len(hallmark_keys)), np.nan)

for ci, cl in enumerate(CLS):
    T_full = cl_data[cl]["T_full"]
    full_names = cl_data[cl]["full_names"]

    ranked = pd.Series(T_full, index=full_names)
    ranked = ranked[~ranked.index.duplicated()].sort_values(ascending=False)

    try:
        res = gp.prerank(rnk=ranked, gene_sets=hallmark_lib, min_size=5,
                         max_size=500, permutation_num=1000, seed=42,
                         no_plot=True, verbose=False)
        df = res.res2d
        for si, sk in enumerate(hallmark_keys):
            row = df[df["Term"] == sk]
            if len(row) > 0:
                nes_full[ci, si] = float(row["NES"].iloc[0])
                fdr_full[ci, si] = float(row["FDR q-val"].iloc[0])
    except Exception as e:
        print(f"    {cl} GSEA error: {e}")
    print(f"    {cl}: done")

# Filter to sets with valid NES
valid = [si for si in range(len(hallmark_keys)) if not np.all(np.isnan(nes_full[:, si]))]
plot_nes = nes_full[:, valid]
plot_fdr = fdr_full[:, valid]
plot_labels = [hallmark_keys[si].replace(" ", "\n", 1) if len(hallmark_keys[si]) > 15
               else hallmark_keys[si] for si in valid]

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
# S1e: Split-half noise correction details
# ═══════════════════════════════════════════════
print("\n  S1e: Split-half noise correction...")

N_SPLITS = 50
rng = np.random.RandomState(42)

# Load single-cell data for split-half
cl_sc = {}
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    pert_cells = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        idx = np.where(genes == pn)[0]
        if len(idx) >= 10:
            pert_cells[pn] = X[idx]
    cl_sc[cl] = {"ctrl_mean": X[genes == "non-targeting"].mean(0), "pert_cells": pert_cells}

shared = sorted(set.intersection(*[set(cl_sc[cl]["pert_cells"].keys()) for cl in CLS]))
n_cl, n_p = len(CLS), len(shared)
G = cl_sc[CLS[0]]["ctrl_mean"].shape[0]

# Full-data ANOVA
D_tensor = np.zeros((n_cl, n_p, G))
for ci, cl in enumerate(CLS):
    for pi, p in enumerate(shared):
        D_tensor[ci, pi] = cl_sc[cl]["pert_cells"][p].mean(0) - cl_sc[cl]["ctrl_mean"]

mu = D_tensor.mean(axis=(0, 1))
alpha_a = D_tensor.mean(axis=1) - mu
beta_a = D_tensor.mean(axis=0) - mu
gamma_a = D_tensor - mu - alpha_a[:, None, :] - beta_a[None, :, :]

ss_total = np.mean([np.sum(D_tensor[c, p] ** 2) for c in range(n_cl) for p in range(n_p)])

# Split-half across multiple resamples
split_results = {"alpha": [], "beta": [], "gamma": []}

for si in range(N_SPLITS):
    D_h1 = np.zeros_like(D_tensor)
    D_h2 = np.zeros_like(D_tensor)
    for ci, cl in enumerate(CLS):
        for pi, p in enumerate(shared):
            cells = cl_sc[cl]["pert_cells"][p]
            perm = rng.permutation(len(cells))
            half = len(cells) // 2
            D_h1[ci, pi] = cells[perm[:half]].mean(0) - cl_sc[cl]["ctrl_mean"]
            D_h2[ci, pi] = cells[perm[half:2 * half]].mean(0) - cl_sc[cl]["ctrl_mean"]

    mu1 = D_h1.mean(axis=(0, 1)); mu2 = D_h2.mean(axis=(0, 1))
    a1 = D_h1.mean(axis=1) - mu1; a2 = D_h2.mean(axis=1) - mu2
    b1 = D_h1.mean(axis=0) - mu1; b2 = D_h2.mean(axis=0) - mu2
    g1 = D_h1 - mu1 - a1[:, None, :] - b1[None, :, :]
    g2 = D_h2 - mu2 - a2[:, None, :] - b2[None, :, :]

    split_results["alpha"].append(np.mean([np.dot(a1[c], a2[c]) for c in range(n_cl)]))
    split_results["beta"].append(np.mean([np.dot(b1[p], b2[p]) for p in range(n_p)]))
    split_results["gamma"].append(np.mean([np.dot(g1[c, p], g2[c, p])
                                           for c in range(n_cl) for p in range(n_p)]))
    if (si + 1) % 10 == 0:
        print(f"    Split {si + 1}/{N_SPLITS}")

fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.5))
comp_names = ["alpha", "beta", "gamma"]
comp_colors = [C_TEMPLATE, C_CONSERVED, C_INTERACTION]
comp_labels = ["Cell line\ntemplate (α)", "Conserved\npert. (β)", "Interaction\n(γ)"]

for ai, (comp, col, lab) in enumerate(zip(comp_names, comp_colors, comp_labels)):
    ax = axes[ai]
    vals = np.array(split_results[comp])
    uncorrected = {"alpha": np.mean([np.sum(alpha_a[c] ** 2) for c in range(n_cl)]),
                   "beta": np.mean([np.sum(beta_a[p] ** 2) for p in range(n_p)]),
                   "gamma": np.mean([np.sum(gamma_a[c, p] ** 2) for c in range(n_cl) for p in range(n_p)])}

    ax.hist(vals / ss_total * 100, bins=15, color=col, alpha=0.7, edgecolor="white")
    ax.axvline(x=uncorrected[comp] / ss_total * 100, color=C_TEXT, linewidth=1.2,
               linestyle="--", label="Uncorrected")
    ax.axvline(x=np.mean(vals) / ss_total * 100, color=col, linewidth=1.5,
               label="Signal (mean)")

    ax.set_xlabel("% of total variance", fontsize=9, color=C_TEXT)
    ax.set_title(lab, fontsize=10, color=C_TEXT, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if ai == 0:
        ax.set_ylabel("Count", fontsize=9, color=C_TEXT)
        ax.legend(frameon=False, fontsize=7)

plt.tight_layout()
fig.savefig(OUT / "s1e_split_half_noise.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s1e_split_half_noise.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s1e")


# ═══════════════════════════════════════════════
# S1f: ANOVA orthogonality diagnostics
# ═══════════════════════════════════════════════
print("\n  S1f: ANOVA orthogonality diagnostics...")

# Cross-component dot products (should be ~0)
cross_ab = np.mean([np.dot(alpha_a[c], beta_a[p]) for c in range(n_cl) for p in range(n_p)])
cross_ag = np.mean([np.dot(alpha_a[c], gamma_a[c, p]) for c in range(n_cl) for p in range(n_p)])
cross_bg = np.mean([np.dot(beta_a[p], gamma_a[c, p]) for c in range(n_cl) for p in range(n_p)])

# Additivity check: sum of component SS vs total SS
ss_mu = np.sum(mu ** 2)
ss_a = np.mean([np.sum(alpha_a[c] ** 2) for c in range(n_cl)])
ss_b = np.mean([np.sum(beta_a[p] ** 2) for p in range(n_p)])
ss_g = np.mean([np.sum(gamma_a[c, p] ** 2) for c in range(n_cl) for p in range(n_p)])
ss_sum = ss_mu + ss_a + ss_b + ss_g

# Per-CL variance decomposition
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))

# Left panel: orthogonality
ax = axes[0]
pairs = ["α · β", "α · γ", "β · γ"]
cross_vals = [cross_ab, cross_ag, cross_bg]
colors = [C_CONSERVED, C_INTERACTION, C_CTRL]

ax.barh(range(3), cross_vals, height=0.5, color=colors, edgecolor="white", linewidth=0.5)
ax.axvline(x=0, color=C_TEXT, linewidth=0.6)
ax.set_yticks(range(3))
ax.set_yticklabels(pairs, fontsize=10, color=C_TEXT)
ax.set_xlabel("Mean cross-component\ndot product", fontsize=9, color=C_TEXT)
ax.set_title("Orthogonality check", fontsize=11, color=C_TEXT, pad=6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, color=C_GRID, linewidth=0.5)
ax.set_axisbelow(True)

# Right panel: per-CL β/γ ratio
ax = axes[1]
x = np.arange(len(CLS))
width = 0.3

beta_per_cl = []
gamma_per_cl = []
for ci in range(n_cl):
    b_v = np.mean([np.sum(beta_a[p] ** 2) for p in range(n_p)])
    g_v = np.mean([np.sum(gamma_a[ci, p] ** 2) for p in range(n_p)])
    beta_per_cl.append(b_v)
    gamma_per_cl.append(g_v)

ax.bar(x - width / 2, beta_per_cl, width, color=C_CONSERVED, edgecolor="white",
       linewidth=0.5, label="β (conserved)", zorder=2)
ax.bar(x + width / 2, gamma_per_cl, width, color=C_INTERACTION, edgecolor="white",
       linewidth=0.5, label="γ (interaction)", zorder=2)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel("Mean variance", fontsize=10, color=C_TEXT)
ax.set_title("Per-CL component variance", fontsize=11, color=C_TEXT, pad=6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="upper right")

plt.tight_layout()
fig.savefig(OUT / "s1f_anova_diagnostics.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s1f_anova_diagnostics.pdf", dpi=300, bbox_inches="tight")
plt.close()

print("    Saved s1f")
print(f"    Orthogonality: α·β={cross_ab:.6f}, α·γ={cross_ag:.6f}, β·γ={cross_bg:.6f}")
print(f"    Additivity: Σ components={ss_sum:.4f}, total={ss_total:.4f}, ratio={ss_sum/ss_total:.6f}")

# ── Save all results ──
all_results = {
    "s1b": {
        "cos_delta_pc1": {cl: float(cos_delta_pc1[i]) for i, cl in enumerate(CLS)},
        "cos_ctrl_pc1": {cl: float(cos_ctrl_pc1[i]) for i, cl in enumerate(CLS)},
    },
    "s1f": {
        "cross_dot_products": {"alpha_beta": float(cross_ab), "alpha_gamma": float(cross_ag),
                               "beta_gamma": float(cross_bg)},
        "additivity_ratio": float(ss_sum / ss_total),
        "beta_per_cl": {cl: float(beta_per_cl[i]) for i, cl in enumerate(CLS)},
        "gamma_per_cl": {cl: float(gamma_per_cl[i]) for i, cl in enumerate(CLS)},
    },
}
with open(OUT / "fig1_supp_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll supplementary panels saved to {OUT}")