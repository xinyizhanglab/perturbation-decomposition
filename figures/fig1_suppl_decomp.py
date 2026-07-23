"""
Decomposition supplementary: template removal method comparison.

Panel a: Stacked bars — residual fractions under three removal methods
Panel b: Per-CL template geometry — variance along T under each method

Saved to decomp_supplementary/.

Run: python3 decomp_supp.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import anndata as ad
import json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "decomp_supplementary"
OUT.mkdir(exist_ok=True)

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

# ── Load data ──────────────────────────────────────────────
print("Loading data...")
cl_sc = {}
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)
    pert_cells = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        idx = np.where(genes == pn)[0]
        if len(idx) >= 10:
            pert_cells[pn] = X[idx]
    cl_sc[cl] = {"ctrl_mean": ctrl_mean, "pert_cells": pert_cells}

shared = sorted(set.intersection(*[set(cl_sc[cl]["pert_cells"].keys()) for cl in CLS]))
n_cl, n_p = len(CLS), len(shared)
G = cl_sc[CLS[0]]["ctrl_mean"].shape[0]
print(f"  Shared perts: {n_p}, CLs: {n_cl}, genes: {G}")

# ── Full-data tensor ──────────────────────────────────────
D = np.zeros((n_cl, n_p, G))
for ci, cl in enumerate(CLS):
    for pi, p in enumerate(shared):
        D[ci, pi] = cl_sc[cl]["pert_cells"][p].mean(0) - cl_sc[cl]["ctrl_mean"]

ss_total = np.mean(np.sum(D ** 2, axis=2))

# ── Centroid-removed tensor ───────────────────────────────
D_centroid = D - D.mean(axis=1, keepdims=True)
ss_centroid = np.mean(np.sum(D_centroid ** 2, axis=2))

# ── Projective-removed tensor ────────────────────────────
D_proj = np.zeros_like(D)
T_per_cl = {}
for ci, cl in enumerate(CLS):
    T_c = D[ci].mean(0)
    Tn_c = T_c / (np.linalg.norm(T_c) + 1e-30)
    T_per_cl[cl] = {"T": T_c, "Tn": Tn_c}
    for pi in range(n_p):
        D_proj[ci, pi] = D[ci, pi] - np.dot(D[ci, pi], Tn_c) * Tn_c
ss_proj = np.mean(np.sum(D_proj ** 2, axis=2))

# ── Split-half noise for all three ────────────────────────
print(f"Running {N_SPLITS} split-half resamples...")
rng = np.random.RandomState(42)


def anova_sigs(Dh1, Dh2):
    mu1 = Dh1.mean((0, 1)); mu2 = Dh2.mean((0, 1))
    a1 = Dh1.mean(1) - mu1;  a2 = Dh2.mean(1) - mu2
    b1 = Dh1.mean(0) - mu1;  b2 = Dh2.mean(0) - mu2
    g1 = Dh1 - mu1 - a1[:, None, :] - b1[None, :, :]
    g2 = Dh2 - mu2 - a2[:, None, :] - b2[None, :, :]
    return {
        "mu":    np.dot(mu1, mu2),
        "alpha": np.mean([np.dot(a1[c], a2[c]) for c in range(n_cl)]),
        "beta":  np.mean([np.dot(b1[p], b2[p]) for p in range(n_p)]),
        "gamma": np.mean([np.dot(g1[c, p], g2[c, p])
                          for c in range(n_cl) for p in range(n_p)]),
    }


all_sigs = {"orig": [], "cent": [], "proj": []}

for si in range(N_SPLITS):
    D1 = np.zeros_like(D); D2 = np.zeros_like(D)
    for ci, cl in enumerate(CLS):
        for pi, p in enumerate(shared):
            cells = cl_sc[cl]["pert_cells"][p]
            perm = rng.permutation(len(cells))
            h = len(cells) // 2
            D1[ci, pi] = cells[perm[:h]].mean(0) - cl_sc[cl]["ctrl_mean"]
            D2[ci, pi] = cells[perm[h:2 * h]].mean(0) - cl_sc[cl]["ctrl_mean"]

    # Method 1: original
    all_sigs["orig"].append(anova_sigs(D1, D2))

    # Method 2: centroid removal
    D1_c = D1 - D1.mean(axis=1, keepdims=True)
    D2_c = D2 - D2.mean(axis=1, keepdims=True)
    all_sigs["cent"].append(anova_sigs(D1_c, D2_c))

    # Method 3: projective removal
    D1_p = np.zeros_like(D1); D2_p = np.zeros_like(D2)
    for ci in range(n_cl):
        T1 = D1[ci].mean(0); Tn1 = T1 / (np.linalg.norm(T1) + 1e-30)
        T2 = D2[ci].mean(0); Tn2 = T2 / (np.linalg.norm(T2) + 1e-30)
        for pi in range(n_p):
            D1_p[ci, pi] = D1[ci, pi] - np.dot(D1[ci, pi], Tn1) * Tn1
            D2_p[ci, pi] = D2[ci, pi] - np.dot(D2[ci, pi], Tn2) * Tn2
    all_sigs["proj"].append(anova_sigs(D1_p, D2_p))

    if (si + 1) % 5 == 0:
        print(f"  {si + 1}/{N_SPLITS}")

# ── Compute residual fractions for each method ────────────
method_specs = [
    ("ANOVA variance\naccounting", ss_total,    all_sigs["orig"]),
    ("Centroid\nremoval",          ss_centroid,  all_sigs["cent"]),
    ("Projective\nremoval",        ss_proj,      all_sigs["proj"]),
]

methods = {}
for label, ss_tot, sigs_list in method_specs:
    sig = {k: np.mean([s[k] for s in sigs_list])
           for k in ["mu", "alpha", "beta", "gamma"]}
    noise = ss_tot - sum(sig.values())
    resid = ss_tot - sig["mu"] - sig["alpha"]
    methods[label] = {
        "beta_pct":  sig["beta"]  / resid * 100,
        "gamma_pct": sig["gamma"] / resid * 100,
        "noise_pct": noise        / resid * 100,
    }

# ── Per-CL template geometry ─────────────────────────────
geom = {}
for ci, cl in enumerate(CLS):
    Tn_c = T_per_cl[cl]["Tn"]
    var_orig = np.mean([np.dot(D[ci, pi], Tn_c) ** 2           for pi in range(n_p)])
    var_cent = np.mean([np.dot(D_centroid[ci, pi], Tn_c) ** 2  for pi in range(n_p)])
    var_proj = np.mean([np.dot(D_proj[ci, pi], Tn_c) ** 2      for pi in range(n_p)])
    geom[cl] = {
        "T_norm":    float(np.linalg.norm(T_per_cl[cl]["T"])),
        "original":  float(var_orig),
        "centroid":  float(var_cent),
        "projected": float(var_proj),
    }


# ═══════════════════════════════════════════════════════════
# Panel a: Stacked bars
# ═══════════════════════════════════════════════════════════
print("\n  Panel a: stacked bars...")

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
print("    Saved panel b")


# ── Print summary ─────────────────────────────────────────
print(f"\n{'='*70}")
print("Panel a: Residual fractions")
print(f"{'='*70}")
header = f"{'Method':<25} {'beta':>8} {'gamma':>8} {'noise':>8}"
print(header)
print("-" * len(header))
for mk in method_keys:
    m = methods[mk]
    label = mk.replace("\n", " ")
    print(f"{label:<25} {m['beta_pct']:>7.1f}% {m['gamma_pct']:>7.1f}% {m['noise_pct']:>7.1f}%")

print(f"\n{'='*70}")
print("Panel b: Per-CL template geometry")
print(f"{'='*70}")
print(f"{'CL':<8} {'||T||':>8} {'Original':>10} {'Centroid':>10} {'Projected':>10}")
print("-" * 50)
for cl in CLS:
    g = geom[cl]
    print(f"{CL_LABELS[cl]:<8} {g['T_norm']:>8.3f} {g['original']:>10.3f} "
          f"{g['centroid']:>10.3f} {g['projected']:>10.6f}")

# ── Save JSON ─────────────────────────────────────────────
save_data = {
    "methods": {},
    "geometry": geom,
    "n_shared_perts": n_p,
    "n_splits": N_SPLITS,
}
for mk in method_keys:
    save_data["methods"][mk.replace("\n", " ")] = {
        k: float(v) for k, v in methods[mk].items()
    }

with open(OUT / "decomp_supp_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")