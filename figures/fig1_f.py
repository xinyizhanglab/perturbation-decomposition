"""
Figure 1f: ANOVA decomposition of perturbation responses.
Stacked horizontal bars: full response and template-removed residual.
Components: template (α), conserved perturbation (β), interaction (γ), noise.
Noise estimated via split-half resampling.

The lower bar uses projective template removal:
  ε(c,p) = δ(c,p) - [δ(c,p) · T̂_c] T̂_c
where T̂_c is the unit template direction per cell line.

Run: python3 fig1_anova_stacked.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import anndata as ad
import json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure1"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]

C_TEMPLATE = "#3B5BA9"
C_CONSERVED = "#6F5CC4"
C_INTERACTION = "#D26A3A"
C_NOISE = "#C9CED6"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

N_SPLITS = 50

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT,
})

# ── Load single-cell data ─────────────────────────────────
print("Loading data for all cell lines...")
cl_sc_data = {}
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

    cl_sc_data[cl] = {"ctrl_mean": ctrl_mean, "pert_cells": pert_cells}
    print(f"  {cl}: {len(pert_cells)} perturbations with >=10 cells")

# ── Shared perturbations ──────────────────────────────────
shared = sorted(set.intersection(*[set(cl_sc_data[cl]["pert_cells"].keys()) for cl in CLS]))
print(f"\nShared perturbations: {len(shared)}")
n_cl = len(CLS)
n_p = len(shared)

# ── Full-data pseudobulk tensor ───────────────────────────
G = cl_sc_data[CLS[0]]["ctrl_mean"].shape[0]
D_full = np.zeros((n_cl, n_p, G))
for ci, cl in enumerate(CLS):
    ctrl_mean = cl_sc_data[cl]["ctrl_mean"]
    for pi, p in enumerate(shared):
        D_full[ci, pi] = cl_sc_data[cl]["pert_cells"][p].mean(0) - ctrl_mean

# ── ANOVA on full response ────────────────────────────────
mu = D_full.mean(axis=(0, 1))
alpha = D_full.mean(axis=1) - mu
beta = D_full.mean(axis=0) - mu
gamma = D_full - mu[None, None, :] - alpha[:, None, :] - beta[None, :, :]

ss_total = np.mean([np.sum(D_full[c, p] ** 2) for c in range(n_cl) for p in range(n_p)])

print(f"\nUncorrected ANOVA (% of total):")
for name, ss in [("mu", np.sum(mu ** 2)),
                  ("alpha", np.mean([np.sum(alpha[c] ** 2) for c in range(n_cl)])),
                  ("beta", np.mean([np.sum(beta[p] ** 2) for p in range(n_p)])),
                  ("gamma", np.mean([np.sum(gamma[c, p] ** 2) for c in range(n_cl) for p in range(n_p)]))]:
    print(f"  {name}:  {ss / ss_total * 100:.1f}%")


# ── Projective template removal ──────────────────────────
def project_out_template(D_tensor):
    """Remove per-CL template direction from each observation.
    ε(c,p) = δ(c,p) - [δ(c,p) · T̂_c] T̂_c
    """
    D_out = np.zeros_like(D_tensor)
    for ci in range(D_tensor.shape[0]):
        T_c = D_tensor[ci].mean(0)
        Tn_c = T_c / (np.linalg.norm(T_c) + 1e-30)
        for pi in range(D_tensor.shape[1]):
            D_out[ci, pi] = D_tensor[ci, pi] - np.dot(D_tensor[ci, pi], Tn_c) * Tn_c
    return D_out


D_resid = project_out_template(D_full)
ss_resid_total = np.mean(np.sum(D_resid ** 2, axis=2))

print(f"\nProjective removal:")
print(f"  SS full:     {ss_total:.3f}")
print(f"  SS residual: {ss_resid_total:.3f}  ({ss_resid_total / ss_total * 100:.1f}% of full)")


# ── Split-half noise estimation ───────────────────────────
def split_cells(rng):
    """Split cells into two halves, return two delta tensors."""
    D_h1 = np.zeros((n_cl, n_p, G))
    D_h2 = np.zeros((n_cl, n_p, G))
    for ci, cl in enumerate(CLS):
        ctrl_mean = cl_sc_data[cl]["ctrl_mean"]
        for pi, p in enumerate(shared):
            cells = cl_sc_data[cl]["pert_cells"][p]
            perm = rng.permutation(len(cells))
            half = len(cells) // 2
            D_h1[ci, pi] = cells[perm[:half]].mean(0) - ctrl_mean
            D_h2[ci, pi] = cells[perm[half:2 * half]].mean(0) - ctrl_mean
    return D_h1, D_h2


def anova_cross_sigs(D_h1, D_h2):
    """Cross-half dot products for each ANOVA component."""
    mu1 = D_h1.mean(axis=(0, 1)); mu2 = D_h2.mean(axis=(0, 1))
    a1 = D_h1.mean(axis=1) - mu1; a2 = D_h2.mean(axis=1) - mu2
    b1 = D_h1.mean(axis=0) - mu1; b2 = D_h2.mean(axis=0) - mu2
    g1 = D_h1 - mu1 - a1[:, None, :] - b1[None, :, :]
    g2 = D_h2 - mu2 - a2[:, None, :] - b2[None, :, :]
    return {
        "mu":    np.dot(mu1, mu2),
        "alpha": np.mean([np.dot(a1[c], a2[c]) for c in range(n_cl)]),
        "beta":  np.mean([np.dot(b1[p], b2[p]) for p in range(n_p)]),
        "gamma": np.mean([np.dot(g1[c, p], g2[c, p])
                          for c in range(n_cl) for p in range(n_p)]),
    }


print(f"\nRunning {N_SPLITS} split-half resamples...")
rng = np.random.RandomState(42)

splits_full = []
splits_resid = []

for split_i in range(N_SPLITS):
    D_h1, D_h2 = split_cells(rng)

    # Full-response signal
    splits_full.append(anova_cross_sigs(D_h1, D_h2))

    # Projective removal on each half independently
    D_h1_r = project_out_template(D_h1)
    D_h2_r = project_out_template(D_h2)
    splits_resid.append(anova_cross_sigs(D_h1_r, D_h2_r))

    if (split_i + 1) % 10 == 0:
        print(f"  {split_i + 1}/{N_SPLITS}")

# ── Full response numbers ─────────────────────────────────
sig_full = {k: np.mean([s[k] for s in splits_full])
            for k in ["mu", "alpha", "beta", "gamma"]}
noise_full = ss_total - sum(sig_full.values())

template_pct = (sig_full["mu"] + sig_full["alpha"]) / ss_total * 100
beta_pct = sig_full["beta"] / ss_total * 100
gamma_pct = sig_full["gamma"] / ss_total * 100
noise_pct = noise_full / ss_total * 100

print(f"\n  FULL RESPONSE:")
print(f"    Template (mu+alpha): {template_pct:.1f}%")
print(f"    Conserved (beta):    {beta_pct:.1f}%")
print(f"    Interaction (gamma): {gamma_pct:.1f}%")
print(f"    Noise:               {noise_pct:.1f}%")

# ── Template-removed residual numbers ─────────────────────
sig_resid = {k: np.mean([s[k] for s in splits_resid])
             for k in ["mu", "alpha", "beta", "gamma"]}
noise_resid = ss_resid_total - sum(sig_resid.values())

beta_resid_pct = sig_resid["beta"] / ss_resid_total * 100
gamma_resid_pct = sig_resid["gamma"] / ss_resid_total * 100
noise_resid_pct = noise_resid / ss_resid_total * 100

print(f"\n  TEMPLATE-REMOVED RESIDUAL:")
print(f"    Total SS:            {ss_resid_total:.3f}")
print(f"    mu (should be ~0):   {sig_resid['mu'] / ss_resid_total * 100:.4f}%")
print(f"    alpha (should be ~0):{sig_resid['alpha'] / ss_resid_total * 100:.4f}%")
print(f"    Conserved (beta):    {beta_resid_pct:.1f}%")
print(f"    Interaction (gamma): {gamma_resid_pct:.1f}%")
print(f"    Noise:               {noise_resid_pct:.1f}%")

# ── Plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.0, 2.2))

bar_height = 0.45
y_positions = [1, 0]
y_labels = ["Full response", "Template-removed\nresidual"]

# Full response: template | conserved | interaction | noise
full_vals = [template_pct, beta_pct, gamma_pct, noise_pct]
full_colors = [C_TEMPLATE, C_CONSERVED, C_INTERACTION, C_NOISE]
full_labels = [r"Cell-line template ($\mu + \alpha$)",
               r"Conserved perturbation ($\beta$)",
               r"Interaction ($\gamma$)",
               "Noise"]

# Residual: conserved | interaction | noise
resid_vals = [beta_resid_pct, gamma_resid_pct, noise_resid_pct]
resid_colors = [C_CONSERVED, C_INTERACTION, C_NOISE]

# Draw full response bar
left = 0
for val, col, lab in zip(full_vals, full_colors, full_labels):
    ax.barh(y_positions[0], val, left=left, height=bar_height,
            color=col, edgecolor="white", linewidth=0.5, label=lab, zorder=2)
    if val > 6:
        tc = "white" if col != C_NOISE else C_TEXT
        ax.text(left + val / 2, y_positions[0], f"{val:.1f}%",
                ha="center", va="center", fontsize=8, color=tc)
    left += val

# Draw residual bar
left = 0
for val, col in zip(resid_vals, resid_colors):
    ax.barh(y_positions[1], val, left=left, height=bar_height,
            color=col, edgecolor="white", linewidth=0.5, zorder=2)
    if val > 6:
        tc = "white" if col != C_NOISE else C_TEXT
        ax.text(left + val / 2, y_positions[1], f"{val:.1f}%",
                ha="center", va="center", fontsize=8, color=tc)
    left += val

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=10, color=C_TEXT)
ax.set_xlabel("Variance (%)", fontsize=10, color=C_TEXT)
ax.set_xlim(0, 102)

ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center",
           bbox_to_anchor=(0.5, -0.15), ncol=2, columnspacing=1.0)

plt.tight_layout(rect=[0, 0.16, 1, 1])
fig.savefig(OUT / "fig1f_anova_stacked.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1f_anova_stacked.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Save ──────────────────────────────────────────────────
results = {
    "full_response": {
        "template_pct": float(template_pct),
        "conserved_pct": float(beta_pct),
        "interaction_pct": float(gamma_pct),
        "noise_pct": float(noise_pct),
    },
    "template_removed": {
        "method": "projective",
        "conserved_pct": float(beta_resid_pct),
        "interaction_pct": float(gamma_resid_pct),
        "noise_pct": float(noise_resid_pct),
    },
    "raw_signal": {
        "full": {k: float(v) for k, v in sig_full.items()},
        "resid": {k: float(v) for k, v in sig_resid.items()},
    },
    "ss_total_full": float(ss_total),
    "ss_total_resid": float(ss_resid_total),
    "n_shared_perts": n_p,
    "n_splits": N_SPLITS,
}
with open(OUT / "fig1f_anova_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT}")