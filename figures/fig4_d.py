"""
Figure 4d: Target perturbations improve prediction mainly by revealing γ, not β.
Two aligned subpanels: β recovery (left, teal) and γ recovery (right, terracotta).
Each CL has two dots: zero-shot vs target perturbations 30%.

Run: python3 fig4_gamma_reveal.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import anndata as ad
import pickle, json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
MORPH_BASE = Path("/mnt/storage01/home/amolina/agop-perturbation/morph_pipeline")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure4"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_LINE = "#C9CED6"
C_ZEROSHOT = "#8A94A6"
C_CALIBRATED = "#D26A3A"
C_BETA = "#149E9A"
C_GAMMA = "#D26A3A"

CALIB_FRAC = 0.3

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

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

# ── Load DepMap embeddings ──
emb_dict = {}
for ds in ["replogle_k562_essential", "replogle_rpe1_essential", "nadig_hepg2", "nadig_jurkat"]:
    ep = MORPH_BASE / f"outputs/{ds}/pert_embeddings.pkl"
    if ep.exists():
        with open(ep, "rb") as f:
            emb_dict.update(pickle.load(f))

def get_emb(p):
    for k in [p + "+ctrl", p]:
        if k in emb_dict:
            return emb_dict[k].astype(np.float32)
    return None

# ── Load data ──
print("Loading data...")
cl_data = {}
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)
    deltas = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5 and get_emb(pn) is not None:
            deltas[pn] = X[genes == pn].mean(0) - ctrl_mean
    cl_data[cl] = deltas
    print(f"  {cl}: {len(deltas)} perts")

# ── ANOVA on shared perts ──
shared = sorted(set.intersection(*[set(cl_data[cl].keys()) for cl in CLS]))
D_tensor = np.array([[cl_data[cl][p] for p in shared] for cl in CLS])
mu = D_tensor.mean(axis=(0, 1))
alpha_a = D_tensor.mean(axis=1) - mu
beta_a = D_tensor.mean(axis=0) - mu
gamma_a = D_tensor - mu - alpha_a[:, None, :] - beta_a[None, :, :]
cl_idx = {c: i for i, c in enumerate(CLS)}
p_idx = {p: i for i, p in enumerate(shared)}
print(f"ANOVA shared perts: {len(shared)}")

def decompose_anova(perts, yp, tgt):
    c = cl_idx[tgt]
    bc, gc = [], []
    for i, p in enumerate(perts):
        if p not in p_idx:
            continue
        pidx = p_idx[p]
        bc.append(corr(yp[i], beta_a[pidx]))
        gc.append(corr(yp[i], gamma_a[c, pidx]))
    return float(np.mean(bc)), float(np.mean(gc))

# ── Compute zero-shot vs target-only 30% ──
results = {cl: {} for cl in CLS}

for tgt in CLS:
    print(f"\n  Target: {CL_LABELS[tgt]}")
    srcs = [cl for cl in CLS if cl != tgt]

    sV, sD = [], []
    for cl in srcs:
        for pn, d in cl_data[cl].items():
            sV.append(get_emb(pn)); sD.append(d)
    sV = np.array(sV); sD = np.array(sD)

    tgt_shared = [p for p in shared if p in cl_data[tgt]]
    tV = np.array([get_emb(p) for p in tgt_shared])
    tD = np.array([cl_data[tgt][p] for p in tgt_shared])

    rng = np.random.RandomState(0)
    n_calib = int(len(tgt_shared) * CALIB_FRAC)
    perm = rng.permutation(len(tgt_shared))
    calib_idx = perm[:n_calib]
    test_idx = perm[n_calib:]
    test_perts = [tgt_shared[i] for i in test_idx]
    test_V = tV[test_idx]

    # Zero-shot
    pca_zs = PCA(50).fit(sV)
    r_zs = Ridge(alpha=10).fit(pca_zs.transform(sV), sD)
    yp_zs = r_zs.predict(pca_zs.transform(test_V))
    beta_zs, gamma_zs = decompose_anova(test_perts, yp_zs, tgt)
    results[tgt]["zeroshot"] = {"beta": beta_zs, "gamma": gamma_zs}
    print(f"    Zero-shot:            β={beta_zs:.4f}, γ={gamma_zs:.4f}")

    # Target-only 30%
    calib_V = tV[calib_idx]
    calib_D = tD[calib_idx]
    pca_t = PCA(min(50, len(calib_idx) - 1)).fit(calib_V)
    r_t = Ridge(alpha=10).fit(pca_t.transform(calib_V), calib_D)
    yp_t = r_t.predict(pca_t.transform(test_V))
    beta_t, gamma_t = decompose_anova(test_perts, yp_t, tgt)
    results[tgt]["target30"] = {"beta": beta_t, "gamma": gamma_t}
    print(f"    Target perts 30%:     β={beta_t:.4f}, γ={gamma_t:.4f}")

# ── Determine shared y-limits ──
all_vals = []
for cl in CLS:
    for s in ["zeroshot", "target30"]:
        all_vals.append(results[cl][s]["beta"])
        all_vals.append(results[cl][s]["gamma"])
ymin = min(all_vals) - 0.03
ymax = max(all_vals) + 0.03
# Snap to nice bounds
ymin = max(ymin, -0.10)
ymax = min(ymax, 0.50)

# ── Plot ──
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.0, 3.2), sharey=True)

x = np.arange(len(CLS))
offset = 0.15

for ci, cl in enumerate(CLS):
    zs_b = results[cl]["zeroshot"]["beta"]
    t30_b = results[cl]["target30"]["beta"]
    zs_g = results[cl]["zeroshot"]["gamma"]
    t30_g = results[cl]["target30"]["gamma"]

    # Left: β (teal tones)
    ax_left.plot([x[ci] - offset, x[ci] + offset], [zs_b, t30_b],
                 color=C_LINE, linewidth=1.5, zorder=1)
    ax_left.scatter(x[ci] - offset, zs_b, s=50, marker="o", color=C_ZEROSHOT,
                    edgecolor="white", linewidth=0.5, zorder=3,
                    label="Zero-shot" if ci == 0 else None)
    ax_left.scatter(x[ci] + offset, t30_b, s=50, marker="o", color=C_BETA,
                    edgecolor="white", linewidth=0.5, zorder=3,
                    label="Target perturbations 30%" if ci == 0 else None)

    # Right: γ (terracotta tones)
    ax_right.plot([x[ci] - offset, x[ci] + offset], [zs_g, t30_g],
                  color=C_LINE, linewidth=1.5, zorder=1)
    ax_right.scatter(x[ci] - offset, zs_g, s=50, marker="o", color=C_ZEROSHOT,
                     edgecolor="white", linewidth=0.5, zorder=3)
    ax_right.scatter(x[ci] + offset, t30_g, s=50, marker="o", color=C_GAMMA,
                     edgecolor="white", linewidth=0.5, zorder=3)

# Shared y-limits
ax_left.set_ylim(ymin, ymax)

# Left panel
ax_left.axhline(y=0, color=C_TEXT, linewidth=0.8, alpha=0.5, zorder=1)
ax_left.set_xticks(x)
ax_left.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                        rotation=45, ha="right", rotation_mode="anchor")
ax_left.set_ylabel("corr(prediction, component)", fontsize=10, color=C_TEXT)
ax_left.set_title("Conserved component β$_p$(g)", fontsize=10, color=C_TEXT, pad=6)
ax_left.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax_left.xaxis.grid(False)
ax_left.set_axisbelow(True)
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)

# Right panel
ax_right.axhline(y=0, color=C_TEXT, linewidth=0.8, alpha=0.5, zorder=1)
ax_right.set_xticks(x)
ax_right.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                         rotation=45, ha="right", rotation_mode="anchor")
ax_right.set_title("Target interaction γ(c,p,g)", fontsize=10, color=C_TEXT, pad=6)
ax_right.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax_right.xaxis.grid(False)
ax_right.set_axisbelow(True)
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

# Legend below
handles, labels_leg = ax_left.get_legend_handles_labels()
fig.legend(handles, labels_leg, frameon=False, fontsize=8, loc="lower center",
           bbox_to_anchor=(0.5, -0.05), ncol=2, columnspacing=2.0)

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT / "fig4d_gamma_reveal.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4d_gamma_reveal.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print summary ──
print(f"\n{'='*70}")
print("Ridge calibration: β and γ recovery")
print(f"{'='*70}")
print(f"{'Target':<8} {'Setting':<25} {'corr(δ̂,β)':>12} {'corr(δ̂,γ)':>12}")
print("-" * 60)
for cl in CLS:
    for setting, label in [("zeroshot", "Zero-shot"),
                           ("target30", "Target perturbations 30%")]:
        r = results[cl][setting]
        print(f"{CL_LABELS[cl]:<8} {label:<25} {r['beta']:>12.4f} {r['gamma']:>12.4f}")

print(f"\n{'='*70}")
print("Key finding: Δβ vs Δγ (target 30% minus zero-shot)")
print(f"{'='*70}")
print(f"{'Target':<8} {'Δβ':>10} {'Δγ':>10}")
print("-" * 30)
for cl in CLS:
    db = results[cl]["target30"]["beta"] - results[cl]["zeroshot"]["beta"]
    dg = results[cl]["target30"]["gamma"] - results[cl]["zeroshot"]["gamma"]
    print(f"{CL_LABELS[cl]:<8} {db:>+10.4f} {dg:>+10.4f}")

# Save
save_data = {}
for cl in CLS:
    save_data[cl] = {
        setting: {k: float(v) for k, v in results[cl][setting].items()}
        for setting in ["zeroshot", "target30"]
    }

with open(OUT / "fig4d_gamma_reveal_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")