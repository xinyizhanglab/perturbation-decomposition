"""
Figure 2d: Predictive information is distributed across lower-variance DepMap axes.
Line plot: Ridge pref as a function of DepMap PCA rank.

Run: python3 fig2_rank_ablation.py
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
OUT = SCRIPT_DIR / "figure2"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

RANKS = [1, 5, 10, 20, 50, 100, 200]

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

def pref(yt, yp, T):
    return np.mean([corr(yt[i] - T, yp[i] - T) for i in range(len(yt))])

# ── Load DepMap embeddings ──
emb_dict = {}
for ds in DS.values():
    ep = MORPH_BASE / f"outputs/{ds}/pert_embeddings.pkl"
    if ep.exists():
        with open(ep, "rb") as f:
            emb_dict.update(pickle.load(f))

def get_emb(p):
    for k in [p + "+ctrl", p]:
        if k in emb_dict:
            return emb_dict[k].astype(np.float32)
    return None

# ── Compute rank ablation ──
results = {}

for cl in CLS:
    print(f"\n  {cl.upper()}:")
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)
    si = json.load(open(BENCH / "data" / cl / "split_info.json"))

    deltas = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5 and get_emb(pn) is not None:
            deltas[pn] = X[genes == pn].mean(0) - ctrl_mean

    tr = [p for p in si["train_perts"] if p in deltas]
    te = [p for p in si["test_perts"] if p in deltas]
    V_tr = np.array([get_emb(p) for p in tr])
    D_tr = np.array([deltas[p] for p in tr])
    V_te = np.array([get_emb(p) for p in te])
    D_te = np.array([deltas[p] for p in te])
    T_tr = D_tr.mean(0)

    d_full = V_tr.shape[1]

    # Full PCA for max rank
    max_rank = min(max(RANKS), V_tr.shape[0] - 1, d_full)
    pca_full = PCA(n_components=max_rank).fit(V_tr)

    cl_results = {}

    # PCA-truncated Ridge at each rank
    for r in RANKS:
        r_use = min(r, max_rank)
        V_tr_r = pca_full.transform(V_tr)[:, :r_use]
        V_te_r = pca_full.transform(V_te)[:, :r_use]
        ridge = Ridge(alpha=10).fit(V_tr_r, D_tr)
        p_val = pref(D_te, ridge.predict(V_te_r), T_tr)
        cl_results[r] = float(p_val)
        print(f"    rank={r:>4}: pref={p_val:.4f}")

    # Full rank (raw, no PCA)
    ridge_raw = Ridge(alpha=10).fit(V_tr, D_tr)
    p_raw = pref(D_te, ridge_raw.predict(V_te), T_tr)
    cl_results["full"] = float(p_raw)
    print(f"    full ({d_full}): pref={p_raw:.4f}")

    results[cl] = cl_results

# ── Plot ──
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

fig, ax = plt.subplots(figsize=(4.5, 3.2))

# X positions: ranks + full
x_ticks = RANKS + [RANKS[-1] * 1.5]  # extra spacing for "full"
x_labels = [str(r) for r in RANKS] + ["Full"]

for cl in CLS:
    y_vals = [results[cl][r] for r in RANKS] + [results[cl]["full"]]
    ax.plot(range(len(x_ticks)), y_vals, color=CL_COLORS[cl], linewidth=1.5,
            marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.5,
            label=CL_LABELS[cl], zorder=2)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)

ax.set_xticks(range(len(x_ticks)))
ax.set_xticklabels(x_labels, fontsize=9, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("DepMap PCA rank", fontsize=10, color=C_TEXT)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, fontsize=8, loc="center left",
          bbox_to_anchor=(1.02, 0.5), labelspacing=0.6)

plt.tight_layout(rect=[0, 0, 0.80, 1])
fig.savefig(OUT / "fig2d_rank_ablation.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2d_rank_ablation.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print ──
print(f"\n{'='*70}")
print("DepMap rank ablation (Ridge pref)")
print(f"{'='*70}")
header = f"{'Rank':<8}" + "".join(f"  {CL_LABELS[cl]:>10}" for cl in CLS)
print(header)
print("-" * 55)
for r in RANKS:
    line = f"{r:<8}"
    for cl in CLS:
        line += f"  {results[cl][r]:>10.4f}"
    print(line)
line = f"{'Full':<8}"
for cl in CLS:
    line += f"  {results[cl]['full']:>10.4f}"
print(line)

with open(OUT / "fig2d_rank_ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT}")