"""
Figure 3: Cross-cell-line zero-shot transfer — all models in one plot.

X-axis: 4 target cell lines
Grouped dots per CL: Ridge 1→1, Ridge 3→1, MLP 1→1, MLP 3→1, MORPH, STATE

Run: python3 fig3_crosscl_transfer.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import torch, torch.nn as nn
import anndata as ad
import pickle, json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
MORPH_BASE = Path("/mnt/storage01/home/amolina/agop-perturbation/morph_pipeline")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure3"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}
MORPH_SOURCE = {"k562": "rpe1", "rpe1": "k562", "hepg2": "jurkat", "jurkat": "hepg2"}

N_MLP_SEEDS = 5
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

MODEL_ORDER = ["Ridge 1→1", "Ridge 3→1", "MLP 1→1", "MLP 3→1", "MORPH", "STATE"]
MODEL_COLORS = {
    "Ridge 1→1": "#149E9A", "Ridge 3→1": "#149E9A",
    "MLP 1→1":   "#0D7377", "MLP 3→1":   "#0D7377",
    "MORPH":     "#6F5CC4",
    "STATE":     "#8A94A6",
}
MODEL_MARKERS = {
    "Ridge 1→1": "s", "Ridge 3→1": "s",
    "MLP 1→1":   "^", "MLP 3→1":   "^",
    "MORPH":     "D",
    "STATE":     "o",
}
# Open markers for 1→1, filled for 3→1
MODEL_FILL = {
    "Ridge 1→1": False, "Ridge 3→1": True,
    "MLP 1→1":   False, "MLP 3→1":   True,
    "MORPH":     True,
    "STATE":     True,
}

dev = "cuda:0" if torch.cuda.is_available() else "cpu"


def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def pref(yt, yp, T):
    return np.mean([corr(yt[i] - T, yp[i] - T) for i in range(len(yt))])


class MLP(nn.Module):
    def __init__(self, d_in, G):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, G))
    def forward(self, v):
        return self.net(v)


def train_mlp(V_tr, D_tr, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(V_tr.shape[1], D_tr.shape[1]).to(dev).float()
    opt = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
    Vt = torch.tensor(V_tr, dtype=torch.float32, device=dev)
    Dt = torch.tensor(D_tr, dtype=torch.float32, device=dev)
    n = len(V_tr); m.train()
    for ep in range(400):
        pm = np.random.permutation(n)
        for s in range(0, n, 64):
            idx = pm[s:s + 64]
            loss = nn.MSELoss()(m(Vt[idx]), Dt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
    m.eval(); return m


# ── Load DepMap ──
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
    print(f"  {CL_LABELS[cl]}: {len(deltas)} perts")
    del a

# ── Compute all models ──
results = {m: {cl: [] for cl in CLS} for m in MODEL_ORDER}

for tgt in CLS:
    print(f"\n  Target: {CL_LABELS[tgt]}")

    tgt_perts = sorted(cl_data[tgt].keys())
    tV = np.array([get_emb(p) for p in tgt_perts])
    tD = np.array([cl_data[tgt][p] for p in tgt_perts])
    T_tgt = tD.mean(0)

    # ── 1→1 source ──
    src_1 = MORPH_SOURCE[tgt]
    sps_1 = sorted(cl_data[src_1].keys())
    sV_1 = np.array([get_emb(p) for p in sps_1])
    sD_1 = np.array([cl_data[src_1][p] for p in sps_1])
    pca_1 = PCA(50).fit(sV_1)

    # Ridge 1→1
    r1 = Ridge(alpha=10).fit(pca_1.transform(sV_1), sD_1)
    results["Ridge 1→1"][tgt] = [pref(tD, r1.predict(pca_1.transform(tV)), T_tgt)]
    print(f"    Ridge 1→1: {results['Ridge 1→1'][tgt][0]:.4f}")

    # MLP 1→1
    for seed in range(N_MLP_SEEDS):
        m = train_mlp(pca_1.transform(sV_1), sD_1, seed=seed)
        with torch.no_grad():
            yp = m(torch.tensor(pca_1.transform(tV), dtype=torch.float32, device=dev)).cpu().numpy()
        results["MLP 1→1"][tgt].append(pref(tD, yp, T_tgt))
        del m; torch.cuda.empty_cache()
    print(f"    MLP 1→1:   {np.mean(results['MLP 1→1'][tgt]):.4f} ± {np.std(results['MLP 1→1'][tgt]):.4f}")

    # ── 3→1 source ──
    srcs_3 = [c for c in CLS if c != tgt]
    sV_3, sD_3 = [], []
    for c in srcs_3:
        for pn, d in cl_data[c].items():
            sV_3.append(get_emb(pn)); sD_3.append(d)
    sV_3 = np.array(sV_3); sD_3 = np.array(sD_3)
    pca_3 = PCA(50).fit(sV_3)

    # Ridge 3→1
    r3 = Ridge(alpha=10).fit(pca_3.transform(sV_3), sD_3)
    results["Ridge 3→1"][tgt] = [pref(tD, r3.predict(pca_3.transform(tV)), T_tgt)]
    print(f"    Ridge 3→1: {results['Ridge 3→1'][tgt][0]:.4f}")

    # MLP 3→1
    for seed in range(N_MLP_SEEDS):
        m = train_mlp(pca_3.transform(sV_3), sD_3, seed=seed)
        with torch.no_grad():
            yp = m(torch.tensor(pca_3.transform(tV), dtype=torch.float32, device=dev)).cpu().numpy()
        results["MLP 3→1"][tgt].append(pref(tD, yp, T_tgt))
        del m; torch.cuda.empty_cache()
    print(f"    MLP 3→1:   {np.mean(results['MLP 3→1'][tgt]):.4f} ± {np.std(results['MLP 3→1'][tgt]):.4f}")

    # ── MORPH (1→1 zeroshot) ──
    morph_pair = f"{src_1}_to_{tgt}"
    for seed in range(5):
        f = BENCH / f"morph_crosscl/runs/{morph_pair}/seed{seed}/predictions_zeroshot.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True)
            results["MORPH"][tgt].append(pref(d["true_deltas"], d["predictions"], d["train_mean"]))
    if results["MORPH"][tgt]:
        print(f"    MORPH:     {np.mean(results['MORPH'][tgt]):.4f} ± {np.std(results['MORPH'][tgt]):.4f} "
              f"({len(results['MORPH'][tgt])} seeds)")
    else:
        print(f"    MORPH:     not found")

    # ── STATE (zeroshot) ──
    for seed in range(5):
        f = BENCH / f"runs/STATE_zeroshot/{tgt}/seed{seed}/predictions.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True)
            results["STATE"][tgt].append(pref(d["true_deltas"], d["predictions"], d["train_mean"]))
    if results["STATE"][tgt]:
        print(f"    STATE:     {np.mean(results['STATE'][tgt]):.4f} ± {np.std(results['STATE'][tgt]):.4f} "
              f"({len(results['STATE'][tgt])} seeds)")
    else:
        print(f"    STATE:     not found")

# ══════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.0, 3.5))

x = np.arange(len(CLS))
n_models = len(MODEL_ORDER)
offsets = np.linspace(-0.38, 0.38, n_models)

for mi, model in enumerate(MODEL_ORDER):
    color = MODEL_COLORS[model]
    marker = MODEL_MARKERS[model]
    filled = MODEL_FILL[model]

    for xi, cl in enumerate(CLS):
        vals = results[model][cl]
        if not vals:
            continue
        mean_v = np.mean(vals)
        xpos = x[xi] + offsets[mi]

        if len(vals) > 1:
            ax.errorbar(xpos, mean_v, yerr=np.std(vals),
                        fmt="none", ecolor=color, elinewidth=1.0,
                        capsize=2, capthick=0.8, zorder=2)

        if filled:
            ax.scatter(xpos, mean_v, s=45, marker=marker,
                       color=color, edgecolor="white", linewidth=0.5, zorder=3)
        else:
            ax.scatter(xpos, mean_v, s=45, marker=marker,
                       facecolors="white", edgecolor=color, linewidth=1.2, zorder=3)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend: open = 1 source, filled = 3 sources / architecture
leg_handles = [
    Line2D([0], [0], marker="s", color="#149E9A", markersize=6,
           markerfacecolor="white", markeredgecolor="#149E9A", markeredgewidth=1.2,
           linewidth=0, label="Ridge (1 cell line in train)"),
    Line2D([0], [0], marker="s", color="#149E9A", markersize=6,
           markeredgecolor="white", markeredgewidth=0.5,
           linewidth=0, label="Ridge (3 cell lines in train)"),
    Line2D([0], [0], marker="^", color="#0D7377", markersize=6,
           markerfacecolor="white", markeredgecolor="#0D7377", markeredgewidth=1.2,
           linewidth=0, label="MLP (1 cell line in train)"),
    Line2D([0], [0], marker="^", color="#0D7377", markersize=6,
           markeredgecolor="white", markeredgewidth=0.5,
           linewidth=0, label="MLP (3 cell lines in train)"),
    Line2D([0], [0], marker="D", color="#6F5CC4", markersize=6,
           markeredgecolor="white", markeredgewidth=0.5,
           linewidth=0, label="MORPH"),
    Line2D([0], [0], marker="o", color="#8A94A6", markersize=6,
           markeredgecolor="white", markeredgewidth=0.5,
           linewidth=0, label="STATE"),
]
ax.legend(handles=leg_handles, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.6)

plt.tight_layout(rect=[0, 0, 0.65, 1])
fig.savefig(OUT / "fig3_crosscl_transfer.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3_crosscl_transfer.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Summary ──
print(f"\n{'='*85}")
print("Cross-CL transfer (pref)")
print(f"{'='*85}")
print(f"{'Target':<8} " + "  ".join(f"{m:>12}" for m in MODEL_ORDER))
print("-" * 85)
for cl in CLS:
    line = f"{CL_LABELS[cl]:<8} "
    for model in MODEL_ORDER:
        vals = results[model][cl]
        if not vals:
            line += f"  {'—':>12}"
        elif len(vals) > 1:
            line += f"  {np.mean(vals):.3f}±{np.std(vals):.3f}"
        else:
            line += f"  {vals[0]:>12.4f}"
    print(line)

# Save
save_data = {}
for model in MODEL_ORDER:
    save_data[model] = {}
    for cl in CLS:
        vals = results[model][cl]
        save_data[model][cl] = {
            "values": [float(v) for v in vals],
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if len(vals) > 1 else None,
        }
with open(OUT / "fig3_crosscl_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")