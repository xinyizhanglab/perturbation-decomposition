"""
Figure 3c: Double-unseen prediction — both perturbation and cell line unseen.
Ridge (deterministic) and MLP (5 seeds) with error bars.

Run: python3 fig3_double_unseen.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

MODEL_STYLE = {
    "Ridge": {"color": "#149E9A", "marker": "s", "label": "Ridge (DepMap)"},
    "MLP":   {"color": "#0D7377", "marker": "^", "label": "MLP (DepMap)"},
}

N_MLP_SEEDS = 5
DU_FRAC = 0.3
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

def pref(yt, yp, T):
    return np.mean([corr(yt[i] - T, yp[i] - T) for i in range(len(yt))])

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
            idx = pm[s:s+64]
            loss = nn.MSELoss()(m(Vt[idx]), Dt[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
    m.eval(); return m

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

# ── Load all CL deltas ──
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

# ── Compute double-unseen results ──
results = {model: {} for model in MODEL_STYLE}

for tgt in CLS:
    print(f"\n  Target: {CL_LABELS[tgt]}")
    srcs = [c for c in CLS if c != tgt]

    # Find perturbations shared between source CLs and target
    src_all_perts = set()
    for c in srcs:
        src_all_perts |= set(cl_data[c].keys())
    shared = sorted(set(cl_data[tgt].keys()) & src_all_perts)

    # Hold out DU_FRAC of shared perturbations
    rng = np.random.RandomState(0)
    n_du = int(len(shared) * DU_FRAC)
    perm = rng.permutation(len(shared))
    du_perts = sorted([shared[i] for i in perm[:n_du]])
    du_set = set(du_perts)

    print(f"    Shared: {len(shared)}, Double-unseen: {len(du_perts)}")

    # Source training: exclude double-unseen perts
    sV, sD = [], []
    for c in srcs:
        for pn, d in cl_data[c].items():
            if pn not in du_set:
                sV.append(get_emb(pn))
                sD.append(d)
    sV = np.array(sV)
    sD = np.array(sD)

    # Target test: double-unseen perts only
    duV = np.array([get_emb(p) for p in du_perts])
    duD = np.array([cl_data[tgt][p] for p in du_perts])
    T_du = duD.mean(0)

    # Ridge (deterministic)
    pca = PCA(50).fit(sV)
    r = Ridge(alpha=10).fit(pca.transform(sV), sD)
    p_r = pref(duD, r.predict(pca.transform(duV)), T_du)
    results["Ridge"][tgt] = [p_r]
    print(f"    Ridge: {p_r:.4f}")

    # MLP (5 seeds)
    mlp_vals = []
    for seed in range(N_MLP_SEEDS):
        m = train_mlp(pca.transform(sV), sD, seed=seed)
        with torch.no_grad():
            yp = m(torch.tensor(pca.transform(duV), dtype=torch.float32, device=dev)).cpu().numpy()
        mlp_vals.append(pref(duD, yp, T_du))
        del m; torch.cuda.empty_cache()
    results["MLP"][tgt] = mlp_vals
    print(f"    MLP:   {np.mean(mlp_vals):.4f} ± {np.std(mlp_vals):.4f}")

# ── Plot ──
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.set_title("Cell line and perturbation held-out", fontsize=11, color=C_TEXT, pad=8)

x = np.arange(len(CLS))
n_models = len(MODEL_STYLE)
offsets = np.linspace(-0.12, 0.12, n_models)

for mi, (model, style) in enumerate(MODEL_STYLE.items()):
    for i, cl in enumerate(CLS):
        vals = results[model].get(cl, [0])
        mean_v = np.mean(vals)
        if len(vals) > 1:
            std_v = np.std(vals)
            ax.errorbar(x[i] + offsets[mi], mean_v, yerr=std_v,
                        fmt="none", ecolor=style["color"], elinewidth=1.0,
                        capsize=2, capthick=0.8, zorder=2)
        ax.scatter(x[i] + offsets[mi], mean_v, s=45, marker=style["marker"],
                   color=style["color"], edgecolor="white", linewidth=0.5,
                   zorder=3, label=style["label"] if i == 0 else None)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Target cell line", fontsize=10, color=C_TEXT)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
seen = set(); uh, ul = [], []
for h, l in zip(handles, labels):
    if l not in seen: seen.add(l); uh.append(h); ul.append(l)
ax.legend(uh, ul, frameon=False, fontsize=8,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.8)

plt.tight_layout(rect=[0, 0, 0.76, 1])
fig.savefig(OUT / "fig3c_double_unseen.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3c_double_unseen.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print summary ──
print(f"\n{'='*65}")
print("Double-unseen 3→1 (pref, mean ± std)")
print(f"{'='*65}")
print(f"{'Target':<10}", end="")
for model in MODEL_STYLE:
    print(f"  {model:>18}", end="")
print()
print("-" * 50)
for cl in CLS:
    print(f"{CL_LABELS[cl]:<10}", end="")
    for model in MODEL_STYLE:
        vals = results[model].get(cl, [0])
        if len(vals) > 1:
            print(f"  {np.mean(vals):.3f}±{np.std(vals):.3f}", end="")
        else:
            print(f"  {vals[0]:>18.4f}", end="")
    print()

# Save
save_data = {}
for model in MODEL_STYLE:
    save_data[model] = {}
    for cl in CLS:
        vals = results[model].get(cl, [])
        save_data[model][cl] = {
            "values": [float(v) for v in vals],
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if len(vals) > 1 else None,
        }

with open(OUT / "fig3c_double_unseen_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")