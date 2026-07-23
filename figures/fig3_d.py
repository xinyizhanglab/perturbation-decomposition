"""
Figure 3d: Target response landscape determines recoverable signal.
Scatter: residual effective rank vs MLP cross-CL pref.
Two markers per CL: seen perts (circle) and double-unseen (triangle).

Run: python3 fig3_target_landscape.py
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
CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

DU_FRAC = 0.3
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

def pref(yt, yp, T):
    return np.mean([corr(yt[i] - T, yp[i] - T) for i in range(len(yt))])

def effective_rank(M):
    n_comp = min(100, M.shape[0] - 1, M.shape[1])
    pca = PCA(n_components=n_comp).fit(M)
    ev = pca.explained_variance_
    ev = ev[ev > 1e-12]
    p = ev / ev.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

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

# ── Load data and compute target properties ──
print("Loading data and computing target properties...")
cl_data = {}
target_props = {}

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

    D = np.array(list(deltas.values()))

    # Full response effective rank
    erank_full = effective_rank(D)

    # Template removal → residual effective rank
    T = D.mean(0)
    Tn = T / (np.linalg.norm(T) + 1e-30)
    alpha = D @ Tn
    R = D - np.outer(alpha, Tn)
    erank_resid = effective_rank(R)

    # Template variance
    template_var = np.mean(alpha ** 2)
    total_var = np.mean(np.sum(D ** 2, axis=1))
    template_pct = template_var / total_var * 100

    target_props[cl] = {
        "erank_full": erank_full,
        "erank_resid": erank_resid,
        "template_pct": template_pct,
    }
    print(f"  {cl}: erank_full={erank_full:.1f}, erank_resid={erank_resid:.1f}, "
          f"template={template_pct:.1f}%, {len(deltas)} perts")

# ── Compute MLP cross-CL performance ──
print("\nComputing MLP 3→1 predictions...")
pref_seen = {}
pref_du = {}

for tgt in CLS:
    print(f"\n  Target: {CL_LABELS[tgt]}")
    srcs = [c for c in CLS if c != tgt]

    src_all_perts = set()
    for c in srcs:
        src_all_perts |= set(cl_data[c].keys())
    shared = sorted(set(cl_data[tgt].keys()) & src_all_perts)

    # Double-unseen split
    rng = np.random.RandomState(0)
    n_du = int(len(shared) * DU_FRAC)
    perm = rng.permutation(len(shared))
    du_perts = sorted([shared[i] for i in perm[:n_du]])
    du_set = set(du_perts)

    # ── 3→1 seen ──
    sV, sD = [], []
    for c in srcs:
        for pn, d in cl_data[c].items():
            sV.append(get_emb(pn)); sD.append(d)
    sV = np.array(sV); sD = np.array(sD)

    tps = sorted(cl_data[tgt].keys())
    tV = np.array([get_emb(p) for p in tps])
    tD = np.array([cl_data[tgt][p] for p in tps])
    T_tgt = tD.mean(0)

    pca = PCA(50).fit(sV)

    seen_vals = []
    for seed in range(3):
        m = train_mlp(pca.transform(sV), sD, seed=seed)
        with torch.no_grad():
            yp = m(torch.tensor(pca.transform(tV), dtype=torch.float32, device=dev)).cpu().numpy()
        seen_vals.append(pref(tD, yp, T_tgt))
        del m; torch.cuda.empty_cache()
    pref_seen[tgt] = float(np.mean(seen_vals))
    print(f"    Seen:   {pref_seen[tgt]:.4f}")

    # ── Double-unseen ──
    sV_du, sD_du = [], []
    for c in srcs:
        for pn, d in cl_data[c].items():
            if pn not in du_set:
                sV_du.append(get_emb(pn)); sD_du.append(d)
    sV_du = np.array(sV_du); sD_du = np.array(sD_du)

    duV = np.array([get_emb(p) for p in du_perts])
    duD = np.array([cl_data[tgt][p] for p in du_perts])
    T_du = duD.mean(0)

    pca_du = PCA(50).fit(sV_du)

    du_vals = []
    for seed in range(3):
        m = train_mlp(pca_du.transform(sV_du), sD_du, seed=seed)
        with torch.no_grad():
            yp = m(torch.tensor(pca_du.transform(duV), dtype=torch.float32, device=dev)).cpu().numpy()
        du_vals.append(pref(duD, yp, T_du))
        del m; torch.cuda.empty_cache()
    pref_du[tgt] = float(np.mean(du_vals))
    print(f"    DU:     {pref_du[tgt]:.4f}")

# ── Plot ──
fig, ax = plt.subplots(figsize=(4.0, 3.5))

for cl in CLS:
    x_val = target_props[cl]["erank_resid"]

    # Seen perts (circle)
    ax.scatter(x_val, pref_seen[cl], s=60, marker="o",
               color=CL_COLORS[cl], edgecolor="white", linewidth=0.8,
               zorder=3)

    # Double-unseen (triangle)
    ax.scatter(x_val, pref_du[cl], s=60, marker="^",
               color=CL_COLORS[cl], edgecolor="white", linewidth=0.8,
               zorder=3)

    # Connecting line
    ax.plot([x_val, x_val], [pref_seen[cl], pref_du[cl]],
            color=CL_COLORS[cl], linewidth=1.0, alpha=0.5, zorder=2)

    # Label
    ax.annotate(CL_LABELS[cl], (x_val, pref_seen[cl]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=C_TEXT, ha="left", va="bottom")

ax.set_xlabel("Residual effective rank", fontsize=10, color=C_TEXT)
ax.set_ylabel("MLP perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend for marker shapes
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_TEXT,
           markeredgecolor="white", markersize=7, label="3→1 (seen perts)"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor=C_TEXT,
           markeredgecolor="white", markersize=7, label="Double unseen"),
]
ax.legend(handles=legend_elements, frameon=False, fontsize=8,
          loc="lower right")

plt.tight_layout()
fig.savefig(OUT / "fig3d_target_landscape.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3d_target_landscape.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print summary ──
print(f"\n{'='*75}")
print("Target landscape vs MLP cross-CL performance")
print(f"{'='*75}")
print(f"{'CL':<8} {'Erank full':>11} {'Erank resid':>12} {'Template%':>10} "
      f"{'Seen pref':>10} {'DU pref':>10}")
print("-" * 65)
for cl in CLS:
    p = target_props[cl]
    print(f"{CL_LABELS[cl]:<8} {p['erank_full']:>11.1f} {p['erank_resid']:>12.1f} "
          f"{p['template_pct']:>9.1f}% {pref_seen[cl]:>10.4f} {pref_du[cl]:>10.4f}")

# Save
save_data = {
    "target_properties": {cl: {k: float(v) for k, v in target_props[cl].items()} for cl in CLS},
    "pref_seen": {cl: float(pref_seen[cl]) for cl in CLS},
    "pref_double_unseen": {cl: float(pref_du[cl]) for cl in CLS},
}
with open(OUT / "fig3d_target_landscape_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")