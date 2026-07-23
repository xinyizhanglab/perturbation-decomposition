"""
Figure 4b: Cross-CL zero-shot prediction recovers β but not γ.
Two aligned subpanels: β recovery (left) and γ recovery (right).
Ridge and MLP, 3→1 zero-shot.

Run: python3 fig4_crosscl_decomp.py
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
OUT = SCRIPT_DIR / "figure4"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}

C_RIDGE = "#149E9A"
C_MLP = "#0D7377"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_CONSERVED = "#6F5CC4"
C_INTERACTION = "#D26A3A"

N_MLP_SEEDS = 3
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

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

print(f"Shared perts for ANOVA: {len(shared)}")

# ── Compute cross-CL 3→1 decomposition ──
# results[model][tgt] = {"beta": [vals], "gamma": [vals]}
results = {
    "Ridge": {cl: {"beta": [], "gamma": []} for cl in CLS},
    "MLP":   {cl: {"beta": [], "gamma": []} for cl in CLS},
}

for tgt in CLS:
    print(f"\n  Target: {CL_LABELS[tgt]}")
    c_tgt = cl_idx[tgt]
    srcs = [c for c in CLS if c != tgt]

    # Source training data
    sV, sD = [], []
    for c in srcs:
        for pn, d in cl_data[c].items():
            sV.append(get_emb(pn)); sD.append(d)
    sV = np.array(sV); sD = np.array(sD)

    # Target: shared perts only (for ANOVA decomposition)
    tgt_shared = [p for p in shared if p in cl_data[tgt]]
    tV = np.array([get_emb(p) for p in tgt_shared])
    tD = np.array([cl_data[tgt][p] for p in tgt_shared])

    pca = PCA(50).fit(sV)

    def decompose_anova(yp, perts):
        """Compute corr(δ̂, β) and corr(δ̂, γ) for predictions."""
        bc, gc = [], []
        for i, p in enumerate(perts):
            if p not in p_idx:
                continue
            pidx = p_idx[p]
            bc.append(corr(yp[i], beta_a[pidx]))
            gc.append(corr(yp[i], gamma_a[c_tgt, pidx]))
        return float(np.mean(bc)), float(np.mean(gc))

    # Ridge (deterministic)
    r = Ridge(alpha=10).fit(pca.transform(sV), sD)
    yp_r = r.predict(pca.transform(tV))
    b, g = decompose_anova(yp_r, tgt_shared)
    results["Ridge"][tgt]["beta"].append(b)
    results["Ridge"][tgt]["gamma"].append(g)
    print(f"    Ridge: β={b:.4f}, γ={g:.4f}")

    # MLP (multiple seeds)
    for seed in range(N_MLP_SEEDS):
        m = train_mlp(pca.transform(sV), sD, seed=seed)
        with torch.no_grad():
            yp_m = m(torch.tensor(pca.transform(tV), dtype=torch.float32, device=dev)).cpu().numpy()
        b, g = decompose_anova(yp_m, tgt_shared)
        results["MLP"][tgt]["beta"].append(b)
        results["MLP"][tgt]["gamma"].append(g)
        del m; torch.cuda.empty_cache()
    print(f"    MLP:   β={np.mean(results['MLP'][tgt]['beta']):.4f}±{np.std(results['MLP'][tgt]['beta']):.4f}, "
          f"γ={np.mean(results['MLP'][tgt]['gamma']):.4f}±{np.std(results['MLP'][tgt]['gamma']):.4f}")

# ── Plot: two aligned subpanels ──
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6.5, 3.2), sharey=True)

y = np.arange(len(CLS))
y_labels = [CL_LABELS[cl] for cl in CLS]
offset = 0.12

model_styles = {
    "Ridge": {"color": C_RIDGE, "marker": "s", "label": "Ridge (DepMap)"},
    "MLP":   {"color": C_MLP,   "marker": "^", "label": "MLP (DepMap)"},
}

# Left panel: β recovery
for mi, (model, style) in enumerate(model_styles.items()):
    off = -offset if mi == 0 else offset
    for ci, cl in enumerate(CLS):
        vals = results[model][cl]["beta"]
        mean_v = np.mean(vals)
        if len(vals) > 1:
            std_v = np.std(vals)
            ax_left.errorbar(mean_v, y[ci] + off, xerr=std_v,
                             fmt="none", ecolor=style["color"], elinewidth=0.8,
                             capsize=1.5, capthick=0.6, zorder=2)
        ax_left.scatter(mean_v, y[ci] + off, s=45, marker=style["marker"],
                        color=style["color"], edgecolor="white", linewidth=0.5,
                        zorder=3, label=style["label"] if ci == 0 else None)

ax_left.axvline(x=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)
ax_left.set_xlabel("corr(δ̂, β)", fontsize=10)
ax_left.set_title("Conserved component\nrecovery", fontsize=10, color=C_TEXT, pad=6)
ax_left.set_yticks(y)
ax_left.set_yticklabels(y_labels, fontsize=10, color=C_TEXT)
ax_left.invert_yaxis()
ax_left.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax_left.yaxis.grid(False)
ax_left.set_axisbelow(True)
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)

# Right panel: γ recovery
for mi, (model, style) in enumerate(model_styles.items()):
    off = -offset if mi == 0 else offset
    for ci, cl in enumerate(CLS):
        vals = results[model][cl]["gamma"]
        mean_v = np.mean(vals)
        if len(vals) > 1:
            std_v = np.std(vals)
            ax_right.errorbar(mean_v, y[ci] + off, xerr=std_v,
                              fmt="none", ecolor=style["color"], elinewidth=0.8,
                              capsize=1.5, capthick=0.6, zorder=2)
        ax_right.scatter(mean_v, y[ci] + off, s=45, marker=style["marker"],
                         color=style["color"], edgecolor="white", linewidth=0.5,
                         zorder=3)

ax_right.axvline(x=0, color=C_TEXT, linewidth=0.8, alpha=0.6, zorder=1)
ax_right.set_xlabel("corr(δ̂, γ)", fontsize=10)
ax_right.set_title("Interaction\nrecovery", fontsize=10, color=C_TEXT, pad=6)
ax_right.invert_yaxis()
ax_right.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax_right.yaxis.grid(False)
ax_right.set_axisbelow(True)
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

# Shared legend below
handles, labels_leg = ax_left.get_legend_handles_labels()
seen = set(); uh, ul = [], []
for h, l in zip(handles, labels_leg):
    if l not in seen:
        seen.add(l)
        uh.append(h)
        ul.append(l)
fig.legend(uh, ul, frameon=False, fontsize=8, loc="lower center",
           bbox_to_anchor=(0.5, -0.05), ncol=2, columnspacing=2.0)

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT / "fig4b_crosscl_decomp.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4b_crosscl_decomp.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print summary ──
print(f"\n{'='*70}")
print("Cross-CL 3→1 zero-shot ANOVA attribution")
print(f"{'='*70}")
print(f"{'Model':<8} {'Target':<8} {'corr(δ̂,β)':>14} {'corr(δ̂,γ)':>14}")
print("-" * 48)
for model in ["Ridge", "MLP"]:
    for cl in CLS:
        b_vals = results[model][cl]["beta"]
        g_vals = results[model][cl]["gamma"]
        if len(b_vals) > 1:
            print(f"{model:<8} {CL_LABELS[cl]:<8} {np.mean(b_vals):.3f}±{np.std(b_vals):.3f}"
                  f"   {np.mean(g_vals):.3f}±{np.std(g_vals):.3f}")
        else:
            print(f"{model:<8} {CL_LABELS[cl]:<8} {b_vals[0]:>14.4f} {g_vals[0]:>14.4f}")

# Save
save_data = {}
for model in ["Ridge", "MLP"]:
    save_data[model] = {}
    for cl in CLS:
        save_data[model][cl] = {
            "beta": [float(v) for v in results[model][cl]["beta"]],
            "gamma": [float(v) for v in results[model][cl]["gamma"]],
            "beta_mean": float(np.mean(results[model][cl]["beta"])),
            "gamma_mean": float(np.mean(results[model][cl]["gamma"])),
        }

with open(OUT / "fig4b_crosscl_decomp_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")