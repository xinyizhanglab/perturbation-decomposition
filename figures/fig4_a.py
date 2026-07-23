"""
Figure 4a: Within-CL models recover template more strongly than residual.
Grouped dot plot: corr(α̂,α) and corr(ε̂,ε) for Ridge, MLP, and MORPH.

Run: python3 fig4_within_decomp.py
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
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

N_MLP_SEEDS = 3
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

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

def decompose_predictions(D_te, yp, Tn):
    """Compute template and residual recovery correlations."""
    true_alpha = D_te @ Tn
    pred_alpha = yp @ Tn
    true_resid = D_te - np.outer(true_alpha, Tn)
    pred_resid = yp - np.outer(pred_alpha, Tn)
    alpha_corr = corr(true_alpha, pred_alpha)
    eps_corrs = [corr(pred_resid[i], true_resid[i]) for i in range(len(D_te))]
    return alpha_corr, float(np.mean(eps_corrs))

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

# ── Compute ──
results = {
    "Ridge": {cl: {"alpha": [], "eps": []} for cl in CLS},
    "MLP":   {cl: {"alpha": [], "eps": []} for cl in CLS},
    "MORPH": {cl: {"alpha": [], "eps": []} for cl in CLS},
}

for cl in CLS:
    print(f"\n  {cl.upper()}:")
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)
    si = json.load(open(BENCH / "data" / cl / "split_info.json"))

    deltas = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5:
            deltas[pn] = X[genes == pn].mean(0) - ctrl_mean

    # For Ridge/MLP: need DepMap
    tr = [p for p in si["train_perts"] if p in deltas and get_emb(p) is not None]
    te = [p for p in si["test_perts"] if p in deltas and get_emb(p) is not None]
    V_tr = np.array([get_emb(p) for p in tr])
    D_tr = np.array([deltas[p] for p in tr])
    V_te = np.array([get_emb(p) for p in te])
    D_te = np.array([deltas[p] for p in te])

    T = D_tr.mean(0)
    Tn = T / (np.linalg.norm(T) + 1e-30)

    pca = PCA(50).fit(V_tr)
    V_tr50 = pca.transform(V_tr)
    V_te50 = pca.transform(V_te)

    # Ridge (deterministic)
    r = Ridge(alpha=10).fit(V_tr50, D_tr)
    yp_r = r.predict(V_te50)
    ac, ec = decompose_predictions(D_te, yp_r, Tn)
    results["Ridge"][cl]["alpha"].append(ac)
    results["Ridge"][cl]["eps"].append(ec)
    print(f"    Ridge: α={ac:.4f}, ε={ec:.4f}")

    # MLP (multiple seeds)
    for seed in range(N_MLP_SEEDS):
        m = train_mlp(V_tr50, D_tr, seed=seed)
        with torch.no_grad():
            yp_m = m(torch.tensor(V_te50, dtype=torch.float32, device=dev)).cpu().numpy()
        ac, ec = decompose_predictions(D_te, yp_m, Tn)
        results["MLP"][cl]["alpha"].append(ac)
        results["MLP"][cl]["eps"].append(ec)
        del m; torch.cuda.empty_cache()
    print(f"    MLP:   α={np.mean(results['MLP'][cl]['alpha']):.4f}±{np.std(results['MLP'][cl]['alpha']):.4f}, "
          f"ε={np.mean(results['MLP'][cl]['eps']):.4f}±{np.std(results['MLP'][cl]['eps']):.4f}")

    # MORPH (from disk, up to 5 seeds)
    for seed in range(5):
        morph_f = BENCH / f"runs/MORPH/{DS[cl]}/seed{seed}/predictions.npz"
        if not morph_f.exists():
            continue
        d = np.load(morph_f, allow_pickle=True)
        morph_perts = list(d["test_perts"])
        morph_yp = d["predictions"]
        morph_yt = d["true_deltas"]
        morph_T = d["train_mean"]
        morph_Tn = morph_T / (np.linalg.norm(morph_T) + 1e-30)

        ac, ec = decompose_predictions(morph_yt, morph_yp, morph_Tn)
        results["MORPH"][cl]["alpha"].append(ac)
        results["MORPH"][cl]["eps"].append(ec)

    if results["MORPH"][cl]["alpha"]:
        n_seeds = len(results["MORPH"][cl]["alpha"])
        print(f"    MORPH: α={np.mean(results['MORPH'][cl]['alpha']):.4f}±{np.std(results['MORPH'][cl]['alpha']):.4f}, "
              f"ε={np.mean(results['MORPH'][cl]['eps']):.4f}±{np.std(results['MORPH'][cl]['eps']):.4f} "
              f"({n_seeds} seeds)")
    else:
        print(f"    MORPH: not available")

# ── Plot ──
fig, ax = plt.subplots(figsize=(6.0, 3.5))

x = np.arange(len(CLS))
group_width = 0.75
# 6 items: Ridge-α, Ridge-ε, MLP-α, MLP-ε, MORPH-α, MORPH-ε
n_items = 6
item_width = group_width / n_items
offsets = np.linspace(-group_width / 2 + item_width / 2,
                       group_width / 2 - item_width / 2, n_items)

configs = [
    ("Ridge", "alpha", "s",  C_TEMPLATE, "Ridge: template (α)"),
    ("Ridge", "eps",   "s",  C_RESIDUAL, "Ridge: residual (ε)"),
    ("MLP",   "alpha", "^",  C_TEMPLATE, "MLP: template (α)"),
    ("MLP",   "eps",   "^",  C_RESIDUAL, "MLP: residual (ε)"),
    ("MORPH", "alpha", "D",  C_TEMPLATE, "MORPH: template (α)"),
    ("MORPH", "eps",   "D",  C_RESIDUAL, "MORPH: residual (ε)"),
]

for ci, (model, comp, marker, color, label) in enumerate(configs):
    for xi, cl in enumerate(CLS):
        vals = results[model][cl][comp]
        if not vals:
            continue
        mean_v = np.mean(vals)

        if len(vals) > 1:
            std_v = np.std(vals)
            ax.errorbar(x[xi] + offsets[ci], mean_v, yerr=std_v,
                        fmt="none", ecolor=color, elinewidth=0.8,
                        capsize=1.5, capthick=0.6, zorder=2, alpha=0.7)

        # Use lighter fill for MORPH to distinguish from Ridge/MLP
        if model == "MORPH":
            ax.scatter(x[xi] + offsets[ci], mean_v, s=35, marker=marker,
                       facecolors=color, edgecolor="white", linewidth=0.5,
                       zorder=3, alpha=0.8,
                       label=label if xi == 0 else None)
        else:
            ax.scatter(x[xi] + offsets[ci], mean_v, s=40, marker=marker,
                       color=color, edgecolor="white", linewidth=0.5,
                       zorder=3, label=label if xi == 0 else None)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Cell line", fontsize=10, color=C_TEXT)
ax.set_ylabel("Correlation with ground truth", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend outside
handles, labels_leg = ax.get_legend_handles_labels()
seen = set(); uh, ul = [], []
for h, l in zip(handles, labels_leg):
    if l not in seen:
        seen.add(l)
        uh.append(h)
        ul.append(l)
ax.legend(uh, ul, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.5)

plt.tight_layout(rect=[0, 0, 0.68, 1])
fig.savefig(OUT / "fig4a_within_decomp.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4a_within_decomp.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print summary ──
print(f"\n{'='*70}")
print("Within-CL prediction attribution")
print(f"{'='*70}")
print(f"{'Model':<8} {'CL':<8} {'corr(α̂,α)':>14} {'corr(ε̂,ε)':>14}")
print("-" * 48)
for model in ["Ridge", "MLP", "MORPH"]:
    for cl in CLS:
        a_vals = results[model][cl]["alpha"]
        e_vals = results[model][cl]["eps"]
        if not a_vals:
            print(f"{model:<8} {CL_LABELS[cl]:<8} {'—':>14} {'—':>14}")
            continue
        if len(a_vals) > 1:
            print(f"{model:<8} {CL_LABELS[cl]:<8} {np.mean(a_vals):.3f}±{np.std(a_vals):.3f}"
                  f"   {np.mean(e_vals):.3f}±{np.std(e_vals):.3f}")
        else:
            print(f"{model:<8} {CL_LABELS[cl]:<8} {a_vals[0]:>14.4f} {e_vals[0]:>14.4f}")

# Save
save_data = {}
for model in ["Ridge", "MLP", "MORPH"]:
    save_data[model] = {}
    for cl in CLS:
        save_data[model][cl] = {
            "alpha": [float(v) for v in results[model][cl]["alpha"]],
            "eps": [float(v) for v in results[model][cl]["eps"]],
            "alpha_mean": float(np.mean(results[model][cl]["alpha"])) if results[model][cl]["alpha"] else None,
            "eps_mean": float(np.mean(results[model][cl]["eps"])) if results[model][cl]["eps"] else None,
        }

with open(OUT / "fig4a_within_decomp_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")