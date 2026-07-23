"""
Figure 4: Cross-CL prediction attribution — template vs residual recovery.

For each target CL, grouped dots showing corr(α̂,α) and corr(ε̂,ε):
  Ridge 1→1, Ridge 3→1, MLP 1→1, MLP 3→1, MORPH, STATE

X-axis: 4 target cell lines.
Two colors: template (α) and residual (ε).

Run: python3 fig4_crosscl_decomp.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.linear_model import Ridge as RidgeModel
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
MORPH_SOURCE = {"k562": "rpe1", "rpe1": "k562", "hepg2": "jurkat", "jurkat": "hepg2"}

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


def decompose_predictions(D_te, yp, Tn):
    """Template and residual recovery correlations."""
    true_alpha = D_te @ Tn
    pred_alpha = yp @ Tn
    true_resid = D_te - np.outer(true_alpha, Tn)
    pred_resid = yp - np.outer(pred_alpha, Tn)
    alpha_corr = corr(true_alpha, pred_alpha)
    eps_corrs = [corr(pred_resid[i], true_resid[i]) for i in range(len(D_te))]
    return alpha_corr, float(np.mean(eps_corrs))


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


# ── Load all CL data ──
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

# ── Models to evaluate ──
MODEL_ORDER = ["Ridge 1→1", "Ridge 3→1", "MLP 1→1", "MLP 3→1", "MORPH", "STATE"]
MODEL_MARKERS = {
    "Ridge 1→1": "s", "Ridge 3→1": "s",
    "MLP 1→1": "^", "MLP 3→1": "^",
    "MORPH": "D", "STATE": "o",
}

results = {m: {cl: {"alpha": [], "eps": []} for cl in CLS} for m in MODEL_ORDER}

for tgt in CLS:
    print(f"\n  Target: {CL_LABELS[tgt]}")
    tgt_deltas = cl_data[tgt]
    tgt_perts = sorted(tgt_deltas.keys())
    tV = np.array([get_emb(p) for p in tgt_perts])
    tD = np.array([tgt_deltas[p] for p in tgt_perts])
    T_tgt = tD.mean(0)
    Tn = T_tgt / (np.linalg.norm(T_tgt) + 1e-30)

    # ── Ridge/MLP 1→1 ──
    src_1 = MORPH_SOURCE[tgt]
    sps_1 = sorted(cl_data[src_1].keys())
    sV_1 = np.array([get_emb(p) for p in sps_1])
    sD_1 = np.array([cl_data[src_1][p] for p in sps_1])
    pca_1 = PCA(50).fit(sV_1)

    # Ridge 1→1
    ridge_1 = RidgeModel(alpha=10).fit(pca_1.transform(sV_1), sD_1)
    yp = ridge_1.predict(pca_1.transform(tV))
    ac, ec = decompose_predictions(tD, yp, Tn)
    results["Ridge 1→1"][tgt]["alpha"].append(ac)
    results["Ridge 1→1"][tgt]["eps"].append(ec)
    print(f"    Ridge 1→1: α={ac:.4f}, ε={ec:.4f}")

    # MLP 1→1
    for seed in range(N_MLP_SEEDS):
        m = train_mlp(pca_1.transform(sV_1), sD_1, seed=seed)
        with torch.no_grad():
            yp = m(torch.tensor(pca_1.transform(tV), dtype=torch.float32, device=dev)).cpu().numpy()
        ac, ec = decompose_predictions(tD, yp, Tn)
        results["MLP 1→1"][tgt]["alpha"].append(ac)
        results["MLP 1→1"][tgt]["eps"].append(ec)
        del m; torch.cuda.empty_cache()
    print(f"    MLP 1→1:   α={np.mean(results['MLP 1→1'][tgt]['alpha']):.4f}, "
          f"ε={np.mean(results['MLP 1→1'][tgt]['eps']):.4f}")

    # ── Ridge/MLP 3→1 ──
    srcs_3 = [c for c in CLS if c != tgt]
    sV_3, sD_3 = [], []
    for c in srcs_3:
        for pn, d in cl_data[c].items():
            sV_3.append(get_emb(pn)); sD_3.append(d)
    sV_3 = np.array(sV_3); sD_3 = np.array(sD_3)
    pca_3 = PCA(50).fit(sV_3)

    # Ridge 3→1
    ridge_3 = RidgeModel(alpha=10).fit(pca_3.transform(sV_3), sD_3)
    yp = ridge_3.predict(pca_3.transform(tV))
    ac, ec = decompose_predictions(tD, yp, Tn)
    results["Ridge 3→1"][tgt]["alpha"].append(ac)
    results["Ridge 3→1"][tgt]["eps"].append(ec)
    print(f"    Ridge 3→1: α={ac:.4f}, ε={ec:.4f}")

    # MLP 3→1
    for seed in range(N_MLP_SEEDS):
        m = train_mlp(pca_3.transform(sV_3), sD_3, seed=seed)
        with torch.no_grad():
            yp = m(torch.tensor(pca_3.transform(tV), dtype=torch.float32, device=dev)).cpu().numpy()
        ac, ec = decompose_predictions(tD, yp, Tn)
        results["MLP 3→1"][tgt]["alpha"].append(ac)
        results["MLP 3→1"][tgt]["eps"].append(ec)
        del m; torch.cuda.empty_cache()
    print(f"    MLP 3→1:   α={np.mean(results['MLP 3→1'][tgt]['alpha']):.4f}, "
          f"ε={np.mean(results['MLP 3→1'][tgt]['eps']):.4f}")

    # ── MORPH (1→1 zeroshot) ──
    morph_pair = f"{src_1}_to_{tgt}"
    for seed in range(5):
        f = BENCH / f"morph_crosscl/runs/{morph_pair}/seed{seed}/predictions_zeroshot.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True)
            morph_T = d["train_mean"]
            morph_Tn = morph_T / (np.linalg.norm(morph_T) + 1e-30)
            ac, ec = decompose_predictions(d["true_deltas"], d["predictions"], morph_Tn)
            results["MORPH"][tgt]["alpha"].append(ac)
            results["MORPH"][tgt]["eps"].append(ec)
    if results["MORPH"][tgt]["alpha"]:
        print(f"    MORPH:     α={np.mean(results['MORPH'][tgt]['alpha']):.4f}, "
              f"ε={np.mean(results['MORPH'][tgt]['eps']):.4f}")

    # ── STATE (zeroshot) ──
    for seed in range(5):
        f = BENCH / f"runs/STATE_zeroshot/{tgt}/seed{seed}/predictions.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True)
            st_T = d["train_mean"]
            st_Tn = st_T / (np.linalg.norm(st_T) + 1e-30)
            ac, ec = decompose_predictions(d["true_deltas"], d["predictions"], st_Tn)
            results["STATE"][tgt]["alpha"].append(ac)
            results["STATE"][tgt]["eps"].append(ec)
    if results["STATE"][tgt]["alpha"]:
        print(f"    STATE:     α={np.mean(results['STATE'][tgt]['alpha']):.4f}, "
              f"ε={np.mean(results['STATE'][tgt]['eps']):.4f}")

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

fig, ax = plt.subplots(figsize=(7.0, 3.8))

x = np.arange(len(CLS))
n_models = len(MODEL_ORDER)
group_width = 0.80
offsets = np.linspace(-group_width / 2, group_width / 2, n_models)

for mi, model in enumerate(MODEL_ORDER):
    marker = MODEL_MARKERS[model]
    for xi, cl in enumerate(CLS):
        a_vals = results[model][cl]["alpha"]
        e_vals = results[model][cl]["eps"]
        if not a_vals:
            continue

        a_mean = np.mean(a_vals)
        e_mean = np.mean(e_vals)
        xpos = x[xi] + offsets[mi]

        # Template dot
        if len(a_vals) > 1:
            ax.errorbar(xpos - 0.02, a_mean, yerr=np.std(a_vals),
                        fmt="none", ecolor=C_TEMPLATE, elinewidth=0.8,
                        capsize=1.5, capthick=0.6, zorder=2, alpha=0.6)
        ax.scatter(xpos - 0.02, a_mean, s=35, marker=marker,
                   color=C_TEMPLATE, edgecolor="white", linewidth=0.5, zorder=3)

        # Residual dot
        if len(e_vals) > 1:
            ax.errorbar(xpos + 0.02, e_mean, yerr=np.std(e_vals),
                        fmt="none", ecolor=C_RESIDUAL, elinewidth=0.8,
                        capsize=1.5, capthick=0.6, zorder=2, alpha=0.6)
        ax.scatter(xpos + 0.02, e_mean, s=35, marker=marker,
                   color=C_RESIDUAL, edgecolor="white", linewidth=0.5, zorder=3)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)

# Model labels along top
for mi, model in enumerate(MODEL_ORDER):
    # Place at mean x position
    ax.text(np.mean(x) + offsets[mi], ax.get_ylim()[1] if ax.get_ylim()[1] else 1.0,
            model, ha="center", va="bottom", fontsize=6.5, color=C_TEXT,
            rotation=45, rotation_mode="anchor")

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT)
ax.set_ylabel("Correlation with ground truth", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend: components + model markers
leg_comp = [
    Line2D([0], [0], marker="o", color=C_TEMPLATE, markersize=6,
           markeredgecolor="white", linewidth=0, label="Template (α)"),
    Line2D([0], [0], marker="o", color=C_RESIDUAL, markersize=6,
           markeredgecolor="white", linewidth=0, label="Residual (ε)"),
]
leg_model = [
    Line2D([0], [0], marker="s", color=C_TEXT, markersize=5,
           markeredgecolor="white", linewidth=0, label="Ridge"),
    Line2D([0], [0], marker="^", color=C_TEXT, markersize=5,
           markeredgecolor="white", linewidth=0, label="MLP"),
    Line2D([0], [0], marker="D", color=C_TEXT, markersize=5,
           markeredgecolor="white", linewidth=0, label="MORPH"),
    Line2D([0], [0], marker="o", color=C_TEXT, markersize=5,
           markeredgecolor="white", linewidth=0, label="STATE"),
]

ax.legend(handles=leg_comp + leg_model, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.6)

plt.tight_layout(rect=[0, 0, 0.80, 0.92])
fig.savefig(OUT / "fig4_crosscl_decomp.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4_crosscl_decomp.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Summary ──
print(f"\n{'='*80}")
print("Cross-CL prediction: template vs residual recovery")
print(f"{'='*80}")
print(f"{'Model':<14} {'CL':<8} {'corr(α̂,α)':>14} {'corr(ε̂,ε)':>14}")
print("-" * 52)
for model in MODEL_ORDER:
    for cl in CLS:
        a = results[model][cl]["alpha"]
        e = results[model][cl]["eps"]
        if not a:
            print(f"{model:<14} {CL_LABELS[cl]:<8} {'—':>14} {'—':>14}")
        elif len(a) > 1:
            print(f"{model:<14} {CL_LABELS[cl]:<8} "
                  f"{np.mean(a):.3f}±{np.std(a):.3f}   "
                  f"{np.mean(e):.3f}±{np.std(e):.3f}")
        else:
            print(f"{model:<14} {CL_LABELS[cl]:<8} {a[0]:>14.4f} {e[0]:>14.4f}")

# Save
save_data = {}
for model in MODEL_ORDER:
    save_data[model] = {}
    for cl in CLS:
        save_data[model][cl] = {
            "alpha": [float(v) for v in results[model][cl]["alpha"]],
            "eps": [float(v) for v in results[model][cl]["eps"]],
        }
with open(OUT / "fig4_crosscl_decomp_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")