"""
Supplementary Figure 4: Prediction decomposition and feature ablations.
All panels saved to figure4_supplementary/.

S4a: Feature ablation (within-CL heatmap + cross-CL heatmap)
S4b: Full within-CL prediction decomposition (α, ε, variance split)
S4c: Full cross-CL Ridge/MLP component decomposition
S4d: Full STATE and MORPH component heatmaps
S4e: Pooled vs target-only calibration (β and γ)
S4f: Full metrics table (saved as JSON)

Run: python3 fig4_supp.py
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn
import anndata as ad
import pickle, json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
MORPH_BASE = Path("/mnt/storage01/home/amolina/agop-perturbation/morph_pipeline")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure4_supplementary"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_TEMPLATE = "#3B5BA9"
C_RESIDUAL = "#149E9A"
C_CONSERVED = "#6F5CC4"
C_INTERACTION = "#D26A3A"
C_SOURCE = "#8A94A6"
C_LINE = "#C9CED6"
C_ZERO = "#FFFFFF"

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT,
})

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

def pred_mlp(m, V):
    with torch.no_grad():
        return m(torch.tensor(V, dtype=torch.float32, device=dev)).cpu().numpy()

# ── Load all data ──
print("Loading data...")
cl_full = {}
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X_hvg = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_cells = X_hvg[genes == "non-targeting"]
    ctrl_mean = ctrl_cells.mean(0)
    hvg_names = list(a.var_names[a.var["highly_variable"]])
    gene_to_col = {g: i for i, g in enumerate(hvg_names)}
    si = json.load(open(BENCH / "data" / cl / "split_info.json"))

    deltas = {}
    ctrl_features = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5:
            deltas[pn] = X_hvg[genes == pn].mean(0) - ctrl_mean
            if pn in gene_to_col:
                col = gene_to_col[pn]
                ctrl_features[pn] = np.array([
                    ctrl_cells[:, col].mean(), ctrl_cells[:, col].std(),
                    np.median(ctrl_cells[:, col]), (ctrl_cells[:, col] == 0).mean(),
                    np.percentile(ctrl_cells[:, col], 25),
                    np.percentile(ctrl_cells[:, col], 75),
                ])
            else:
                ctrl_features[pn] = np.zeros(6)

    cl_full[cl] = {
        "deltas": deltas, "ctrl_features": ctrl_features,
        "ctrl_cells": ctrl_cells, "gene_to_col": gene_to_col,
        "train_perts": si["train_perts"], "test_perts": si["test_perts"],
        "X_hvg": X_hvg, "genes": genes,
    }
    print(f"  {cl}: {len(deltas)} perts")

# ── ANOVA on shared perts ──
shared = sorted(set.intersection(*[set(cl_full[cl]["deltas"].keys()) for cl in CLS]))
D_tensor = np.array([[cl_full[cl]["deltas"][p] for p in shared] for cl in CLS])
mu = D_tensor.mean(axis=(0, 1))
alpha_a = D_tensor.mean(axis=1) - mu
beta_a = D_tensor.mean(axis=0) - mu
gamma_a = D_tensor - mu - alpha_a[:, None, :] - beta_a[None, :, :]
cl_idx = {c: i for i, c in enumerate(CLS)}
p_idx = {p: i for i, p in enumerate(shared)}
G = D_tensor.shape[2]
print(f"ANOVA shared perts: {len(shared)}")


# ═══════════════════════════════════════════════
# S4a: Feature ablation heatmaps
# ═══════════════════════════════════════════════
print("\n  S4a: Feature ablation...")

WITHIN_FEATS = ["DepMap PCA50", "Ctrl expression", "Delta columns",
                "Co-perturbation", "Gene-gene corr", "DepMap + ctrl expr"]

within_results = np.full((len(CLS), len(WITHIN_FEATS)), np.nan)

for ci, cl in enumerate(CLS):
    print(f"    {cl}...")
    d = cl_full[cl]
    tr = [p for p in d["train_perts"] if p in d["deltas"] and get_emb(p) is not None]
    te = [p for p in d["test_perts"] if p in d["deltas"] and get_emb(p) is not None]
    D_tr = np.array([d["deltas"][p] for p in tr])
    D_te = np.array([d["deltas"][p] for p in te])
    T = D_tr.mean(0)

    for fi, fname in enumerate(WITHIN_FEATS):
        try:
            if fname == "DepMap PCA50":
                V_tr = np.array([get_emb(p) for p in tr])
                V_te = np.array([get_emb(p) for p in te])
                pca = PCA(50).fit(V_tr)
                r = Ridge(10).fit(pca.transform(V_tr), D_tr)
                within_results[ci, fi] = pref(D_te, r.predict(pca.transform(V_te)), T)

            elif fname == "Ctrl expression":
                tr_f = [p for p in tr if p in d["ctrl_features"]]
                te_f = [p for p in te if p in d["ctrl_features"]]
                if len(tr_f) > 10 and len(te_f) > 5:
                    Ftr = np.array([d["ctrl_features"][p] for p in tr_f])
                    Fte = np.array([d["ctrl_features"][p] for p in te_f])
                    sc = StandardScaler().fit(Ftr)
                    Dtr = np.array([d["deltas"][p] for p in tr_f])
                    Dte = np.array([d["deltas"][p] for p in te_f])
                    r = Ridge(10).fit(sc.transform(Ftr), Dtr)
                    within_results[ci, fi] = pref(Dte, r.predict(sc.transform(Fte)), Dtr.mean(0))

            elif fname == "Delta columns":
                D_all_tr = np.array([d["deltas"][p] for p in tr])
                tr_dc = [p for p in tr if p in d["gene_to_col"]]
                te_dc = [p for p in te if p in d["gene_to_col"]]
                if len(tr_dc) > 10 and len(te_dc) > 5:
                    Ftr = np.array([D_all_tr[:, d["gene_to_col"][p]] for p in tr_dc])
                    Fte = np.array([D_all_tr[:, d["gene_to_col"][p]] for p in te_dc])
                    Dtr = np.array([d["deltas"][p] for p in tr_dc])
                    Dte = np.array([d["deltas"][p] for p in te_dc])
                    pca_dc = PCA(min(50, Ftr.shape[1]-1, Ftr.shape[0]-1)).fit(Ftr)
                    r = Ridge(10).fit(pca_dc.transform(Ftr), Dtr)
                    within_results[ci, fi] = pref(Dte, r.predict(pca_dc.transform(Fte)), Dtr.mean(0))

            elif fname == "Co-perturbation":
                D_all_tr = np.array([d["deltas"][p] for p in tr])
                C = D_all_tr.T @ D_all_tr
                C_n = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                tr_co = [p for p in tr if p in d["gene_to_col"]]
                te_co = [p for p in te if p in d["gene_to_col"]]
                if len(tr_co) > 10 and len(te_co) > 5:
                    Ftr = np.array([C_n[d["gene_to_col"][p]] for p in tr_co])
                    Fte = np.array([C_n[d["gene_to_col"][p]] for p in te_co])
                    Dtr = np.array([d["deltas"][p] for p in tr_co])
                    Dte = np.array([d["deltas"][p] for p in te_co])
                    pca_co = PCA(min(50, Ftr.shape[1]-1, Ftr.shape[0]-1)).fit(Ftr)
                    r = Ridge(10).fit(pca_co.transform(Ftr), Dtr)
                    within_results[ci, fi] = pref(Dte, r.predict(pca_co.transform(Fte)), Dtr.mean(0))

            elif fname == "Gene-gene corr":
                X = cl_full[cl]["X_hvg"]
                corr_mat = np.corrcoef(X.T)
                corr_mat = np.nan_to_num(corr_mat)
                tr_gc = [p for p in tr if p in d["gene_to_col"]]
                te_gc = [p for p in te if p in d["gene_to_col"]]
                if len(tr_gc) > 10 and len(te_gc) > 5:
                    Ftr = np.array([corr_mat[d["gene_to_col"][p]] for p in tr_gc])
                    Fte = np.array([corr_mat[d["gene_to_col"][p]] for p in te_gc])
                    Dtr = np.array([d["deltas"][p] for p in tr_gc])
                    Dte = np.array([d["deltas"][p] for p in te_gc])
                    pca_gc = PCA(min(50, Ftr.shape[1]-1, Ftr.shape[0]-1)).fit(Ftr)
                    r = Ridge(10).fit(pca_gc.transform(Ftr), Dtr)
                    within_results[ci, fi] = pref(Dte, r.predict(pca_gc.transform(Fte)), Dtr.mean(0))

            elif fname == "DepMap + ctrl expr":
                tr_f = [p for p in tr if p in d["ctrl_features"] and get_emb(p) is not None]
                te_f = [p for p in te if p in d["ctrl_features"] and get_emb(p) is not None]
                if len(tr_f) > 10 and len(te_f) > 5:
                    comb_tr = np.hstack([np.array([get_emb(p) for p in tr_f]),
                                         np.array([d["ctrl_features"][p] for p in tr_f])])
                    comb_te = np.hstack([np.array([get_emb(p) for p in te_f]),
                                         np.array([d["ctrl_features"][p] for p in te_f])])
                    pca = PCA(50).fit(comb_tr)
                    Dtr = np.array([d["deltas"][p] for p in tr_f])
                    Dte = np.array([d["deltas"][p] for p in te_f])
                    r = Ridge(10).fit(pca.transform(comb_tr), Dtr)
                    within_results[ci, fi] = pref(Dte, r.predict(pca.transform(comb_te)), Dtr.mean(0))
        except Exception as e:
            print(f"      {fname} failed: {e}")

# Cross-CL feature ablation
CROSS_FEATS = ["DepMap", "Target ctrl", "DepMap + tgt ctrl", "DepMap + src ctrl"]
cross_results = np.full((len(CLS), len(CROSS_FEATS)), np.nan)

for ci, tgt in enumerate(CLS):
    print(f"    cross-CL →{tgt}...")
    srcs = [c for c in CLS if c != tgt]
    sV, sD, sCtrl = [], [], []
    for c in srcs:
        for pn, dd in cl_full[c]["deltas"].items():
            if get_emb(pn) is not None:
                sV.append(get_emb(pn)); sD.append(dd)
                sCtrl.append(cl_full[c]["ctrl_features"].get(pn, np.zeros(6)))
    sV = np.array(sV); sD = np.array(sD); sCtrl = np.array(sCtrl)

    tgt_shared = [p for p in shared if p in cl_full[tgt]["deltas"] and get_emb(p) is not None]
    tV = np.array([get_emb(p) for p in tgt_shared])
    tD = np.array([cl_full[tgt]["deltas"][p] for p in tgt_shared])
    tCtrl = np.array([cl_full[tgt]["ctrl_features"].get(p, np.zeros(6)) for p in tgt_shared])
    T_tgt = tD.mean(0)

    for fi, fname in enumerate(CROSS_FEATS):
        try:
            if fname == "DepMap":
                pca = PCA(50).fit(sV)
                r = Ridge(10).fit(pca.transform(sV), sD)
                cross_results[ci, fi] = pref(tD, r.predict(pca.transform(tV)), T_tgt)
            elif fname == "Target ctrl":
                sc = StandardScaler().fit(sCtrl)
                r = Ridge(10).fit(sc.transform(sCtrl), sD)
                cross_results[ci, fi] = pref(tD, r.predict(sc.transform(tCtrl)), T_tgt)
            elif fname == "DepMap + tgt ctrl":
                comb_tr = np.hstack([sV, sCtrl])
                comb_te = np.hstack([tV, tCtrl])
                pca = PCA(50).fit(comb_tr)
                r = Ridge(10).fit(pca.transform(comb_tr), sD)
                cross_results[ci, fi] = pref(tD, r.predict(pca.transform(comb_te)), T_tgt)
            elif fname == "DepMap + src ctrl":
                avg_ctrl = sCtrl.mean(0)
                tCtrl_proxy = np.tile(avg_ctrl, (len(tgt_shared), 1))
                comb_tr = np.hstack([sV, sCtrl])
                comb_te = np.hstack([tV, tCtrl_proxy])
                pca = PCA(50).fit(comb_tr)
                r = Ridge(10).fit(pca.transform(comb_tr), sD)
                cross_results[ci, fi] = pref(tD, r.predict(pca.transform(comb_te)), T_tgt)
        except Exception as e:
            print(f"      {fname} failed: {e}")

# Plot S4a
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.0, 2.8))
cmap_div = LinearSegmentedColormap.from_list("grey_teal", [C_INTERACTION, C_LINE, "#F3F4F6", "#A8D8D5", C_RESIDUAL])

for ax, mat, col_labels, title in [
    (ax_l, within_results, WITHIN_FEATS, "Within-CL"),
    (ax_r, cross_results, CROSS_FEATS, "Cross-CL 3→1"),
]:
    vmax = np.nanmax(np.abs(mat)) * 1.05
    norm = TwoSlopeNorm(vmin=-vmax * 0.3, vcenter=0, vmax=vmax)
    im = ax.imshow(mat, cmap=cmap_div, norm=norm, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v): ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#999"); continue
            tc = "white" if abs(v) > vmax * 0.5 else C_TEXT
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=tc)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.5, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(CLS)))
    ax.set_yticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9)
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

plt.tight_layout()
fig.savefig(OUT / "s4a_feature_ablation.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4a_feature_ablation.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4a")


# ═══════════════════════════════════════════════
# S4b: Full within-CL prediction decomposition
# ═══════════════════════════════════════════════
print("\n  S4b: Within-CL decomposition...")

within_decomp = {model: {cl: {} for cl in CLS} for model in ["Ridge", "MLP"]}

for cl in CLS:
    print(f"    {cl}...")
    d = cl_full[cl]
    tr = [p for p in d["train_perts"] if p in d["deltas"] and get_emb(p) is not None]
    te = [p for p in d["test_perts"] if p in d["deltas"] and get_emb(p) is not None]
    V_tr = np.array([get_emb(p) for p in tr])
    D_tr = np.array([d["deltas"][p] for p in tr])
    V_te = np.array([get_emb(p) for p in te])
    D_te = np.array([d["deltas"][p] for p in te])
    T = D_tr.mean(0); Tn = T / (np.linalg.norm(T) + 1e-30)
    pca = PCA(50).fit(V_tr)

    for model_name in ["Ridge", "MLP"]:
        if model_name == "Ridge":
            r = Ridge(10).fit(pca.transform(V_tr), D_tr)
            yp = r.predict(pca.transform(V_te))
        else:
            m = train_mlp(pca.transform(V_tr), D_tr)
            yp = pred_mlp(m, pca.transform(V_te))
            del m; torch.cuda.empty_cache()

        true_alpha = D_te @ Tn; pred_alpha = yp @ Tn
        true_resid = D_te - np.outer(true_alpha, Tn)
        pred_resid = yp - np.outer(pred_alpha, Tn)

        alpha_c = corr(true_alpha, pred_alpha)
        eps_c = np.mean([corr(pred_resid[i], true_resid[i]) for i in range(len(te))])
        t_var = np.mean(pred_alpha ** 2)
        e_var = np.mean(np.sum(pred_resid ** 2, axis=1))
        total = t_var + e_var

        within_decomp[model_name][cl] = {
            "alpha_corr": alpha_c, "eps_corr": eps_c,
            "t_frac": t_var / total, "e_frac": e_var / total,
            "pref": pref(D_te, yp, T),
        }

# Plot S4b: three aligned panels
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.0, 2.8))

models = ["Ridge", "MLP"]
x = np.arange(len(CLS))
width = 0.35

# Left: α correlation
for mi, model in enumerate(models):
    vals = [within_decomp[model][cl]["alpha_corr"] for cl in CLS]
    ax1.bar(x + mi * width - width / 2, vals, width, color=C_TEMPLATE if mi == 0 else C_RESIDUAL,
            edgecolor="white", linewidth=0.5, label=model, zorder=2)
ax1.set_xticks(x); ax1.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
ax1.set_ylabel("corr(α̂, α)", fontsize=10)
ax1.set_title("Template coefficient", fontsize=10, color=C_TEXT, pad=6)
ax1.legend(frameon=False, fontsize=8); ax1.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
ax1.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# Middle: ε correlation
for mi, model in enumerate(models):
    vals = [within_decomp[model][cl]["eps_corr"] for cl in CLS]
    ax2.bar(x + mi * width - width / 2, vals, width, color=C_TEMPLATE if mi == 0 else C_RESIDUAL,
            edgecolor="white", linewidth=0.5, label=model, zorder=2)
ax2.set_xticks(x); ax2.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
ax2.set_ylabel("corr(ε̂, ε)", fontsize=10)
ax2.set_title("Residual recovery", fontsize=10, color=C_TEXT, pad=6)
ax2.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
ax2.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# Right: variance attribution stacked bars
bar_width = 0.18
for mi, model in enumerate(models):
    for ci, cl in enumerate(CLS):
        tf = within_decomp[model][cl]["t_frac"]
        ef = within_decomp[model][cl]["e_frac"]
        xp = ci + mi * bar_width * 2.5 - bar_width
        ax3.bar(xp, tf, bar_width * 2, color=C_TEMPLATE, edgecolor="white", linewidth=0.3, zorder=2)
        ax3.bar(xp, ef, bar_width * 2, bottom=tf, color=C_RESIDUAL, edgecolor="white", linewidth=0.3, zorder=2)
        ax3.text(xp, 0.5, model[0], ha="center", va="center", fontsize=6, color="white")

ax3.set_xticks(x); ax3.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
ax3.set_ylabel("Predicted variance fraction", fontsize=10)
ax3.set_title("Variance attribution", fontsize=10, color=C_TEXT, pad=6)
ax3.set_ylim(0, 1.05)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

from matplotlib.patches import Patch
ax3.legend(handles=[Patch(facecolor=C_TEMPLATE, label="Template"),
                     Patch(facecolor=C_RESIDUAL, label="Residual")],
           frameon=False, fontsize=7, loc="upper right")

plt.tight_layout()
fig.savefig(OUT / "s4b_within_decomp.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4b_within_decomp.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4b")


# ═══════════════════════════════════════════════
# S4c: Full cross-CL Ridge/MLP component decomposition
# ═══════════════════════════════════════════════
print("\n  S4c: Cross-CL component decomposition...")

cross_decomp = {model: {tgt: {} for tgt in CLS} for model in ["Ridge", "MLP"]}

for tgt in CLS:
    print(f"    →{tgt}...")
    srcs = [c for c in CLS if c != tgt]
    sV, sD = [], []
    for c in srcs:
        for pn, dd in cl_full[c]["deltas"].items():
            if get_emb(pn) is not None:
                sV.append(get_emb(pn)); sD.append(dd)
    sV = np.array(sV); sD = np.array(sD)

    tgt_sh = [p for p in shared if p in cl_full[tgt]["deltas"] and get_emb(p) is not None]
    tV = np.array([get_emb(p) for p in tgt_sh])
    tD = np.array([cl_full[tgt]["deltas"][p] for p in tgt_sh])
    T_tgt = tD.mean(0)
    c_tgt = cl_idx[tgt]

    pca = PCA(50).fit(sV)

    for model_name in ["Ridge", "MLP"]:
        if model_name == "Ridge":
            r = Ridge(10).fit(pca.transform(sV), sD)
            yp = r.predict(pca.transform(tV))
        else:
            m = train_mlp(pca.transform(sV), sD)
            yp = pred_mlp(m, pca.transform(tV))
            del m; torch.cuda.empty_cache()

        bc, gc, r2b, r2g = [], [], [], []
        for i, p in enumerate(tgt_sh):
            if p not in p_idx: continue
            pidx = p_idx[p]
            bc.append(corr(yp[i], beta_a[pidx]))
            gc.append(corr(yp[i], gamma_a[c_tgt, pidx]))
            bn = np.linalg.norm(beta_a[pidx])
            gn = np.linalg.norm(gamma_a[c_tgt, pidx])
            if bn > 1e-12:
                bh = beta_a[pidx] / bn
                r2b.append(np.dot(yp[i], bh) / (np.dot(beta_a[pidx], bh) + 1e-12))
            if gn > 1e-12:
                gh = gamma_a[c_tgt, pidx] / gn
                r2g.append(np.dot(yp[i], gh) / (np.dot(gamma_a[c_tgt, pidx], gh) + 1e-12))

        cross_decomp[model_name][tgt] = {
            "corr_beta": float(np.mean(bc)), "corr_gamma": float(np.mean(gc)),
            "r2_beta": float(np.mean(r2b)) if r2b else 0,
            "r2_gamma": float(np.mean(r2g)) if r2g else 0,
            "pref": pref(tD, yp, T_tgt),
        }

# Plot S4c: 2×2 heatmaps
fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.0))
metrics = [("corr_beta", "corr(δ̂, β)"), ("corr_gamma", "corr(δ̂, γ)"),
           ("r2_beta", "R²(β)"), ("r2_gamma", "R²(γ)")]

for mi, (metric, label) in enumerate(metrics):
    ax = axes[mi // 2, mi % 2]
    mat = np.array([[cross_decomp[model][cl][metric] for model in models] for cl in CLS])
    vmax = max(abs(np.nanmax(mat)), abs(np.nanmin(mat))) * 1.05
    if "gamma" in metric:
        cmap_use = LinearSegmentedColormap.from_list("b_w_t", [C_TEMPLATE, C_ZERO, C_INTERACTION])
    else:
        cmap_use = LinearSegmentedColormap.from_list("b_w_t", [C_TEMPLATE, C_ZERO, C_RESIDUAL])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(mat, cmap=cmap_use, norm=norm, aspect="auto")

    for i in range(len(CLS)):
        for j in range(len(models)):
            v = mat[i, j]
            tc = "white" if abs(v) > vmax * 0.5 else C_TEXT
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=tc)

    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=9)
    ax.set_yticks(range(len(CLS))); ax.set_yticklabels([CL_LABELS[cl] for cl in CLS], fontsize=9)
    ax.set_title(label, fontsize=10, color=C_TEXT, pad=4)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

plt.tight_layout()
fig.savefig(OUT / "s4c_crosscl_decomp.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4c_crosscl_decomp.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4c")


# ═══════════════════════════════════════════════
# S4d: STATE and MORPH component heatmaps
# ═══════════════════════════════════════════════
print("\n  S4d: STATE/MORPH component heatmaps...")

def load_and_decompose(filepath, tgt):
    f = Path(filepath)
    if not f.exists(): return None
    d = np.load(f, allow_pickle=True)
    perts = list(d["test_perts"]); yp = d["predictions"]
    T = np.mean([cl_full[tgt]["deltas"][p] for p in perts if p in cl_full[tgt]["deltas"]], 0)
    Tn = T / (np.linalg.norm(T) + 1e-30)
    c = cl_idx[tgt]
    bc, gc, ec = [], [], []
    yt_l, yp_l = [], []
    for i, p in enumerate(perts):
        if p not in p_idx or p not in cl_full[tgt]["deltas"]: continue
        pidx = p_idx[p]
        true = cl_full[tgt]["deltas"][p]
        pred = yp[i]
        bc.append(corr(pred, beta_a[pidx]))
        gc.append(corr(pred, gamma_a[c, pidx]))
        true_e = true - (true @ Tn) * Tn
        pred_e = pred - (pred @ Tn) * Tn
        ec.append(corr(pred_e, true_e))
        yt_l.append(true); yp_l.append(pred)
    if not bc: return None
    yt_a = np.array(yt_l); yp_a = np.array(yp_l)
    pref_v = np.mean([corr(yt_a[j] - T, yp_a[j] - T) for j in range(len(yt_a))])
    return {"pref": pref_v, "eps_corr": np.mean(ec), "beta_corr": np.mean(bc), "gamma_corr": np.mean(gc)}

# STATE
state_metrics = ["pref", "eps_corr", "beta_corr", "gamma_corr"]
state_settings = ["zeroshot", "calibrated"]
state_data = {cl: {s: None for s in state_settings} for cl in CLS}

for tgt in CLS:
    r = load_and_decompose(BENCH / f"runs/STATE_zeroshot/{tgt}/seed0/predictions.npz", tgt)
    if r: state_data[tgt]["zeroshot"] = r
    r = load_and_decompose(BENCH / f"runs/STATE_crossCL/{tgt}/seed0/predictions.npz", tgt)
    if r: state_data[tgt]["calibrated"] = r

# MORPH
MORPH_PAIRS = [("k562", "rpe1"), ("rpe1", "k562"), ("hepg2", "jurkat"), ("jurkat", "hepg2")]
morph_settings_list = ["Source only", "Target ctrls", "+30% target"]
morph_fnames = {"Source only": "predictions_zeroshot.npz",
                "Target ctrls": "predictions_finetuned.npz",
                "+30% target": "predictions_ft30pct.npz"}

morph_data = {}
for src, tgt in MORPH_PAIRS:
    pk = f"{CL_LABELS[src]}→{CL_LABELS[tgt]}"
    morph_data[pk] = {}
    pair_file = f"{src}_to_{tgt}"
    for setting, fname in morph_fnames.items():
        r = load_and_decompose(BENCH / f"morph_crosscl/runs/{pair_file}/seed0/{fname}", tgt)
        morph_data[pk][setting] = r

# Plot STATE: 4 small heatmaps
metric_labels = {"pref": "Pert-ref Pearson", "eps_corr": "corr(ε̂, ε)",
                 "beta_corr": "corr(δ̂, β)", "gamma_corr": "corr(δ̂, γ)"}

fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.5))
for mi, metric in enumerate(state_metrics):
    ax = axes[mi]
    mat = np.full((len(CLS), 2), np.nan)
    for ci, cl in enumerate(CLS):
        for si, s in enumerate(state_settings):
            if state_data[cl][s] is not None:
                mat[ci, si] = state_data[cl][s][metric]

    vmax = np.nanmax(np.abs(mat)) * 1.1 if not np.all(np.isnan(mat)) else 1
    if metric == "gamma_corr":
        cmap_use = LinearSegmentedColormap.from_list("t", [C_TEMPLATE, C_ZERO, C_INTERACTION])
    else:
        cmap_use = LinearSegmentedColormap.from_list("t", ["#F3F4F6", C_RESIDUAL])
        vmax = max(vmax, 0.01)

    if metric == "gamma_corr":
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = None
        im = ax.imshow(mat, cmap=cmap_use, aspect="auto", vmin=0, vmax=vmax)

    if metric == "gamma_corr":
        im = ax.imshow(mat, cmap=cmap_use, norm=norm, aspect="auto")

    for i in range(len(CLS)):
        for j in range(2):
            v = mat[i, j]
            if np.isnan(v): ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#999"); continue
            tc = "white" if abs(v) > vmax * 0.5 else C_TEXT
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=tc)

    ax.set_xticks([0, 1]); ax.set_xticklabels(["Source\nonly", "+30%\ntarget"], fontsize=7)
    ax.set_yticks(range(len(CLS))); ax.set_yticklabels([CL_LABELS[cl] for cl in CLS] if mi == 0 else [], fontsize=9)
    ax.set_title(metric_labels[metric], fontsize=9, color=C_TEXT, pad=4)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

fig.suptitle("STATE", fontsize=12, color=C_TEXT, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "s4d_state_components.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4d_state_components.pdf", dpi=300, bbox_inches="tight")
plt.close()

# Plot MORPH: 4 small heatmaps
morph_pair_labels = list(morph_data.keys())
fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.5))
for mi, metric in enumerate(state_metrics):
    ax = axes[mi]
    mat = np.full((len(morph_pair_labels), 3), np.nan)
    for pi, pk in enumerate(morph_pair_labels):
        for si, s in enumerate(morph_settings_list):
            if morph_data[pk][s] is not None:
                mat[pi, si] = morph_data[pk][s][metric]

    vmax = np.nanmax(np.abs(mat)) * 1.1 if not np.all(np.isnan(mat)) else 1
    if metric == "gamma_corr":
        cmap_use = LinearSegmentedColormap.from_list("t", [C_TEMPLATE, C_ZERO, C_INTERACTION])
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(mat, cmap=cmap_use, norm=norm, aspect="auto")
    else:
        cmap_use = LinearSegmentedColormap.from_list("t", ["#F3F4F6", C_RESIDUAL])
        im = ax.imshow(mat, cmap=cmap_use, aspect="auto", vmin=0, vmax=max(vmax, 0.01))

    for i in range(len(morph_pair_labels)):
        for j in range(3):
            v = mat[i, j]
            if np.isnan(v): ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#999"); continue
            tc = "white" if abs(v) > vmax * 0.5 else C_TEXT
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=tc)

    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Source", "Ctrls", "+30%"], fontsize=7)
    ax.set_yticks(range(len(morph_pair_labels)))
    ax.set_yticklabels(morph_pair_labels if mi == 0 else [], fontsize=8)
    ax.set_title(metric_labels[metric], fontsize=9, color=C_TEXT, pad=4)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

fig.suptitle("MORPH", fontsize=12, color=C_TEXT, y=1.02)
plt.tight_layout()
fig.savefig(OUT / "s4d_morph_components.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4d_morph_components.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4d")


# ═══════════════════════════════════════════════
# S4e: Pooled vs target-only calibration (β and γ)
# ═══════════════════════════════════════════════
print("\n  S4e: Pooled vs target-only calibration...")

CALIB_FRAC = 0.3
calib_settings = ["Zero-shot", "Source +30%", "Target-only 30%"]
calib_results = {cl: {s: {} for s in calib_settings} for cl in CLS}

for tgt in CLS:
    print(f"    {tgt}...")
    srcs = [c for c in CLS if c != tgt]
    sV, sD = [], []
    for c in srcs:
        for pn, dd in cl_full[c]["deltas"].items():
            if get_emb(pn) is not None:
                sV.append(get_emb(pn)); sD.append(dd)
    sV = np.array(sV); sD = np.array(sD)

    tgt_sh = [p for p in shared if p in cl_full[tgt]["deltas"] and get_emb(p) is not None]
    tV = np.array([get_emb(p) for p in tgt_sh])
    tD = np.array([cl_full[tgt]["deltas"][p] for p in tgt_sh])
    c_tgt = cl_idx[tgt]

    rng = np.random.RandomState(0)
    n_calib = int(len(tgt_sh) * CALIB_FRAC)
    perm = rng.permutation(len(tgt_sh))
    calib_idx = perm[:n_calib]; test_idx = perm[n_calib:]
    test_perts = [tgt_sh[i] for i in test_idx]
    test_V = tV[test_idx]

    def compute_bg(yp, perts):
        bc, gc = [], []
        for i, p in enumerate(perts):
            if p not in p_idx: continue
            pidx = p_idx[p]
            bc.append(corr(yp[i], beta_a[pidx]))
            gc.append(corr(yp[i], gamma_a[c_tgt, pidx]))
        return float(np.mean(bc)), float(np.mean(gc))

    # Zero-shot
    pca = PCA(50).fit(sV)
    r = Ridge(10).fit(pca.transform(sV), sD)
    yp = r.predict(pca.transform(test_V))
    b, g = compute_bg(yp, test_perts)
    calib_results[tgt]["Zero-shot"] = {"beta": b, "gamma": g}

    # Source +30%
    pV = np.vstack([sV, tV[calib_idx]]); pD = np.vstack([sD, tD[calib_idx]])
    pca_p = PCA(50).fit(pV)
    r = Ridge(10).fit(pca_p.transform(pV), pD)
    yp = r.predict(pca_p.transform(test_V))
    b, g = compute_bg(yp, test_perts)
    calib_results[tgt]["Source +30%"] = {"beta": b, "gamma": g}

    # Target-only 30%
    cV = tV[calib_idx]; cD = tD[calib_idx]
    pca_t = PCA(min(50, n_calib - 1)).fit(cV)
    r = Ridge(10).fit(pca_t.transform(cV), cD)
    yp = r.predict(pca_t.transform(test_V))
    b, g = compute_bg(yp, test_perts)
    calib_results[tgt]["Target-only 30%"] = {"beta": b, "gamma": g}

# Plot
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)

x = np.arange(len(CLS))
s_colors = [C_SOURCE, C_CONSERVED, C_INTERACTION]
s_markers = ["o", "D", "o"]
offsets_3 = [-0.2, 0, 0.2]

for ci, cl in enumerate(CLS):
    # β panel
    for si, (s, col, mk) in enumerate(zip(calib_settings, s_colors, s_markers)):
        v = calib_results[cl][s]["beta"]
        ax_l.scatter(x[ci] + offsets_3[si], v, s=40, marker=mk, color=col,
                     edgecolor="white", linewidth=0.5, zorder=3,
                     label=s if ci == 0 else None)
    vals = [calib_results[cl][s]["beta"] for s in calib_settings]
    ax_l.plot([x[ci] + o for o in offsets_3], vals, color=C_LINE, linewidth=1.0, zorder=1)

    # γ panel
    for si, (s, col, mk) in enumerate(zip(calib_settings, s_colors, s_markers)):
        v = calib_results[cl][s]["gamma"]
        ax_r.scatter(x[ci] + offsets_3[si], v, s=40, marker=mk, color=col,
                     edgecolor="white", linewidth=0.5, zorder=3)
    vals = [calib_results[cl][s]["gamma"] for s in calib_settings]
    ax_r.plot([x[ci] + o for o in offsets_3], vals, color=C_LINE, linewidth=1.0, zorder=1)

for ax, title in [(ax_l, "Conserved component β"), (ax_r, "Target interaction γ")]:
    ax.axhline(y=0, color=C_TEXT, linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontsize=10, color=C_TEXT, pad=6)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax_l.set_ylabel("corr(prediction, component)", fontsize=10)
ax_l.legend(frameon=False, fontsize=7.5, loc="lower left")

plt.tight_layout()
fig.savefig(OUT / "s4e_pooled_calibration.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s4e_pooled_calibration.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s4e")


# ═══════════════════════════════════════════════
# S4f: Full metrics table (JSON)
# ═══════════════════════════════════════════════
print("\n  S4f: Full metrics table...")

all_tables = {
    "s4a_within_features": {
        "features": WITHIN_FEATS,
        "values": {CL_LABELS[cl]: {f: float(within_results[ci, fi]) if not np.isnan(within_results[ci, fi]) else None
                                    for fi, f in enumerate(WITHIN_FEATS)}
                   for ci, cl in enumerate(CLS)},
    },
    "s4a_cross_features": {
        "features": CROSS_FEATS,
        "values": {CL_LABELS[cl]: {f: float(cross_results[ci, fi]) if not np.isnan(cross_results[ci, fi]) else None
                                    for fi, f in enumerate(CROSS_FEATS)}
                   for ci, cl in enumerate(CLS)},
    },
    "s4b_within_decomp": within_decomp,
    "s4c_cross_decomp": cross_decomp,
    "s4d_state": {cl: {s: state_data[cl][s] for s in state_settings} for cl in CLS},
    "s4d_morph": morph_data,
    "s4e_calibration": calib_results,
}

with open(OUT / "fig4_supp_results.json", "w") as f:
    json.dump(all_tables, f, indent=2, default=str)

# Print summary tables
print(f"\n{'='*80}")
print("S4b: Within-CL decomposition")
print(f"{'='*80}")
print(f"{'Model':<8} {'CL':<8} {'pref':>8} {'α corr':>8} {'ε corr':>8} {'T%':>6} {'ε%':>6}")
for model in models:
    for cl in CLS:
        d = within_decomp[model][cl]
        print(f"{model:<8} {CL_LABELS[cl]:<8} {d['pref']:>8.4f} {d['alpha_corr']:>8.4f} "
              f"{d['eps_corr']:>8.4f} {d['t_frac']:>5.1%} {d['e_frac']:>5.1%}")

print(f"\n{'='*80}")
print("S4c: Cross-CL 3→1 decomposition")
print(f"{'='*80}")
print(f"{'Model':<8} {'Target':<8} {'pref':>8} {'β corr':>8} {'γ corr':>8} {'R²(β)':>8} {'R²(γ)':>8}")
for model in models:
    for cl in CLS:
        d = cross_decomp[model][cl]
        print(f"{model:<8} {CL_LABELS[cl]:<8} {d['pref']:>8.4f} {d['corr_beta']:>8.4f} "
              f"{d['corr_gamma']:>8.4f} {d['r2_beta']:>8.4f} {d['r2_gamma']:>8.4f}")

print(f"\n{'='*80}")
print("S4e: Calibration β/γ")
print(f"{'='*80}")
print(f"{'Target':<8} {'Setting':<20} {'β':>8} {'γ':>8}")
for cl in CLS:
    for s in calib_settings:
        r = calib_results[cl][s]
        print(f"{CL_LABELS[cl]:<8} {s:<20} {r['beta']:>8.4f} {r['gamma']:>8.4f}")

print(f"\nAll supplementary panels saved to {OUT}")