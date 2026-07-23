"""
Supplementary Figure 2.3: Expression-derived descriptors and PLS control.

S2.3a: Feature ablation — pref across different perturbation descriptors
       (DepMap, delta columns, co-perturbation matrix, gene-gene correlation,
        control expression stats, DepMap + control expression)
S2.3b: PLS vs Ridge vs MLP — decoder-free response alignment control

All saved to supplementary_2_3/

Run: python3 fig2_supp_features.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import anndata as ad
import pickle, json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
MORPH_BASE = Path("/mnt/storage01/home/amolina/agop-perturbation/morph_pipeline")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "supplementary_2_3"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_RESIDUAL = "#149E9A"
C_CONSERVED = "#6F5CC4"

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


# ═══════════════════════════════════════════════
# S2.3a: Feature ablation
# ═══════════════════════════════════════════════
print("S2.3a: Feature ablation across perturbation descriptors...")

FEATURE_NAMES = [
    "DepMap PCA50",
    "Delta columns",
    "Co-perturbation",
    "Gene-gene corr",
    "Control expr",
    "DepMap + ctrl expr",
]

feature_results = {fn: {} for fn in FEATURE_NAMES}

for cl in CLS:
    print(f"\n  {cl.upper()}:")
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    G = X.shape[1]
    genes = a.obs["gene"].astype(str).values
    ctrl_cells = X[genes == "non-targeting"]
    ctrl_mean = ctrl_cells.mean(0)
    si = json.load(open(BENCH / "data" / cl / "split_info.json"))

    # HVG gene names
    hvg_names = list(a.var_names[a.var["highly_variable"]])
    gene_to_col = {g: i for i, g in enumerate(hvg_names)}

    # All perturbation deltas
    all_perts = sorted(set(genes) - {"non-targeting"})
    deltas = {}
    for pn in all_perts:
        if (genes == pn).sum() >= 5:
            deltas[pn] = X[genes == pn].mean(0) - ctrl_mean

    # Perturbations with DepMap
    perts_with_emb = [p for p in sorted(deltas.keys()) if get_emb(p) is not None]
    perts_all = sorted(deltas.keys())

    tr_emb = [p for p in si["train_perts"] if p in deltas and get_emb(p) is not None]
    te_emb = [p for p in si["test_perts"] if p in deltas and get_emb(p) is not None]
    tr_all = [p for p in si["train_perts"] if p in deltas]
    te_all = [p for p in si["test_perts"] if p in deltas]

    D_tr = np.array([deltas[p] for p in tr_emb])
    D_te = np.array([deltas[p] for p in te_emb])
    T_tr = D_tr.mean(0)

    # ── DepMap PCA50 ──
    V_tr = np.array([get_emb(p) for p in tr_emb])
    V_te = np.array([get_emb(p) for p in te_emb])
    pca = PCA(50).fit(V_tr)
    r = Ridge(alpha=10).fit(pca.transform(V_tr), D_tr)
    p_dep = pref(D_te, r.predict(pca.transform(V_te)), T_tr)
    feature_results["DepMap PCA50"][cl] = p_dep
    print(f"    DepMap PCA50:      {p_dep:.4f}")

    # ── Delta columns ──
    # For each pert gene p, its delta column = how gene p responds when OTHER genes
    # are knocked down. Feature vector = deltas[:, gene_col_p]
    D_tr_all = np.array([deltas[p] for p in tr_all])
    D_te_all = np.array([deltas[p] for p in te_all])
    T_tr_all = D_tr_all.mean(0)

    dcol_tr, dcol_te = [], []
    tr_dc, te_dc = [], []
    for p in tr_all:
        if p in gene_to_col:
            col = gene_to_col[p]
            dcol_tr.append(D_tr_all[:, col])
            tr_dc.append(p)
    for p in te_all:
        if p in gene_to_col:
            col = gene_to_col[p]
            dcol_te.append(D_tr_all[:, col])  # use TRAIN deltas as features
            te_dc.append(p)

    if len(dcol_tr) > 10 and len(dcol_te) > 5:
        Ftr = np.array(dcol_tr)
        Fte = np.array(dcol_te)
        Dtr_dc = np.array([deltas[p] for p in tr_dc])
        Dte_dc = np.array([deltas[p] for p in te_dc])
        T_dc = Dtr_dc.mean(0)
        pca_dc = PCA(min(50, Ftr.shape[1] - 1, Ftr.shape[0] - 1)).fit(Ftr)
        r_dc = Ridge(alpha=10).fit(pca_dc.transform(Ftr), Dtr_dc)
        p_dc = pref(Dte_dc, r_dc.predict(pca_dc.transform(Fte)), T_dc)
    else:
        p_dc = 0.0
    feature_results["Delta columns"][cl] = p_dc
    print(f"    Delta columns:     {p_dc:.4f}")

    # ── Co-perturbation matrix ──
    # C = D^T D — each row is the co-perturbation profile of a gene
    D_all_train = np.array([deltas[p] for p in tr_all])
    C = D_all_train.T @ D_all_train  # (G, G)
    C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    copt_tr, copt_te = [], []
    tr_co, te_co = [], []
    for p in tr_all:
        if p in gene_to_col:
            copt_tr.append(C_norm[gene_to_col[p]])
            tr_co.append(p)
    for p in te_all:
        if p in gene_to_col:
            copt_te.append(C_norm[gene_to_col[p]])
            te_co.append(p)

    if len(copt_tr) > 10 and len(copt_te) > 5:
        Ftr = np.array(copt_tr)
        Fte = np.array(copt_te)
        Dtr_co = np.array([deltas[p] for p in tr_co])
        Dte_co = np.array([deltas[p] for p in te_co])
        T_co = Dtr_co.mean(0)
        pca_co = PCA(min(50, Ftr.shape[1] - 1, Ftr.shape[0] - 1)).fit(Ftr)
        r_co = Ridge(alpha=10).fit(pca_co.transform(Ftr), Dtr_co)
        p_co = pref(Dte_co, r_co.predict(pca_co.transform(Fte)), T_co)
    else:
        p_co = 0.0
    feature_results["Co-perturbation"][cl] = p_co
    print(f"    Co-perturbation:   {p_co:.4f}")

    # ── Gene-gene correlation ──
    # Correlation of gene p's expression across ALL cells with other genes
    corr_mat = np.corrcoef(X.T)  # (G, G)
    corr_mat = np.nan_to_num(corr_mat)

    gc_tr, gc_te = [], []
    tr_gc, te_gc = [], []
    for p in tr_all:
        if p in gene_to_col:
            gc_tr.append(corr_mat[gene_to_col[p]])
            tr_gc.append(p)
    for p in te_all:
        if p in gene_to_col:
            gc_te.append(corr_mat[gene_to_col[p]])
            te_gc.append(p)

    if len(gc_tr) > 10 and len(gc_te) > 5:
        Ftr = np.array(gc_tr)
        Fte = np.array(gc_te)
        Dtr_gc = np.array([deltas[p] for p in tr_gc])
        Dte_gc = np.array([deltas[p] for p in te_gc])
        T_gc = Dtr_gc.mean(0)
        pca_gc = PCA(min(50, Ftr.shape[1] - 1, Ftr.shape[0] - 1)).fit(Ftr)
        r_gc = Ridge(alpha=10).fit(pca_gc.transform(Ftr), Dtr_gc)
        p_gc = pref(Dte_gc, r_gc.predict(pca_gc.transform(Fte)), T_gc)
    else:
        p_gc = 0.0
    feature_results["Gene-gene corr"][cl] = p_gc
    print(f"    Gene-gene corr:    {p_gc:.4f}")

    # ── Control expression stats ──
    # For target gene p: mean, std, median, dropout, q25, q75 in control cells
    ctrl_feats_tr, ctrl_feats_te = [], []
    tr_cf, te_cf = [], []
    for p in tr_emb:
        if p in gene_to_col:
            col = gene_to_col[p]
            ctrl_feats_tr.append([
                ctrl_cells[:, col].mean(),
                ctrl_cells[:, col].std(),
                np.median(ctrl_cells[:, col]),
                (ctrl_cells[:, col] == 0).mean(),
                np.percentile(ctrl_cells[:, col], 25),
                np.percentile(ctrl_cells[:, col], 75),
            ])
            tr_cf.append(p)
    for p in te_emb:
        if p in gene_to_col:
            col = gene_to_col[p]
            ctrl_feats_te.append([
                ctrl_cells[:, col].mean(),
                ctrl_cells[:, col].std(),
                np.median(ctrl_cells[:, col]),
                (ctrl_cells[:, col] == 0).mean(),
                np.percentile(ctrl_cells[:, col], 25),
                np.percentile(ctrl_cells[:, col], 75),
            ])
            te_cf.append(p)

    if len(ctrl_feats_tr) > 10 and len(ctrl_feats_te) > 5:
        Ftr = np.array(ctrl_feats_tr)
        Fte = np.array(ctrl_feats_te)
        sc = StandardScaler().fit(Ftr)
        Dtr_cf = np.array([deltas[p] for p in tr_cf])
        Dte_cf = np.array([deltas[p] for p in te_cf])
        T_cf = Dtr_cf.mean(0)
        r_cf = Ridge(alpha=10).fit(sc.transform(Ftr), Dtr_cf)
        p_cf = pref(Dte_cf, r_cf.predict(sc.transform(Fte)), T_cf)
    else:
        p_cf = 0.0
    feature_results["Control expr"][cl] = p_cf
    print(f"    Control expr:      {p_cf:.4f}")

    # ── DepMap + control expression ──
    if len(ctrl_feats_tr) > 10:
        V_tr_cf = np.array([get_emb(p) for p in tr_cf])
        V_te_cf = np.array([get_emb(p) for p in te_cf])
        combined_tr = np.hstack([V_tr_cf, np.array(ctrl_feats_tr)])
        combined_te = np.hstack([V_te_cf, np.array(ctrl_feats_te)])
        pca_comb = PCA(50).fit(combined_tr)
        r_comb = Ridge(alpha=10).fit(pca_comb.transform(combined_tr), Dtr_cf)
        p_comb = pref(Dte_cf, r_comb.predict(pca_comb.transform(combined_te)), T_cf)
    else:
        p_comb = 0.0
    feature_results["DepMap + ctrl expr"][cl] = p_comb
    print(f"    DepMap + ctrl:     {p_comb:.4f}")

# ── Plot S2.3a ──
fig, ax = plt.subplots(figsize=(6.0, 3.5))

x = np.arange(len(CLS))
n_feats = len(FEATURE_NAMES)
width = 0.75 / n_feats
offsets = np.linspace(-(n_feats - 1) / 2 * width, (n_feats - 1) / 2 * width, n_feats)

feat_colors = ["#149E9A", "#8A94A6", "#8A94A6", "#8A94A6", "#8A94A6", "#6F5CC4"]

for fi, fn in enumerate(FEATURE_NAMES):
    vals = [feature_results[fn].get(cl, 0) for cl in CLS]
    ax.bar(x + offsets[fi], vals, width * 0.9, color=feat_colors[fi], edgecolor="white",
           linewidth=0.5, label=fn, zorder=2)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Cell line", fontsize=10, color=C_TEXT)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, fontsize=7, loc="center left",
          bbox_to_anchor=(1.02, 0.5), labelspacing=0.5)

plt.tight_layout(rect=[0, 0, 0.72, 1])
fig.savefig(OUT / "s2_3a_feature_ablation.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s2_3a_feature_ablation.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("\n    Saved s2_3a")


# ═══════════════════════════════════════════════
# S2.3b: PLS vs Ridge vs MLP
# ═══════════════════════════════════════════════
print("\n\nS2.3b: PLS vs Ridge comparison...")

PLS_COMPONENTS = [5, 10, 20, 30, 50]
pls_results = {nc: {} for nc in PLS_COMPONENTS}
ridge_pca50 = {}

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

    # Ridge PCA50
    pca = PCA(50).fit(V_tr)
    r = Ridge(alpha=10).fit(pca.transform(V_tr), D_tr)
    p_r = pref(D_te, r.predict(pca.transform(V_te)), T_tr)
    ridge_pca50[cl] = p_r
    print(f"    Ridge PCA50: {p_r:.4f}")

    # PLS at various components
    for nc in PLS_COMPONENTS:
        nc_use = min(nc, V_tr.shape[0] - 1, V_tr.shape[1])
        try:
            pls = PLSRegression(n_components=nc_use, max_iter=1000)
            pls.fit(V_tr, D_tr)
            p_pls = pref(D_te, pls.predict(V_te), T_tr)
        except Exception:
            p_pls = 0.0
        pls_results[nc][cl] = p_pls
        print(f"    PLS n={nc:>3}: {p_pls:.4f}")

# ── Plot S2.3b ──
fig, ax = plt.subplots(figsize=(5.0, 3.2))

x = np.arange(len(CLS))

for cl in CLS:
    ci = CLS.index(cl)
    pls_vals = [pls_results[nc][cl] for nc in PLS_COMPONENTS]
    ax.plot(range(len(PLS_COMPONENTS)), pls_vals, color=CL_COLORS[cl],
            linewidth=1.5, marker="o", markersize=4, markeredgecolor="white",
            markeredgewidth=0.3, label=CL_LABELS[cl], zorder=2)

    # Ridge baseline as horizontal dashed line
    ax.axhline(y=ridge_pca50[cl], color=CL_COLORS[cl], linewidth=1.0,
               linestyle="--", alpha=0.5, zorder=1)

ax.set_xticks(range(len(PLS_COMPONENTS)))
ax.set_xticklabels([str(nc) for nc in PLS_COMPONENTS], fontsize=9, color=C_TEXT)
ax.set_xlabel("PLS components", fontsize=10, color=C_TEXT)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Custom legend: lines = PLS, dashed = Ridge PCA50
from matplotlib.lines import Line2D
legend_cl = [Line2D([0], [0], color=CL_COLORS[cl], linewidth=1.5, label=CL_LABELS[cl])
             for cl in CLS]
legend_method = [Line2D([0], [0], color=C_TEXT, linewidth=1.0, linestyle="--",
                        alpha=0.5, label="Ridge PCA50")]
ax.legend(handles=legend_cl + legend_method, frameon=False, fontsize=7.5,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.5)

plt.tight_layout(rect=[0, 0, 0.78, 1])
fig.savefig(OUT / "s2_3b_pls_comparison.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s2_3b_pls_comparison.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("\n    Saved s2_3b")

# ── Print summary table ──
print(f"\n{'='*70}")
print("Feature ablation summary (pref)")
print(f"{'='*70}")
print(f"{'Feature':<25}", end="")
for cl in CLS:
    print(f"  {CL_LABELS[cl]:>10}", end="")
print()
print("-" * 70)
for fn in FEATURE_NAMES:
    print(f"{fn:<25}", end="")
    for cl in CLS:
        v = feature_results[fn].get(cl, float("nan"))
        print(f"  {v:>10.4f}", end="")
    print()

# Save
save_data = {
    "feature_ablation": {fn: {cl: float(feature_results[fn].get(cl, 0))
                              for cl in CLS} for fn in FEATURE_NAMES},
    "pls": {str(nc): {cl: float(pls_results[nc].get(cl, 0)) for cl in CLS}
            for nc in PLS_COMPONENTS},
    "ridge_pca50": {cl: float(ridge_pca50[cl]) for cl in CLS},
}
with open(OUT / "s2_3_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")