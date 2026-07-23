"""
Supplementary Figure 3: Cross-cell-line transfer supporting analyses.
All panels saved to figure3_supplementary/.

S3a: Full 1→1 transfer heatmaps (Ridge, MLP)
S3b: STATE/MORPH slope plots (source-only → +30% target)
S3c: Target calibration saturation curves
S3d: Response basis gap heatmap vs MLP transfer heatmap
S3e: Reference baselines (zero-shot, target-only 30%, within-CL full)

Run: python3 fig3_supp.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import torch, torch.nn as nn
import anndata as ad
import pickle, json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
MORPH_BASE = Path("/mnt/storage01/home/amolina/agop-perturbation/morph_pipeline")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure3_supplementary"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_LINE = "#C9CED6"
C_RIDGE = "#149E9A"
C_MLP = "#0D7377"
C_SOURCE = "#8A94A6"
C_FTCTRL = "#6F5CC4"
C_CALIB = "#D26A3A"

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
cl_data = {}
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)
    si = json.load(open(BENCH / "data" / cl / "split_info.json"))
    deltas = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5 and get_emb(pn) is not None:
            deltas[pn] = X[genes == pn].mean(0) - ctrl_mean
    cl_data[cl] = {"deltas": deltas, "train_perts": si["train_perts"],
                   "test_perts": si["test_perts"]}
    print(f"  {cl}: {len(deltas)} perts")


# ═══════════════════════════════════════════════
# S3a: Full 1→1 transfer heatmaps
# ═══════════════════════════════════════════════
print("\n  S3a: Full 1→1 transfer heatmaps...")

n_cl = len(CLS)
ridge_matrix = np.full((n_cl, n_cl), np.nan)
mlp_matrix = np.full((n_cl, n_cl), np.nan)

for si_idx, src in enumerate(CLS):
    sps = sorted(cl_data[src]["deltas"].keys())
    sV = np.array([get_emb(p) for p in sps])
    sD = np.array([cl_data[src]["deltas"][p] for p in sps])
    pca = PCA(50).fit(sV)

    for ti_idx, tgt in enumerate(CLS):
        if src == tgt:
            continue
        tps = sorted(cl_data[tgt]["deltas"].keys())
        tV = np.array([get_emb(p) for p in tps])
        tD = np.array([cl_data[tgt]["deltas"][p] for p in tps])
        T_tgt = tD.mean(0)

        # Ridge
        r = Ridge(alpha=10).fit(pca.transform(sV), sD)
        ridge_matrix[si_idx, ti_idx] = pref(tD, r.predict(pca.transform(tV)), T_tgt)

        # MLP (seed=0)
        m = train_mlp(pca.transform(sV), sD, seed=0)
        mlp_matrix[si_idx, ti_idx] = pref(tD, pred_mlp(m, pca.transform(tV)), T_tgt)
        del m; torch.cuda.empty_cache()

    print(f"    Source {src} done")

# Plot
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 3.2))
cmap = LinearSegmentedColormap.from_list("white_teal", ["#F3F4F6", C_MLP])
vmin = 0
vmax = max(np.nanmax(ridge_matrix), np.nanmax(mlp_matrix)) * 1.05
cl_labels = [CL_LABELS[cl] for cl in CLS]

for ax, mat, title in [(ax_l, ridge_matrix, "Ridge 1→1 transfer"),
                        (ax_r, mlp_matrix, "MLP 1→1 transfer")]:
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(n_cl):
        for j in range(n_cl):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor="#F0F0F0", edgecolor="none"))
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#CCC")
            else:
                v = mat[i, j]
                tc = "white" if v > vmax * 0.55 else C_TEXT
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=tc)
    ax.set_xticks(range(n_cl))
    ax.set_xticklabels(cl_labels, fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_cl))
    ax.set_yticklabels(cl_labels, fontsize=9)
    ax.set_xlabel("Target", fontsize=10); ax.set_ylabel("Source", fontsize=10)
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

cb = fig.colorbar(im, ax=[ax_l, ax_r], shrink=0.8, pad=0.02)
cb.set_label("Perturbed-reference Pearson", fontsize=9)
cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(0.5)

plt.tight_layout()
fig.savefig(OUT / "s3a_1to1_heatmaps.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s3a_1to1_heatmaps.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s3a")


# ═══════════════════════════════════════════════
# S3b: STATE/MORPH slope plots
# ═══════════════════════════════════════════════
print("\n  S3b: STATE/MORPH slope plots...")

CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}

def load_pref_from_npz(filepath, cl):
    f = Path(filepath)
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    yt, yp, T = d["true_deltas"], d["predictions"], d["train_mean"]
    return pref(yt, yp, T)

# STATE
state_data = {}
for tgt in CLS:
    state_data[tgt] = {}
    # Zero-shot
    vals = []
    for seed in range(5):
        v = load_pref_from_npz(BENCH / f"runs/STATE_zeroshot/{tgt}/seed{seed}/predictions.npz", tgt)
        if v is not None: vals.append(v)
    state_data[tgt]["source"] = vals if vals else [0.0]

    # +30%
    vals = []
    for seed in range(5):
        v = load_pref_from_npz(BENCH / f"runs/STATE_crossCL/{tgt}/seed{seed}/predictions.npz", tgt)
        if v is not None: vals.append(v)
    state_data[tgt]["calibrated"] = vals if vals else [0.0]

# MORPH
MORPH_PAIRS = [("k562", "rpe1"), ("rpe1", "k562"), ("hepg2", "jurkat"), ("jurkat", "hepg2")]
morph_settings = ["Source only", "Fine-tune ctrl", "+30% target"]
morph_fnames = {"Source only": "predictions_zeroshot.npz",
                "Fine-tune ctrl": "predictions_finetuned.npz",
                "+30% target": "predictions_ft30pct.npz"}

morph_data = {}
for src, tgt in MORPH_PAIRS:
    pk = f"{CL_LABELS[src]}→{CL_LABELS[tgt]}"
    morph_data[pk] = {}
    pair_file = f"{src}_to_{tgt}"
    for setting, fname in morph_fnames.items():
        v = load_pref_from_npz(BENCH / f"morph_crosscl/runs/{pair_file}/seed0/{fname}", tgt)
        morph_data[pk][setting] = v if v is not None else 0.0

# Plot
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 3.5))

# STATE slope plot
x_state = [0, 1]
for tgt in CLS:
    s_mean = np.mean(state_data[tgt]["source"])
    c_mean = np.mean(state_data[tgt]["calibrated"])
    ax_l.plot(x_state, [s_mean, c_mean], color=CL_COLORS[tgt], linewidth=1.5, zorder=1)
    ax_l.scatter(0, s_mean, s=45, color=CL_COLORS[tgt], edgecolor="white",
                 linewidth=0.5, zorder=3, label=CL_LABELS[tgt])
    ax_l.scatter(1, c_mean, s=45, color=CL_COLORS[tgt], edgecolor="white",
                 linewidth=0.5, zorder=3)
    # Error bars
    if len(state_data[tgt]["source"]) > 1:
        ax_l.errorbar(0, s_mean, yerr=np.std(state_data[tgt]["source"]),
                      fmt="none", ecolor=CL_COLORS[tgt], elinewidth=0.8, capsize=2, capthick=0.6)
    if len(state_data[tgt]["calibrated"]) > 1:
        ax_l.errorbar(1, c_mean, yerr=np.std(state_data[tgt]["calibrated"]),
                      fmt="none", ecolor=CL_COLORS[tgt], elinewidth=0.8, capsize=2, capthick=0.6)

ax_l.set_xticks(x_state)
ax_l.set_xticklabels(["Source only", "+30% target\nperturbations"], fontsize=9)
ax_l.set_ylabel("Perturbed-reference Pearson", fontsize=10)
ax_l.set_title("STATE", fontsize=11, color=C_TEXT, pad=6)
ax_l.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
ax_l.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax_l.xaxis.grid(False)
ax_l.set_axisbelow(True)
ax_l.spines["top"].set_visible(False); ax_l.spines["right"].set_visible(False)
ax_l.legend(frameon=False, fontsize=7.5, loc="upper left")

# MORPH slope plot
PAIR_COLORS = ["#4A90D9", "#D9534F", "#5CB85C", "#F0AD4E"]
x_morph = [0, 1, 2]
for pi, pk in enumerate(morph_data):
    vals = [morph_data[pk][s] for s in morph_settings]
    ax_r.plot(x_morph, vals, color=PAIR_COLORS[pi], linewidth=1.5, zorder=1)
    for xi, v in enumerate(vals):
        ax_r.scatter(xi, v, s=40, color=PAIR_COLORS[pi], edgecolor="white",
                     linewidth=0.5, zorder=3, label=pk if xi == 0 else None)

ax_r.set_xticks(x_morph)
ax_r.set_xticklabels(["Source\nonly", "Target\nctrls", "+30% target\nperturbations"], fontsize=9)
ax_r.set_ylabel("Perturbed-reference Pearson", fontsize=10)
ax_r.set_title("MORPH", fontsize=11, color=C_TEXT, pad=6)
ax_r.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
ax_r.yaxis.grid(True, color=C_GRID, linewidth=0.5); ax_r.xaxis.grid(False)
ax_r.set_axisbelow(True)
ax_r.spines["top"].set_visible(False); ax_r.spines["right"].set_visible(False)
ax_r.legend(frameon=False, fontsize=7, loc="upper left")

plt.tight_layout()
fig.savefig(OUT / "s3b_slope_plots.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s3b_slope_plots.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s3b")


# ═══════════════════════════════════════════════
# S3c: Target calibration saturation curves
# ═══════════════════════════════════════════════
print("\n  S3c: Target calibration saturation curves...")

CALIB_COUNTS = [1, 5, 10, 25, 50, 100, 150]
N_REPS = 3

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)

for tgt in CLS:
    print(f"    {tgt}...")
    srcs = [c for c in CLS if c != tgt]
    tgt_perts = sorted(cl_data[tgt]["deltas"].keys())
    tV = np.array([get_emb(p) for p in tgt_perts])
    tD = np.array([cl_data[tgt]["deltas"][p] for p in tgt_perts])

    # Source data
    sV, sD = [], []
    for c in srcs:
        for pn, d in cl_data[c]["deltas"].items():
            sV.append(get_emb(pn)); sD.append(d)
    sV = np.array(sV); sD = np.array(sD)

    # Within-CL baseline
    tr = [p for p in cl_data[tgt]["train_perts"] if p in cl_data[tgt]["deltas"]]
    te = [p for p in cl_data[tgt]["test_perts"] if p in cl_data[tgt]["deltas"]]
    V_tr_w = np.array([get_emb(p) for p in tr])
    D_tr_w = np.array([cl_data[tgt]["deltas"][p] for p in tr])
    V_te_w = np.array([get_emb(p) for p in te])
    D_te_w = np.array([cl_data[tgt]["deltas"][p] for p in te])
    T_w = D_tr_w.mean(0)
    pca_w = PCA(50).fit(V_tr_w)
    within_pref = pref(D_te_w, Ridge(10).fit(pca_w.transform(V_tr_w), D_tr_w).predict(pca_w.transform(V_te_w)), T_w)

    target_only_means = []
    pooled_means = []
    valid_counts = []

    for nc in CALIB_COUNTS:
        if nc >= len(tgt_perts) - 10:
            continue
        valid_counts.append(nc)
        to_vals, po_vals = [], []

        for rep in range(N_REPS):
            rng = np.random.RandomState(rep)
            perm = rng.permutation(len(tgt_perts))
            calib_idx = perm[:nc]
            test_idx = perm[nc:]

            cV = tV[calib_idx]; cD = tD[calib_idx]
            hV = tV[test_idx]; hD = tD[test_idx]
            T_c = cD.mean(0)

            # Target-only
            n_pca = min(50, nc - 1)
            if n_pca < 1: n_pca = 1
            pca_t = PCA(n_pca).fit(cV)
            r_t = Ridge(10).fit(pca_t.transform(cV), cD)
            to_vals.append(pref(hD, r_t.predict(pca_t.transform(hV)), T_c))

            # Pooled (source + target)
            pV = np.vstack([sV, cV]); pD = np.vstack([sD, cD])
            pca_p = PCA(50).fit(pV)
            r_p = Ridge(10).fit(pca_p.transform(pV), pD)
            po_vals.append(pref(hD, r_p.predict(pca_p.transform(hV)), T_c))

        target_only_means.append((np.mean(to_vals), np.std(to_vals)))
        pooled_means.append((np.mean(po_vals), np.std(po_vals)))

    # Plot target-only (left)
    means_to = [m[0] for m in target_only_means]
    stds_to = [m[1] for m in target_only_means]
    ax_l.plot(valid_counts, means_to, color=CL_COLORS[tgt], linewidth=1.5,
              marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.3,
              label=CL_LABELS[tgt], zorder=2)
    ax_l.fill_between(valid_counts,
                      [m - s for m, s in zip(means_to, stds_to)],
                      [m + s for m, s in zip(means_to, stds_to)],
                      color=CL_COLORS[tgt], alpha=0.1)

    # Plot pooled (right)
    means_po = [m[0] for m in pooled_means]
    stds_po = [m[1] for m in pooled_means]
    ax_r.plot(valid_counts, means_po, color=CL_COLORS[tgt], linewidth=1.5,
              marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.3,
              label=CL_LABELS[tgt], zorder=2)
    ax_r.fill_between(valid_counts,
                      [m - s for m, s in zip(means_po, stds_po)],
                      [m + s for m, s in zip(means_po, stds_po)],
                      color=CL_COLORS[tgt], alpha=0.1)

    # Within-CL baseline
    ax_l.axhline(y=within_pref, color=CL_COLORS[tgt], linewidth=0.7, linestyle=":", alpha=0.4)
    ax_r.axhline(y=within_pref, color=CL_COLORS[tgt], linewidth=0.7, linestyle=":", alpha=0.4)

for ax, title in [(ax_l, "Target-only calibration"), (ax_r, "Source + target pooled")]:
    ax.set_xlabel("Target perturbations used", fontsize=10)
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax_l.set_ylabel("Perturbed-reference Pearson", fontsize=10)
ax_l.legend(frameon=False, fontsize=7.5, loc="lower right")

plt.tight_layout()
fig.savefig(OUT / "s3c_saturation_curves.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s3c_saturation_curves.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s3c")


# ═══════════════════════════════════════════════
# S3d: Response basis gap heatmap vs MLP transfer
# ═══════════════════════════════════════════════
print("\n  S3d: Basis coverage vs MLP transfer heatmaps...")

from scipy.stats import spearmanr

BASIS_RANK = 20
overlap_matrix = np.full((n_cl, n_cl), np.nan)

# Compute residuals
cl_residuals = {}
for cl in CLS:
    D = np.array(list(cl_data[cl]["deltas"].values()))
    T = D.mean(0); Tn = T / (np.linalg.norm(T) + 1e-30)
    R = D - np.outer(D @ Tn, Tn)
    cl_residuals[cl] = R

for si_idx, src in enumerate(CLS):
    for ti_idx, tgt in enumerate(CLS):
        if src == tgt:
            continue
        R_src = cl_residuals[src]; R_tgt = cl_residuals[tgt]
        r_use = min(BASIS_RANK, R_src.shape[0] - 1)
        pca_src = PCA(n_components=r_use).fit(R_src)
        B_src = pca_src.components_
        R_tgt_in = R_tgt @ B_src.T @ B_src
        e_in = np.mean(np.sum(R_tgt_in ** 2, axis=1))
        e_tot = np.mean(np.sum(R_tgt ** 2, axis=1))
        overlap_matrix[si_idx, ti_idx] = e_in / (e_tot + 1e-12) * 100

# Spearman
off = [(i, j) for i in range(n_cl) for j in range(n_cl) if i != j]
ov_flat = [overlap_matrix[i, j] for i, j in off]
mlp_flat = [mlp_matrix[i, j] for i, j in off]
rho, p_val = spearmanr(ov_flat, mlp_flat)

# Plot
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.0, 3.2))
cmap_basis = LinearSegmentedColormap.from_list("white_grey", ["#F3F4F6", "#8A94A6"])
cmap_perf = LinearSegmentedColormap.from_list("white_teal", ["#F3F4F6", C_MLP])

for ax, mat, cmap_use, title, fmt in [
    (ax_l, overlap_matrix, cmap_basis, f"Basis coverage (r={BASIS_RANK})", "{:.0f}%"),
    (ax_r, mlp_matrix, cmap_perf, "MLP 1→1 transfer", "{:.3f}"),
]:
    vmax = np.nanmax(mat) * 1.05
    im = ax.imshow(mat, cmap=cmap_use, aspect="auto", vmin=0, vmax=vmax)
    for i in range(n_cl):
        for j in range(n_cl):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor="#F0F0F0", edgecolor="none"))
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#CCC")
            else:
                v = mat[i, j]
                tc = "white" if v > vmax * 0.55 else C_TEXT
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=tc)
    ax.set_xticks(range(n_cl))
    ax.set_xticklabels(cl_labels, fontsize=9, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_cl))
    ax.set_yticklabels(cl_labels, fontsize=9)
    ax.set_xlabel("Target", fontsize=10); ax.set_ylabel("Source", fontsize=10)
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)

fig.text(0.5, 0.01, f"Spearman ρ(basis coverage, MLP pref) = {rho:.2f}, p = {p_val:.2f}",
         ha="center", fontsize=9, color=C_TEXT)

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT / "s3d_basis_gap_heatmap.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s3d_basis_gap_heatmap.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"    Saved s3d (ρ={rho:.3f})")


# ═══════════════════════════════════════════════
# S3e: Reference baselines
# ═══════════════════════════════════════════════
print("\n  S3e: Reference baselines...")

# Collect: zero-shot 3→1, target-only 30%, within-CL full
baseline_results = {"Ridge": {}, "MLP": {}}

for tgt in CLS:
    print(f"    {tgt}...")
    srcs = [c for c in CLS if c != tgt]

    # Source data
    sV, sD = [], []
    for c in srcs:
        for pn, d in cl_data[c]["deltas"].items():
            sV.append(get_emb(pn)); sD.append(d)
    sV = np.array(sV); sD = np.array(sD)

    # Target data
    tgt_perts = sorted(cl_data[tgt]["deltas"].keys())
    tV = np.array([get_emb(p) for p in tgt_perts])
    tD = np.array([cl_data[tgt]["deltas"][p] for p in tgt_perts])
    T_tgt = tD.mean(0)

    # 30% split
    rng = np.random.RandomState(0)
    n_calib = int(len(tgt_perts) * 0.3)
    perm = rng.permutation(len(tgt_perts))
    calib_idx = perm[:n_calib]; test_idx = perm[n_calib:]
    cV = tV[calib_idx]; cD = tD[calib_idx]
    hV = tV[test_idx]; hD = tD[test_idx]
    T_c = cD.mean(0)

    # Within-CL full
    tr = [p for p in cl_data[tgt]["train_perts"] if p in cl_data[tgt]["deltas"]]
    te = [p for p in cl_data[tgt]["test_perts"] if p in cl_data[tgt]["deltas"]]
    V_tr_w = np.array([get_emb(p) for p in tr])
    D_tr_w = np.array([cl_data[tgt]["deltas"][p] for p in tr])
    V_te_w = np.array([get_emb(p) for p in te])
    D_te_w = np.array([cl_data[tgt]["deltas"][p] for p in te])
    T_w = D_tr_w.mean(0)

    for model_name in ["Ridge", "MLP"]:
        res = {}

        if model_name == "Ridge":
            # Zero-shot
            pca = PCA(50).fit(sV)
            res["3→1 zero-shot"] = pref(tD, Ridge(10).fit(pca.transform(sV), sD).predict(pca.transform(tV)), T_tgt)
            # Target-only 30%
            pca_t = PCA(min(50, n_calib - 1)).fit(cV)
            res["Target-only 30%"] = pref(hD, Ridge(10).fit(pca_t.transform(cV), cD).predict(pca_t.transform(hV)), T_c)
            # Within-CL full
            pca_w = PCA(50).fit(V_tr_w)
            res["Within-CL full"] = pref(D_te_w, Ridge(10).fit(pca_w.transform(V_tr_w), D_tr_w).predict(pca_w.transform(V_te_w)), T_w)
        else:
            pca = PCA(50).fit(sV)
            m = train_mlp(pca.transform(sV), sD); res["3→1 zero-shot"] = pref(tD, pred_mlp(m, pca.transform(tV)), T_tgt); del m; torch.cuda.empty_cache()
            pca_t = PCA(min(50, n_calib - 1)).fit(cV)
            m = train_mlp(pca_t.transform(cV), cD); res["Target-only 30%"] = pref(hD, pred_mlp(m, pca_t.transform(hV)), T_c); del m; torch.cuda.empty_cache()
            pca_w = PCA(50).fit(V_tr_w)
            m = train_mlp(pca_w.transform(V_tr_w), D_tr_w); res["Within-CL full"] = pref(D_te_w, pred_mlp(m, pca_w.transform(V_te_w)), T_w); del m; torch.cuda.empty_cache()

        baseline_results[model_name][tgt] = res

# Plot
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)
settings = ["3→1 zero-shot", "Target-only 30%", "Within-CL full"]

for ax, model_name, title in [(ax_l, "Ridge", "Ridge (DepMap)"), (ax_r, "MLP", "MLP (DepMap)")]:
    x = np.arange(len(CLS))
    for ci, cl in enumerate(CLS):
        vals = [baseline_results[model_name][cl][s] for s in settings]
        ax.plot(range(len(settings)), vals, color=CL_COLORS[cl], linewidth=1.5,
                marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.5,
                label=CL_LABELS[cl] if model_name == "Ridge" else None, zorder=2)

    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(settings, fontsize=8, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontsize=11, color=C_TEXT, pad=6)
    ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5)
    ax.xaxis.grid(False); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax_l.set_ylabel("Perturbed-reference Pearson", fontsize=10)
ax_l.legend(frameon=False, fontsize=7.5, loc="upper left")

plt.tight_layout()
fig.savefig(OUT / "s3e_reference_baselines.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "s3e_reference_baselines.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved s3e")

# ── Save all results ──
save_data = {
    "s3a": {
        "ridge_matrix": [[float(v) if not np.isnan(v) else None for v in row] for row in ridge_matrix],
        "mlp_matrix": [[float(v) if not np.isnan(v) else None for v in row] for row in mlp_matrix],
    },
    "s3d": {
        "overlap_matrix": [[float(v) if not np.isnan(v) else None for v in row] for row in overlap_matrix],
        "spearman_rho": float(rho), "spearman_p": float(p_val),
    },
    "s3e": {
        model: {cl: {s: float(v) for s, v in vals.items()} for cl, vals in cl_vals.items()}
        for model, cl_vals in baseline_results.items()
    },
}
with open(OUT / "fig3_supp_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nAll supplementary panels saved to {OUT}")