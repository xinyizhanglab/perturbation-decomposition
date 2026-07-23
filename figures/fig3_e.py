"""
Figure 3e: Source response basis overlap is not sufficient for transfer.
Two aligned heatmaps:
  Left: 1→1 MLP transfer performance (source × target)
  Right: Source basis coverage of target residual variance (source × target)

Run: python3 fig3_basis_vs_transfer.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
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

BASIS_RANK = 20
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

# ── Load data ──
print("Loading data...")
cl_data = {}
cl_residuals = {}

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
    T = D.mean(0)
    Tn = T / (np.linalg.norm(T) + 1e-30)
    R = D - np.outer(D @ Tn, Tn)
    cl_residuals[cl] = R
    print(f"  {cl}: {len(deltas)} perts")

# ── Compute 4×4 matrices ──
n_cl = len(CLS)
mlp_matrix = np.full((n_cl, n_cl), np.nan)
overlap_matrix = np.full((n_cl, n_cl), np.nan)

for si, src in enumerate(CLS):
    for ti, tgt in enumerate(CLS):
        if src == tgt:
            continue

        label = f"{CL_LABELS[src]}→{CL_LABELS[tgt]}"
        print(f"\n  {label}:")

        # ── Basis overlap ──
        R_src = cl_residuals[src]
        R_tgt = cl_residuals[tgt]
        r_use = min(BASIS_RANK, R_src.shape[0] - 1)
        pca_src = PCA(n_components=r_use).fit(R_src)
        B_src = pca_src.components_

        R_tgt_in = R_tgt @ B_src.T @ B_src
        energy_in = np.mean(np.sum(R_tgt_in ** 2, axis=1))
        energy_total = np.mean(np.sum(R_tgt ** 2, axis=1))
        overlap = energy_in / (energy_total + 1e-12) * 100
        overlap_matrix[si, ti] = overlap
        print(f"    Basis coverage: {overlap:.1f}%")

        # ── MLP 1→1 (mean of 3 seeds) ──
        sps = sorted(cl_data[src].keys())
        sV = np.array([get_emb(p) for p in sps])
        sD = np.array([cl_data[src][p] for p in sps])

        tps = sorted(cl_data[tgt].keys())
        tV = np.array([get_emb(p) for p in tps])
        tD = np.array([cl_data[tgt][p] for p in tps])
        T_tgt = tD.mean(0)

        pca = PCA(50).fit(sV)
        mlp_seeds = []
        for seed in range(3):
            m = train_mlp(pca.transform(sV), sD, seed=seed)
            with torch.no_grad():
                yp = m(torch.tensor(pca.transform(tV), dtype=torch.float32, device=dev)).cpu().numpy()
            mlp_seeds.append(pref(tD, yp, T_tgt))
            del m; torch.cuda.empty_cache()
        mlp_matrix[si, ti] = float(np.mean(mlp_seeds))
        print(f"    MLP pref: {mlp_matrix[si, ti]:.4f}")

# ── Spearman on off-diagonal entries ──
off_diag = [(si, ti) for si in range(n_cl) for ti in range(n_cl) if si != ti]
ov_flat = [overlap_matrix[si, ti] for si, ti in off_diag]
mlp_flat = [mlp_matrix[si, ti] for si, ti in off_diag]
rho, p_val = spearmanr(ov_flat, mlp_flat)
print(f"\nSpearman ρ(basis overlap, MLP pref) = {rho:.4f}, p = {p_val:.4f}")

# ── Plot ──
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.5, 3.2))

cl_labels = [CL_LABELS[cl] for cl in CLS]

# Colormaps
cmap_mlp = LinearSegmentedColormap.from_list("white_teal", ["#F3F4F6", "#0D7377"])
cmap_basis = LinearSegmentedColormap.from_list("white_grey", ["#F3F4F6", "#8A94A6"])

# Mask diagonal
mlp_plot = np.copy(mlp_matrix)
overlap_plot = np.copy(overlap_matrix)

# Left: MLP transfer
im_left = ax_left.imshow(mlp_plot, cmap=cmap_mlp, aspect="auto",
                          vmin=0, vmax=np.nanmax(mlp_plot) * 1.05)

for si in range(n_cl):
    for ti in range(n_cl):
        if si == ti:
            ax_left.add_patch(plt.Rectangle((ti - 0.5, si - 0.5), 1, 1,
                              facecolor="#F0F0F0", edgecolor="none"))
            ax_left.text(ti, si, "—", ha="center", va="center",
                        fontsize=9, color="#CCCCCC")
        else:
            v = mlp_plot[si, ti]
            tc = "white" if v > np.nanmax(mlp_plot) * 0.55 else C_TEXT
            ax_left.text(ti, si, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=tc)

ax_left.set_xticks(range(n_cl))
ax_left.set_xticklabels(cl_labels, fontsize=9, color=C_TEXT,
                        rotation=45, ha="right", rotation_mode="anchor")
ax_left.set_yticks(range(n_cl))
ax_left.set_yticklabels(cl_labels, fontsize=9, color=C_TEXT)
ax_left.set_xlabel("Target", fontsize=10, color=C_TEXT)
ax_left.set_ylabel("Source", fontsize=10, color=C_TEXT)
ax_left.set_title("MLP transfer (pref)", fontsize=11, color=C_TEXT, pad=6)

for spine in ax_left.spines.values():
    spine.set_visible(False)
ax_left.tick_params(length=0)

cb_left = fig.colorbar(im_left, ax=ax_left, shrink=0.8, pad=0.02)
cb_left.ax.tick_params(labelsize=7, colors=C_TEXT)
cb_left.outline.set_edgecolor(C_TEXT)
cb_left.outline.set_linewidth(0.5)

# Right: Basis coverage
im_right = ax_right.imshow(overlap_plot, cmap=cmap_basis, aspect="auto",
                            vmin=0, vmax=np.nanmax(overlap_plot) * 1.05)

for si in range(n_cl):
    for ti in range(n_cl):
        if si == ti:
            ax_right.add_patch(plt.Rectangle((ti - 0.5, si - 0.5), 1, 1,
                               facecolor="#F0F0F0", edgecolor="none"))
            ax_right.text(ti, si, "—", ha="center", va="center",
                         fontsize=9, color="#CCCCCC")
        else:
            v = overlap_plot[si, ti]
            tc = "white" if v > np.nanmax(overlap_plot) * 0.55 else C_TEXT
            ax_right.text(ti, si, f"{v:.0f}%", ha="center", va="center",
                         fontsize=8, color=tc)

ax_right.set_xticks(range(n_cl))
ax_right.set_xticklabels(cl_labels, fontsize=9, color=C_TEXT,
                         rotation=45, ha="right", rotation_mode="anchor")
ax_right.set_yticks(range(n_cl))
ax_right.set_yticklabels(cl_labels, fontsize=9, color=C_TEXT)
ax_right.set_xlabel("Target", fontsize=10, color=C_TEXT)
ax_right.set_ylabel("Source", fontsize=10, color=C_TEXT)
ax_right.set_title("Basis coverage (%)", fontsize=11, color=C_TEXT, pad=6)

for spine in ax_right.spines.values():
    spine.set_visible(False)
ax_right.tick_params(length=0)

cb_right = fig.colorbar(im_right, ax=ax_right, shrink=0.8, pad=0.02)
cb_right.ax.tick_params(labelsize=7, colors=C_TEXT)
cb_right.outline.set_edgecolor(C_TEXT)
cb_right.outline.set_linewidth(0.5)

# Spearman annotation
fig.text(0.5, 0.01, f"Spearman ρ(basis coverage, MLP pref) = {rho:.2f}, p = {p_val:.2f}",
         ha="center", fontsize=9, color=C_TEXT)

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(OUT / "fig3e_basis_vs_transfer.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3e_basis_vs_transfer.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print matrices ──
print(f"\n{'='*60}")
print("MLP 1→1 transfer (pref)")
print(f"{'='*60}")
print(f"{'Source':<8}", end="")
for cl in CLS:
    print(f"  →{CL_LABELS[cl]:>8}", end="")
print()
print("-" * 45)
for si, src in enumerate(CLS):
    print(f"{CL_LABELS[src]:<8}", end="")
    for ti, tgt in enumerate(CLS):
        if si == ti:
            print(f"  {'—':>9}", end="")
        else:
            print(f"  {mlp_matrix[si, ti]:>9.4f}", end="")
    print()

print(f"\n{'='*60}")
print(f"Basis coverage (r={BASIS_RANK}, %)")
print(f"{'='*60}")
print(f"{'Source':<8}", end="")
for cl in CLS:
    print(f"  →{CL_LABELS[cl]:>8}", end="")
print()
print("-" * 45)
for si, src in enumerate(CLS):
    print(f"{CL_LABELS[src]:<8}", end="")
    for ti, tgt in enumerate(CLS):
        if si == ti:
            print(f"  {'—':>9}", end="")
        else:
            print(f"  {overlap_matrix[si, ti]:>8.1f}%", end="")
    print()

print(f"\nSpearman ρ = {rho:.4f}, p = {p_val:.4f}")

# Save
save_data = {
    "cls": CLS,
    "mlp_matrix": [[float(v) if not np.isnan(v) else None for v in row]
                    for row in mlp_matrix],
    "overlap_matrix": [[float(v) if not np.isnan(v) else None for v in row]
                       for row in overlap_matrix],
    "spearman_rho": float(rho),
    "spearman_p": float(p_val),
    "basis_rank": BASIS_RANK,
}
with open(OUT / "fig3e_basis_vs_transfer_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")