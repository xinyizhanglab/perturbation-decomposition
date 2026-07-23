"""
Figure 3: Cross-cell-line transfer.

Panel a: Direct 1→1 transfer (MORPH, Ridge, MLP) — MLP 5 seeds
Panel b: 3→1 transfer to unseen target (STATE, Ridge, MLP) — MLP 5 seeds, STATE all available seeds

Run: python3 fig3_crosscl_transfer.py
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
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

PAIRS_1to1 = [("k562", "rpe1"), ("rpe1", "k562"), ("hepg2", "jurkat"), ("jurkat", "hepg2")]
N_MLP_SEEDS = 5

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

MODEL_STYLE_1to1 = {
    "MORPH": {"color": "#6F5CC4", "marker": "D", "label": "MORPH"},
    "Ridge": {"color": "#149E9A", "marker": "s", "label": "Ridge (DepMap)"},
    "MLP":   {"color": "#0D7377", "marker": "^", "label": "MLP (DepMap)"},
}

MODEL_STYLE_3to1 = {
    "STATE": {"color": "#8A94A6", "marker": "o", "label": "STATE"},
    "Ridge": {"color": "#149E9A", "marker": "s", "label": "Ridge (DepMap)"},
    "MLP":   {"color": "#0D7377", "marker": "^", "label": "MLP (DepMap)"},
}

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


# ═══════════════════════════════════════════════
# PANEL A: 1→1 transfer
# ═══════════════════════════════════════════════
print("\n\nPanel A: 1→1 transfer...")

# Store as lists of values for error bars
results_1to1 = {model: {} for model in MODEL_STYLE_1to1}

for src, tgt in PAIRS_1to1:
    pair_key = f"{src}→{tgt}"
    print(f"\n  {CL_LABELS[src]}→{CL_LABELS[tgt]}:")

    sps = sorted(cl_data[src].keys())
    sV = np.array([get_emb(p) for p in sps])
    sD = np.array([cl_data[src][p] for p in sps])

    tps = sorted(cl_data[tgt].keys())
    tV = np.array([get_emb(p) for p in tps])
    tD = np.array([cl_data[tgt][p] for p in tps])
    T_tgt = tD.mean(0)

    # Ridge (deterministic)
    pca = PCA(50).fit(sV)
    r = Ridge(alpha=10).fit(pca.transform(sV), sD)
    p_r = pref(tD, r.predict(pca.transform(tV)), T_tgt)
    results_1to1["Ridge"][pair_key] = [p_r]
    print(f"    Ridge: {p_r:.4f}")

    # MLP (5 seeds)
    mlp_vals = []
    for seed in range(N_MLP_SEEDS):
        m = train_mlp(pca.transform(sV), sD, seed=seed)
        with torch.no_grad():
            yp = m(torch.tensor(pca.transform(tV), dtype=torch.float32, device=dev)).cpu().numpy()
        mlp_vals.append(pref(tD, yp, T_tgt))
        del m; torch.cuda.empty_cache()
    results_1to1["MLP"][pair_key] = mlp_vals
    print(f"    MLP:   {np.mean(mlp_vals):.4f} ± {np.std(mlp_vals):.4f}")

    # MORPH (from disk)
    morph_pair = f"{src}_to_{tgt}"
    morph_vals = []
    morph_f = BENCH / f"morph_crosscl/runs/{morph_pair}/seed0/predictions_zeroshot.npz"
    if morph_f.exists():
        d = np.load(morph_f, allow_pickle=True)
        morph_vals.append(pref(d["true_deltas"], d["predictions"], d["train_mean"]))
    # Also try old MORPH
    morph_f2 = BENCH / f"runs/MORPH_crossCL/{morph_pair}/seed0/predictions.npz"
    if morph_f2.exists() and not morph_vals:
        d = np.load(morph_f2, allow_pickle=True)
        morph_vals.append(pref(d["true_deltas"], d["predictions"], d["train_mean"]))
    if not morph_vals:
        morph_vals = [0.0]
        print(f"    MORPH: not found, using 0.0")
    else:
        print(f"    MORPH: {np.mean(morph_vals):.4f}")
    results_1to1["MORPH"][pair_key] = morph_vals

# ── Plot Panel A ──
# ── Plot Panel A ──
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.set_title("One cell line source", fontsize=11, color=C_TEXT, pad=8)

pair_labels = [f"{CL_LABELS[s]}→{CL_LABELS[t]}" for s, t in PAIRS_1to1]
pair_keys = [f"{s}→{t}" for s, t in PAIRS_1to1]
x = np.arange(len(PAIRS_1to1))
n_models = len(MODEL_STYLE_1to1)
offsets = np.linspace(-0.18, 0.18, n_models)



for mi, (model, style) in enumerate(MODEL_STYLE_1to1.items()):
    for i, pk in enumerate(pair_keys):
        vals = results_1to1[model].get(pk, [0])
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
ax.set_xticklabels(pair_labels, fontsize=9, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Transfer pair", fontsize=10, color=C_TEXT)
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
fig.savefig(OUT / "fig3a_1to1_transfer.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3a_1to1_transfer.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("\n  Saved fig3a")


# ═══════════════════════════════════════════════
# PANEL B: 3→1 transfer
# ═══════════════════════════════════════════════
print("\n\nPanel B: 3→1 transfer...")

results_3to1 = {model: {} for model in MODEL_STYLE_3to1}

for tgt in CLS:
    print(f"\n  →{CL_LABELS[tgt]}:")
    srcs = [c for c in CLS if c != tgt]

    sV, sD = [], []
    for c in srcs:
        for pn, d in cl_data[c].items():
            sV.append(get_emb(pn)); sD.append(d)
    sV = np.array(sV); sD = np.array(sD)

    tps = sorted(cl_data[tgt].keys())
    tV = np.array([get_emb(p) for p in tps])
    tD = np.array([cl_data[tgt][p] for p in tps])
    T_tgt = tD.mean(0)

    # Ridge (deterministic)
    pca = PCA(50).fit(sV)
    r = Ridge(alpha=10).fit(pca.transform(sV), sD)
    p_r = pref(tD, r.predict(pca.transform(tV)), T_tgt)
    results_3to1["Ridge"][tgt] = [p_r]
    print(f"    Ridge: {p_r:.4f}")

    # MLP (5 seeds)
    mlp_vals = []
    for seed in range(N_MLP_SEEDS):
        m = train_mlp(pca.transform(sV), sD, seed=seed)
        with torch.no_grad():
            yp = m(torch.tensor(pca.transform(tV), dtype=torch.float32, device=dev)).cpu().numpy()
        mlp_vals.append(pref(tD, yp, T_tgt))
        del m; torch.cuda.empty_cache()
    results_3to1["MLP"][tgt] = mlp_vals
    print(f"    MLP:   {np.mean(mlp_vals):.4f} ± {np.std(mlp_vals):.4f}")

    # STATE zero-shot (all available seeds)
    state_vals = []
    for seed in range(5):
        state_f = BENCH / f"runs/STATE_zeroshot/{tgt}/seed{seed}/predictions.npz"
        if state_f.exists():
            d = np.load(state_f, allow_pickle=True)
            state_vals.append(pref(d["true_deltas"], d["predictions"], d["train_mean"]))
    if not state_vals:
        state_vals = [0.0]
        print(f"    STATE: not found, using 0.0")
    else:
        print(f"    STATE: {np.mean(state_vals):.4f} ± {np.std(state_vals):.4f} ({len(state_vals)} seeds)")
    results_3to1["STATE"][tgt] = state_vals

# ── Plot Panel B ──
# ── Plot Panel B ──
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.set_title("Three cell lines source", fontsize=11, color=C_TEXT, pad=8)

x = np.arange(len(CLS))
n_models = len(MODEL_STYLE_3to1)
offsets = np.linspace(-0.18, 0.18, n_models)


for mi, (model, style) in enumerate(MODEL_STYLE_3to1.items()):
    for i, cl in enumerate(CLS):
        vals = results_3to1[model].get(cl, [0])
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
fig.savefig(OUT / "fig3b_3to1_transfer.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig3b_3to1_transfer.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("\n  Saved fig3b")

# ── Print summary ──
print(f"\n{'='*80}")
print("Panel A: 1→1 transfer (pref, mean ± std)")
print(f"{'='*80}")
print(f"{'Pair':<20}", end="")
for model in MODEL_STYLE_1to1:
    print(f"  {model:>18}", end="")
print()
print("-" * 80)
for pk in pair_keys:
    print(f"{pk:<20}", end="")
    for model in MODEL_STYLE_1to1:
        vals = results_1to1[model].get(pk, [0])
        if len(vals) > 1:
            print(f"  {np.mean(vals):.3f}±{np.std(vals):.3f}", end="")
        else:
            print(f"  {vals[0]:>18.4f}", end="")
    print()

print(f"\n{'='*80}")
print("Panel B: 3→1 transfer (pref, mean ± std)")
print(f"{'='*80}")
print(f"{'Target':<10}", end="")
for model in MODEL_STYLE_3to1:
    print(f"  {model:>18}", end="")
print()
print("-" * 65)
for cl in CLS:
    print(f"{CL_LABELS[cl]:<10}", end="")
    for model in MODEL_STYLE_3to1:
        vals = results_3to1[model].get(cl, [0])
        if len(vals) > 1:
            print(f"  {np.mean(vals):.3f}±{np.std(vals):.3f}", end="")
        else:
            print(f"  {vals[0]:>18.4f}", end="")
    print()

# Save
save_data = {
    "panel_a_1to1": {},
    "panel_b_3to1": {},
}
for model in MODEL_STYLE_1to1:
    save_data["panel_a_1to1"][model] = {}
    for pk in pair_keys:
        vals = results_1to1[model].get(pk, [])
        save_data["panel_a_1to1"][model][pk] = {
            "values": [float(v) for v in vals],
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if len(vals) > 1 else None,
        }
for model in MODEL_STYLE_3to1:
    save_data["panel_b_3to1"][model] = {}
    for cl in CLS:
        vals = results_3to1[model].get(cl, [])
        save_data["panel_b_3to1"][model][cl] = {
            "values": [float(v) for v in vals],
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if len(vals) > 1 else None,
        }

with open(OUT / "fig3_crosscl_results.json", "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")