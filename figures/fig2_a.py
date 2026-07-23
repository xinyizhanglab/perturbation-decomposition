"""
Figure 2a: Within-CL perturbation-specific prediction across models.
Dodged dot plot with error bars.

- Ridge, MLP, MORPH: 5-fold CV on identical splits (DepMap-eligible perts)
- scGPT: 5 seeds from disk, HVG-subsetted, training centroid

Run: python3 fig2_within_cl_pref.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import anndata as ad
import json

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure2"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
DS = {"k562": "replogle_k562_essential", "rpe1": "replogle_rpe1_essential",
      "hepg2": "nadig_hepg2", "jurkat": "nadig_jurkat"}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

MODEL_STYLE = {
    "scGPT":  {"color": "#8A94A6", "marker": "o", "label": "scGPT"},
    "MORPH":  {"color": "#6F5CC4", "marker": "D", "label": "MORPH"},
    "Ridge":  {"color": "#149E9A", "marker": "s", "label": "Ridge (DepMap)"},
    "MLP":    {"color": "#0D7377", "marker": "^", "label": "MLP (DepMap)"},
}


def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def pref(yt, yp, T):
    return np.mean([corr(yt[i] - T, yp[i] - T) for i in range(len(yt))])


# ── Load results ──
results = {model: {cl: [] for cl in CLS} for model in MODEL_STYLE}

# Ridge, MLP, MORPH: from 5-fold CV
FOLDCV_DIRS = {"Ridge": "Ridge_5foldCV", "MLP": "MLP_5foldCV", "MORPH": "MORPH_5foldCV"}

for model, fdir in FOLDCV_DIRS.items():
    for cl in CLS:
        f = BENCH / f"runs/{fdir}/{DS[cl]}/fold_results.json"
        if f.exists():
            d = json.load(open(f))
            results[model][cl] = [r["pref"] for r in d]

# scGPT: 5 seeds from disk, HVG-subsetted, training centroid
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    hvg_idx = np.where(a.var["highly_variable"].values)[0]
    X_hvg = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X_hvg[genes == "non-targeting"].mean(0)
    si = json.load(open(BENCH / "data" / cl / "split_info.json"))

    # Training centroid
    deltas = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5:
            deltas[pn] = X_hvg[genes == pn].mean(0) - ctrl_mean
    T_train = np.mean([deltas[p] for p in si["train_perts"] if p in deltas], axis=0)

    for seed in range(5):
        scgpt_f = BENCH / f"runs/scGPT/{DS[cl]}/seed{seed}/predictions.npz"
        if scgpt_f.exists():
            sd = np.load(scgpt_f, allow_pickle=True)
            yt_s, yp_s = sd["true_deltas"], sd["predictions"]
            if yt_s.shape[1] == 6642:
                yt_s = yt_s[:, hvg_idx]
                yp_s = yp_s[:, hvg_idx]
            results["scGPT"][cl].append(pref(yt_s, yp_s, T_train))

# ── Print summary ──
print(f"\n{'='*70}")
print("Within-CL pref summary")
print(f"{'='*70}")
print(f"{'Model':<12}  {'source':<10}", end="")
for cl in CLS:
    print(f"  {CL_LABELS[cl]:>14}", end="")
print()
print("-" * 82)
sources = {"scGPT": "5 seeds", "MORPH": "5-fold CV", "Ridge": "5-fold CV", "MLP": "5-fold CV"}
for model in MODEL_STYLE:
    print(f"{model:<12}  {sources[model]:<10}", end="")
    for cl in CLS:
        vals = results[model][cl]
        if len(vals) == 0:
            print(f"  {'(pending)':>14}", end="")
        elif len(vals) == 1:
            print(f"  {vals[0]:>14.4f}", end="")
        else:
            print(f"  {np.mean(vals):.3f}±{np.std(vals):.3f}", end="")
    print()

# ── Plot ──
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

fig, ax = plt.subplots(figsize=(4.5, 3.2))

n_models = len(MODEL_STYLE)
x = np.arange(len(CLS))
offsets = np.linspace(-0.2, 0.2, n_models)

for mi, (model, style) in enumerate(MODEL_STYLE.items()):
    for ci, cl in enumerate(CLS):
        vals = results[model][cl]
        if len(vals) == 0:
            continue

        mean_v = np.mean(vals)

        # Error bars
        if len(vals) > 1:
            std_v = np.std(vals)
            ax.errorbar(x[ci] + offsets[mi], mean_v, yerr=std_v,
                        fmt="none", ecolor=style["color"], elinewidth=1.0,
                        capsize=2, capthick=0.8, zorder=2)

        # Dot
        ax.scatter(x[ci] + offsets[mi], mean_v, s=40, marker=style["marker"],
                   color=style["color"], edgecolor="white", linewidth=0.5,
                   zorder=3, label=style["label"] if ci == 0 else None)

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

# Deduplicated legend outside right
handles, labels = ax.get_legend_handles_labels()
seen = set()
unique_h, unique_l = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l)
        unique_h.append(h)
        unique_l.append(l)

ax.legend(unique_h, unique_l, frameon=False, fontsize=8,
          loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=0.8)

plt.tight_layout(rect=[0, 0, 0.78, 1])
fig.savefig(OUT / "fig2a_within_cl_pref.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2a_within_cl_pref.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Save results JSON ──
save_results = {}
for model in MODEL_STYLE:
    save_results[model] = {}
    for cl in CLS:
        vals = results[model][cl]
        save_results[model][cl] = {
            "values": [float(v) for v in vals],
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if len(vals) > 1 else None,
            "source": sources[model],
        }

with open(OUT / "fig2a_within_cl_pref_results.json", "w") as f:
    json.dump(save_results, f, indent=2)

print(f"\nSaved to {OUT}")