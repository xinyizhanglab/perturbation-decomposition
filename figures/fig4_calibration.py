"""
Figure 4c: STATE and MORPH — source-only vs calibrated.
Multi-seed for all settings.

Run: python3 fig4_calibration.py
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
OUT = SCRIPT_DIR / "figure4"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
MORPH_PAIRS = [("k562", "rpe1"), ("rpe1", "k562"), ("hepg2", "jurkat"), ("jurkat", "hepg2")]

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_LINE = "#C9CED6"
C_CONSERVED = "#6F5CC4"
C_INTERACTION = "#D26A3A"


def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def pref(yt, yp, T):
    return np.mean([corr(yt[i] - T, yp[i] - T) for i in range(len(yt))])


plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ── Load data for ANOVA decomposition ──
print("Loading data...")
cl_data = {}
for cl in CLS:
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")
    X = np.array(a.obsm["X_hvg"])
    genes = a.obs["gene"].astype(str).values
    ctrl_mean = X[genes == "non-targeting"].mean(0)
    deltas = {}
    for pn in sorted(set(genes) - {"non-targeting"}):
        if (genes == pn).sum() >= 5:
            deltas[pn] = X[genes == pn].mean(0) - ctrl_mean
    cl_data[cl] = deltas
    del a

shared = sorted(set.intersection(*[set(cl_data[cl].keys()) for cl in CLS]))
D_tensor = np.array([[cl_data[cl][p] for p in shared] for cl in CLS])
mu = D_tensor.mean(axis=(0, 1))
alpha_a = D_tensor.mean(axis=1) - mu
beta_a = D_tensor.mean(axis=0) - mu
gamma_a = D_tensor - mu - alpha_a[:, None, :] - beta_a[None, :, :]
cl_idx = {c: i for i, c in enumerate(CLS)}
p_idx = {p: i for i, p in enumerate(shared)}
print(f"  ANOVA shared perts: {len(shared)}")


def decompose_preds(perts, yp, tgt):
    c = cl_idx[tgt]
    valid = [(i, p) for i, p in enumerate(perts) if p in p_idx and p in cl_data[tgt]]
    if len(valid) < 5:
        return 0.0, 0.0, 0.0

    T = np.mean([cl_data[tgt][p] for _, p in valid], axis=0)
    bc, gc, yt_list, yp_list = [], [], [], []
    for i, p in valid:
        pidx = p_idx[p]
        bc.append(corr(yp[i], beta_a[pidx]))
        gc.append(corr(yp[i], gamma_a[c, pidx]))
        yt_list.append(cl_data[tgt][p])
        yp_list.append(yp[i])

    yt_arr = np.array(yt_list)
    yp_arr = np.array(yp_list)
    pref_v = np.mean([corr(yt_arr[j] - T, yp_arr[j] - T) for j in range(len(yt_arr))])
    return float(pref_v), float(np.mean(bc)), float(np.mean(gc))


def load_seeds(pattern_fn, max_seeds=5):
    """Load all available seeds matching a path pattern function."""
    results = []
    for seed in range(max_seeds):
        f = pattern_fn(seed)
        if f.exists():
            results.append(np.load(f, allow_pickle=True))
    return results


def eval_setting(seed_data_list, tgt):
    """Evaluate decomposition across seeds, return dict with mean/std."""
    pf_list, bc_list, gc_list = [], [], []
    for d in seed_data_list:
        perts = list(d["test_perts"])
        yp = d["predictions"]
        mask = [i for i, p in enumerate(perts) if p in p_idx and p in cl_data[tgt]]
        if len(mask) < 5:
            continue
        pf, bc, gc = decompose_preds([perts[i] for i in mask], yp[mask], tgt)
        pf_list.append(pf)
        bc_list.append(bc)
        gc_list.append(gc)

    if not pf_list:
        return {"pref": 0.0, "beta": 0.0, "gamma": 0.0, "n_seeds": 0}

    return {
        "pref": float(np.mean(pf_list)),
        "pref_std": float(np.std(pf_list)) if len(pf_list) > 1 else 0.0,
        "beta": float(np.mean(bc_list)),
        "beta_std": float(np.std(bc_list)) if len(bc_list) > 1 else 0.0,
        "gamma": float(np.mean(gc_list)),
        "gamma_std": float(np.std(gc_list)) if len(gc_list) > 1 else 0.0,
        "n_seeds": len(pf_list),
    }


# ═══════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════
print("\n  STATE decomposition...")
state_results = {cl: {} for cl in CLS}

for tgt in CLS:
    # Zeroshot (all seeds)
    seeds = load_seeds(lambda s, t=tgt: BENCH / f"runs/STATE_zeroshot/{t}/seed{s}/predictions.npz")
    state_results[tgt]["zeroshot"] = eval_setting(seeds, tgt)
    r = state_results[tgt]["zeroshot"]
    print(f"    {tgt} zeroshot: pref={r['pref']:.4f}, β={r['beta']:.4f}, γ={r['gamma']:.4f} "
          f"({r['n_seeds']} seeds)")

    # CrossCL / +30% target (all seeds)
    seeds = load_seeds(lambda s, t=tgt: BENCH / f"runs/STATE_crossCL/{t}/seed{s}/predictions.npz")
    state_results[tgt]["calibrated"] = eval_setting(seeds, tgt)
    r = state_results[tgt]["calibrated"]
    print(f"    {tgt} +30%:    pref={r['pref']:.4f}, β={r['beta']:.4f}, γ={r['gamma']:.4f} "
          f"({r['n_seeds']} seeds)")


# ═══════════════════════════════════════════════
# MORPH
# ═══════════════════════════════════════════════
print("\n  MORPH decomposition...")
morph_results = {}

for src, tgt in MORPH_PAIRS:
    pair_key = f"{src}→{tgt}"
    pair_file = f"{src}_to_{tgt}"
    morph_results[pair_key] = {}

    for setting, suffix in [
        ("Source only", "predictions_zeroshot.npz"),
        ("Fine-tune ctrl", "predictions_finetuned.npz"),
        ("+30% target", "predictions_ft30pct.npz"),
    ]:
        seeds = load_seeds(lambda s, pf=pair_file, sf=suffix: BENCH / f"morph_crosscl/runs/{pf}/seed{s}/{sf}")
        morph_results[pair_key][setting] = eval_setting(seeds, tgt)
        r = morph_results[pair_key][setting]
        print(f"    {pair_key} {setting}: pref={r['pref']:.4f}, β={r['beta']:.4f}, "
              f"γ={r['gamma']:.4f} ({r['n_seeds']} seeds)")


# ═══════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.6, 4.3), sharey=True)

rows = []
for cl in CLS:
    rows.append({
        "group": "STATE", "label": CL_LABELS[cl],
        "values": [
            ("Source only", state_results[cl]["zeroshot"]),
            ("+30% target", state_results[cl]["calibrated"]),
        ],
    })
for src, tgt in MORPH_PAIRS:
    pk = f"{src}→{tgt}"
    rows.append({
        "group": "MORPH", "label": f"{CL_LABELS[src]}→{CL_LABELS[tgt]}",
        "values": [
            ("Source only", morph_results[pk]["Source only"]),
            ("Fine-tune ctrl", morph_results[pk]["Fine-tune ctrl"]),
            ("+30% target", morph_results[pk]["+30% target"]),
        ],
    })

y = np.arange(len(rows))

setting_styles = {
    "Source only": {"color": C_CONSERVED, "marker": "o", "label": "Source cell line"},
    "Fine-tune ctrl": {"color": C_LINE, "marker": "D", "label": "Fine-tune target baseline"},
    "+30% target": {"color": C_INTERACTION, "marker": "o", "label": "+30% target perturbations"},
}


def plot_metric(ax, metric, xlabel):
    used_labels = set()
    for ri, row in enumerate(rows):
        vals = [res[metric] for _, res in row["values"] if res.get("n_seeds", 0) > 0]
        if len(vals) > 1:
            ax.plot(vals, [y[ri]] * len(vals), color=C_LINE, linewidth=1.2, zorder=1)

        for setting, res in row["values"]:
            if res.get("n_seeds", 0) == 0:
                continue
            st = setting_styles[setting]
            xval = res[metric]
            xstd = res.get(metric + "_std", 0.0)

            if xstd and xstd > 0:
                ax.errorbar(xval, y[ri], xerr=xstd, fmt="none",
                            ecolor=st["color"], elinewidth=0.8,
                            capsize=1.5, capthick=0.6, zorder=2)

            label = st["label"] if st["label"] not in used_labels else None
            used_labels.add(st["label"])
            ax.scatter(xval, y[ri], s=45, marker=st["marker"],
                       color=st["color"], edgecolor="white", linewidth=0.5,
                       zorder=3, label=label)

    ax.axvline(x=0, color=C_TEXT, linewidth=0.6, alpha=0.5, zorder=1)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plot_metric(ax_left, "pref", "Perturbed-reference Pearson")
plot_metric(ax_right, "gamma", "corr(δ̂, γ)")

ax_left.set_yticks(y)
ax_left.set_yticklabels([])
ax_left.set_ylim(len(rows) - 0.5, -0.5)
ax_right.set_yticks(y)
ax_right.set_yticklabels([])
ax_right.set_ylim(len(rows) - 0.5, -0.5)

for ri, lab in enumerate([r["label"] for r in rows]):
    ax_left.text(-0.075, y[ri], lab, transform=ax_left.get_yaxis_transform(),
                 ha="right", va="center", fontsize=8.5, color=C_TEXT, clip_on=False)

ax_left.set_title("Prediction performance", fontsize=10, color=C_TEXT, pad=6)
ax_right.set_title("Interaction recovery", fontsize=10, color=C_TEXT, pad=6)

sep_y = len(CLS) - 0.5
for ax in [ax_left, ax_right]:
    ax.axhline(sep_y, color=C_GRID, linewidth=0.8, zorder=0)

ax_left.text(-0.34, 0.22, "STATE", transform=ax_left.transAxes, rotation=90,
             ha="center", va="center", fontsize=8, color=C_TEXT, clip_on=False)
ax_left.text(-0.34, 0.73, "MORPH", transform=ax_left.transAxes, rotation=90,
             ha="center", va="center", fontsize=8, color=C_TEXT, clip_on=False)

handles, labels_leg = ax_left.get_legend_handles_labels()
seen = set(); uh, ul = [], []
for h, l in zip(handles, labels_leg):
    if l and l not in seen:
        seen.add(l); uh.append(h); ul.append(l)
fig.legend(uh, ul, frameon=False, fontsize=8, loc="lower center",
           bbox_to_anchor=(0.5, -0.03), ncol=3, columnspacing=1.8, handletextpad=0.5)

plt.tight_layout(rect=[0.24, 0.08, 1, 1])
fig.savefig(OUT / "fig4c_calibration.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig4c_calibration.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Summary ──
print(f"\n{'='*80}")
print("STATE: source-only vs +30% target")
print(f"{'='*80}")
print(f"{'Target':<8} {'Setting':<15} {'pref':>8} {'β':>8} {'γ':>8} {'seeds':>6}")
print("-" * 55)
for cl in CLS:
    for setting in ["zeroshot", "calibrated"]:
        r = state_results[cl][setting]
        label = "Source only" if setting == "zeroshot" else "+30% target"
        print(f"{CL_LABELS[cl]:<8} {label:<15} "
              f"{r['pref']:>8.4f} {r['beta']:>8.4f} {r['gamma']:>8.4f} {r['n_seeds']:>6}")

print(f"\n{'='*80}")
print("MORPH: source-only vs ft-ctrl vs +30%")
print(f"{'='*80}")
print(f"{'Pair':<15} {'Setting':<15} {'pref':>8} {'β':>8} {'γ':>8} {'seeds':>6}")
print("-" * 60)
for src, tgt in MORPH_PAIRS:
    pk = f"{src}→{tgt}"
    for s in ["Source only", "Fine-tune ctrl", "+30% target"]:
        r = morph_results[pk][s]
        print(f"{pk:<15} {s:<15} "
              f"{r['pref']:>8.4f} {r['beta']:>8.4f} {r['gamma']:>8.4f} {r['n_seeds']:>6}")

save_data = {"state": {cl: state_results[cl] for cl in CLS}, "morph": morph_results}
with open(OUT / "fig4c_calibration_results.json", "w") as f:
    json.dump(save_data, f, indent=2)
print(f"\nSaved to {OUT}")