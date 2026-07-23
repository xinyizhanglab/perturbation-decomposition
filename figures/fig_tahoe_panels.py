"""
Figure: Tahoe drug perturbation prediction — 3 separate plots.

Panel A — Information-regime ladder (drug-held-out vs MoA-held-out)
Panel B — Source basis saturation
Panel C — Template/leakage controls

Run: python3 fig_tahoe_panels.py
"""
import json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
DATA = BENCH / "data" / "tahoe_cc"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure_tahoe"
OUT.mkdir(exist_ok=True)

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_DRUG = "#3572B0"
C_MOA = "#D26A3A"
C_NULL = "#C9CED6"

CLS = ["C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1"]
CL_COLORS = {"C32": "#4A90D9", "HOP62": "#D9534F", "HepG2/C3A": "#5CB85C",
             "Hs 766T": "#F0AD4E", "PANC-1": "#6F5CC4"}

def corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

def pref(yt, yp, T):
    return np.mean([corr(yt[i]-T, yp[i]-T) for i in range(len(yt))])

# ── Load results ──
with open(DATA / "testing_battery_results.json") as f:
    battery = json.load(f)
with open(DATA / "moa_holdout_results.json") as f:
    moa_res = json.load(f)

# ── Compute MoA-held-out k=10 ──
print("Computing MoA-held-out k=10...")
with open(DATA / "pseudobulk_deltas.pkl", "rb") as f:
    pdata = pickle.load(f)
with open(DATA / "drug_moa.json") as f:
    drug_moa = json.load(f)

deltas = pdata["deltas"]
cl_responses = {}
for cl in CLS:
    cl_responses[cl] = {}
    for (c, drug, conc), delta in deltas.items():
        if c == cl:
            cl_responses[cl][(drug, conc)] = delta

moa_drugs = defaultdict(set)
for drug, m in drug_moa.items():
    if m != "unclear":
        moa_drugs[m].add(drug)
valid_moas = {m: d for m, d in moa_drugs.items() if len(d) >= 3}
all_moa_drugs = set()
for d in valid_moas.values():
    all_moa_drugs |= d

moa_k10_source = {}
for tgt in CLS:
    srcs = [c for c in CLS if c != tgt]
    obs = []
    for dc in sorted(cl_responses[tgt].keys()):
        drug, conc = dc
        if drug not in all_moa_drugs: continue
        m = drug_moa.get(drug, "unclear")
        if m == "unclear": continue
        src_resps = [cl_responses[s][dc] for s in srcs if dc in cl_responses[s]]
        if src_resps:
            obs.append({"drug": drug, "moa": m,
                        "src": np.mean(src_resps, axis=0).astype(np.float32),
                        "tgt": cl_responses[tgt][dc].astype(np.float32)})
    if len(obs) < 20: continue
    prefs = []
    for held_moa in valid_moas:
        train = [o for o in obs if o["moa"] != held_moa]
        test = [o for o in obs if o["moa"] == held_moa]
        if len(train) < 10 or len(test) < 2: continue
        V_tr = np.array([o["src"] for o in train])
        D_tr = np.array([o["tgt"] for o in train])
        V_te = np.array([o["src"] for o in test])
        D_te = np.array([o["tgt"] for o in test])
        nc = min(10, V_tr.shape[0]-1)
        pca = PCA(nc).fit(V_tr)
        ridge = Ridge(alpha=10).fit(pca.transform(V_tr), D_tr)
        prefs.append(pref(D_te, ridge.predict(pca.transform(V_te)), D_tr.mean(0)))
    if prefs:
        moa_k10_source[tgt] = np.mean(prefs)

# ── Aggregate values ──
drug_morgan = np.mean([battery["test1_ladder"][cl]["morgan"] for cl in CLS])
drug_cc = np.mean([battery["test1_ladder"][cl]["cc"] for cl in CLS])
drug_direct = np.mean([battery["test1_ladder"][cl]["direct"] for cl in CLS])
drug_k10 = np.mean([battery["test2_saturation"][cl]["10"]["mean"] for cl in CLS])
drug_k50 = np.mean([battery["test2_saturation"][cl]["50"]["mean"] for cl in CLS])

moa_morgan = np.mean([moa_res["test_b_moa_holdout_prior"][f"{cl}__Morgan"]["mean"]
                       for cl in CLS if f"{cl}__Morgan" in moa_res["test_b_moa_holdout_prior"]])
moa_cc = np.mean([moa_res["test_b_moa_holdout_prior"][f"{cl}__CC"]["mean"]
                   for cl in CLS if f"{cl}__CC" in moa_res["test_b_moa_holdout_prior"]])
moa_direct = np.mean([moa_res["test_a_moa_holdout_source"][cl]["mean_direct"] for cl in CLS])
moa_k10 = np.mean(list(moa_k10_source.values()))
moa_k50 = np.mean([moa_res["test_a_moa_holdout_source"][cl]["mean_source"] for cl in CLS])

null_val = np.mean([battery["test3_leakage"][cl]["null_mean"]
                     for cl in battery["test3_leakage"]])

print(f"\n{'Method':<35} {'Drug-out':>10} {'MoA-out':>10}")
print("-" * 58)
for name, dv, mv in [
    ("Morgan", drug_morgan, moa_morgan),
    ("Chemical Checker", drug_cc, moa_cc),
    ("Mean effect source CL", drug_direct, moa_direct),
    ("Source CL pert decomp k=10", drug_k10, moa_k10),
    ("Source CL pert decomp k=50", drug_k50, moa_k50),
]:
    print(f"{name:<35} {dv:>10.4f} {mv:>10.4f}")

# ── Style ──
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.labelsize": 10,
    "axes.linewidth": 0.6, "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": C_TEXT, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ══════════════════════════════════════════════════════════════
# PANEL A: Information ladder
# ══════════════════════════════════════════════════════════════
fig_a, ax = plt.subplots(figsize=(5.5, 3.5))

methods = [
    "Morgan",
    "Chemical Checker",
    "Mean effect source CL",
    "Source CL pert decomp k=10",
    "Source CL pert decomp k=50",
]
d_vals = [drug_morgan, drug_cc, drug_direct, drug_k10, drug_k50]
m_vals = [moa_morgan, moa_cc, moa_direct, moa_k10, moa_k50]

y = np.arange(len(methods))
bar_h = 0.32

for i in range(len(methods)):
    ax.barh(y[i] - bar_h/2, d_vals[i], height=bar_h, color=C_DRUG, alpha=0.85, edgecolor="white")
    ax.barh(y[i] + bar_h/2, m_vals[i], height=bar_h, color=C_MOA, alpha=0.85, edgecolor="white")

ax.axvline(null_val, color=C_NULL, linewidth=1.2, linestyle=":", zorder=1)

ax.set_yticks(y)
ax.set_yticklabels(methods, fontsize=8)
ax.set_xlabel("Residual pref", fontsize=10)
ax.set_xlim(-0.05, 0.92)
ax.xaxis.grid(True, color=C_GRID, linewidth=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.invert_yaxis()

leg_a = [
    Line2D([0], [0], color=C_DRUG, linewidth=6, alpha=0.85, label="Drug-held-out"),
    Line2D([0], [0], color=C_MOA, linewidth=6, alpha=0.85, label="MoA-held-out"),
    Line2D([0], [0], color=C_NULL, linewidth=1.2, linestyle=":", label="Permutation null"),
]
ax.legend(handles=leg_a, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
fig_a.savefig(OUT / "fig_tahoe_a_ladder.svg", dpi=300, bbox_inches="tight")
fig_a.savefig(OUT / "fig_tahoe_a_ladder.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_a)
print("Saved Panel A")

# ══════════════════════════════════════════════════════════════
# PANEL B: Source basis saturation
# ══════════════════════════════════════════════════════════════
fig_b, ax = plt.subplots(figsize=(4.5, 3.2))

sat = battery["test2_saturation"]
RANKS = [1, 2, 5, 10, 20, 50, 100, 150]

for cl in CLS:
    y_vals = [sat[cl][str(k)]["mean"] for k in RANKS]
    ax.plot(RANKS, y_vals, marker="o", markersize=3.5, linewidth=1.3,
            color=CL_COLORS[cl], markeredgecolor="white", markeredgewidth=0.3)

ax.set_xscale("log")
ax.set_xlabel("Source SVD programs (k)", fontsize=10)
ax.set_ylabel("Residual pref", fontsize=10)
ax.set_xticks(RANKS)
ax.set_xticklabels([str(k) for k in RANKS], fontsize=8)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg_b = [Line2D([0], [0], color=CL_COLORS[cl], linewidth=2, marker="o", markersize=3,
                markeredgecolor="white", label=cl) for cl in CLS]
ax.legend(handles=leg_b, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
fig_b.savefig(OUT / "fig_tahoe_b_saturation.svg", dpi=300, bbox_inches="tight")
fig_b.savefig(OUT / "fig_tahoe_b_saturation.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_b)
print("Saved Panel B")

# ══════════════════════════════════════════════════════════════
# PANEL C: Controls
# ══════════════════════════════════════════════════════════════
fig_c, ax = plt.subplots(figsize=(3.5, 3.2))

ctrl_labels = ["Source CL\npert decomp", "Template\nremoved", "Permuted"]
ctrl_colors = [C_DRUG, "#149E9A", C_NULL]

for cl in CLS:
    source_val = battery["test9_template"][cl]["with_template"]
    notemplate_val = battery["test9_template"][cl]["without_template"]
    null_cl = battery["test3_leakage"].get(cl, {}).get("null_mean", None)

    x_pts = [0, 1]
    y_pts = [source_val, notemplate_val]
    if null_cl is not None:
        x_pts.append(2)
        y_pts.append(null_cl)

    ax.scatter(x_pts, y_pts, s=25, color=CL_COLORS[cl],
               edgecolor="white", linewidth=0.4, zorder=3)

mean_source = np.mean([battery["test9_template"][cl]["with_template"] for cl in CLS])
mean_noT = np.mean([battery["test9_template"][cl]["without_template"] for cl in CLS])
null_cls = [cl for cl in CLS if cl in battery["test3_leakage"]]
mean_null = np.mean([battery["test3_leakage"][cl]["null_mean"] for cl in null_cls])

ax.bar([0, 1, 2], [mean_source, mean_noT, mean_null],
       color=ctrl_colors, alpha=0.25, width=0.6, edgecolor="none")

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(ctrl_labels, fontsize=8, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylabel("Residual pref", fontsize=10)
ax.yaxis.grid(True, color=C_GRID, linewidth=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(-0.08, 0.92)

leg_c = [Line2D([0], [0], marker="o", color=CL_COLORS[cl], markersize=4,
                markeredgecolor="white", linewidth=0, label=cl) for cl in CLS]
ax.legend(handles=leg_c, frameon=False, fontsize=7,
          loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
fig_c.savefig(OUT / "fig_tahoe_c_controls.svg", dpi=300, bbox_inches="tight")
fig_c.savefig(OUT / "fig_tahoe_c_controls.pdf", dpi=300, bbox_inches="tight")
plt.close(fig_c)
print("Saved Panel C")

print(f"\nAll saved to {OUT}")