"""
Figure 2e: MORPH uses DepMap but does not expose the same response-aligned geometry.
Line plot: pref across representation stages inside MORPH.

Run: python3 fig2_morph_mechanism.py
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json as json_mod

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure2"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
CL_COLORS = {"k562": "#4A90D9", "rpe1": "#D9534F", "hepg2": "#5CB85C", "jurkat": "#F0AD4E"}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

STAGES = [
    ("depmap_raw",           "Raw DepMap"),
    ("mechanism_codes",      "Perturbation latent"),
    ("mechanism_plus_broad", "Joint latent"),
    ("depmap_resid",         "Refined latent"),
]

# ── Parse truncated JSON files ──
def parse_results(filepath):
    text = filepath.read_text()
    blocks = re.split(r'"(\w+)":\s*\{', text)
    result = {}
    for i in range(1, len(blocks), 2):
        key = blocks[i]
        body = blocks[i + 1]
        entry = {}
        pref_m = re.search(r'"pref":\s*([0-9e.\-]+)', body)
        std_m = re.search(r'"std":\s*([0-9e.\-]+)', body)
        if pref_m:
            entry["pref"] = float(pref_m.group(1))
        if std_m:
            entry["std"] = float(std_m.group(1))
        result[key] = entry
    return result

# ── Load data ──
all_results = {}
for cl in CLS:
    f = BENCH / f"runs/MECHANISM/{cl}/seed0/results.json"
    all_results[cl] = parse_results(f)

print(f"{'Stage':<25}", end="")
for cl in CLS:
    print(f"  {CL_LABELS[cl]:>10}", end="")
print()
print("-" * 70)
for stage_key, stage_label in STAGES:
    print(f"{stage_label:<25}", end="")
    for cl in CLS:
        v = all_results[cl].get(stage_key, {}).get("pref", float("nan"))
        print(f"  {v:>10.4f}", end="")
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

fig, ax = plt.subplots(figsize=(5.5, 4.2))

x = np.arange(len(STAGES))

for cl in CLS:
    y = []
    for stage_key, _ in STAGES:
        v = all_results[cl].get(stage_key, {}).get("pref", float("nan"))
        y.append(v)
    ax.plot(x, y, color=CL_COLORS[cl], linewidth=1.5,
            marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.5,
            label=CL_LABELS[cl], zorder=2)

ax.axhline(y=0, color=C_TEXT, linewidth=0.5, alpha=0.4, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in STAGES], fontsize=9, color=C_TEXT,
                   rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel("Representation stage", fontsize=10, color=C_TEXT)
ax.set_ylabel("Perturbed-reference Pearson", fontsize=10, color=C_TEXT)

ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, fontsize=8, loc="center left",
          bbox_to_anchor=(1.02, 0.5), labelspacing=0.6)

plt.tight_layout(rect=[0, 0, 0.80, 1])
fig.savefig(OUT / "fig2e_morph_mechanism.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig2e_morph_mechanism.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Save ──
save_data = {}
for cl in CLS:
    save_data[cl] = {}
    for stage_key, stage_label in STAGES:
        save_data[cl][stage_key] = {
            "label": stage_label,
            "pref": all_results[cl].get(stage_key, {}).get("pref"),
            "std": all_results[cl].get(stage_key, {}).get("std"),
        }

with open(OUT / "fig2e_morph_mechanism_results.json", "w") as f:
    json_mod.dump(save_data, f, indent=2)

print(f"\nSaved to {OUT}")