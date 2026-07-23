"""
Figure 1d: Template GSEA enrichment.
Signed NES heatmap: cell lines × Hallmark + lineage signatures.
GSEA ranked on FULL gene set (6642 genes) so lineage markers are included.
Cells where the signature has <10 genes in the 2000 HVG are shown with
light gray background and asterisk.

Run: python3 fig1_template_gsea.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pathlib import Path
import anndata as ad
import gseapy as gp
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure1"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}

# ── Colors ──
C_TEXT = "#1F2937"
C_NEG = "#3B5BA9"
C_ZERO = "#FFFFFF"
C_POS = "#D26A3A"
C_GRAY_BG = "#E8EAED"

# ── Hallmark sets ──
HALLMARK_SHOW = [
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
]

HALLMARK_LABELS = {
    "HALLMARK_G2M_CHECKPOINT": "G2M\ncheckpoint",
    "HALLMARK_E2F_TARGETS": "E2F\ntargets",
    "HALLMARK_P53_PATHWAY": "p53\npathway",
    "HALLMARK_APOPTOSIS": "Apoptosis",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": "UPR",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": "OxPhos",
}

# ── Lineage markers: PanglaoDB (Franzen et al., Database, 2019) ──
LINEAGE_SETS = {
    "LINEAGE_EPITHELIAL": {
        "label": "Epithelial\nidentity",
        "genes": [
            "EPCAM", "KRT18", "KRT8", "KRT19", "CDH1", "CLDN3", "CLDN4",
            "CLDN7", "OCLN", "TJP1", "MUC1", "TACSTD2", "GRHL2", "ELF3",
            "KRT7", "LAMB3", "LAMC2", "LAMA3", "DSP", "JUP", "PKP3",
            "CGN", "F11R", "ESRP1", "ESRP2", "IRF6", "OVOL2",
        ],
    },
    "LINEAGE_HEPATOCYTE": {
        "label": "Hepatocyte\nidentity",
        "genes": [
            "ALB", "APOB", "APOA1", "APOC3", "HNF4A", "HNF1A", "SERPINA1",
            "FGA", "FGB", "FGG", "CYP3A4", "CYP1A2", "CYP2E1", "CYP2C9",
            "TTR", "TF", "AHSG", "HPX", "GC", "ITIH1", "ITIH2", "AMBP",
            "RBP4", "ALDOB", "PCK1", "G6PC", "TAT", "HPD", "FAH",
            "CPS1", "OTC", "ASS1", "ARG1", "ASGR1", "ASGR2",
        ],
    },
    "LINEAGE_TCELL": {
        "label": "T cell\nidentity",
        "genes": [
            "CD3D", "CD3E", "CD3G", "CD2", "CD7", "LCK", "ZAP70", "LAT",
            "ITK", "GATA3", "BCL11B", "TCF7", "LEF1", "IL7R", "CD27",
            "CD28", "PTPRC", "IL2RG", "CD5", "TRAC", "TRBC1", "TRBC2",
            "CD8A", "CD8B", "CD4", "FOXP3", "TBX21", "EOMES",
        ],
    },
}

HVG_OVERLAP_THRESHOLD = 10

# ── GSEA ──
def run_preranked(ranked_series, gene_sets, min_size=5, n_perm=1000):
    try:
        res = gp.prerank(
            rnk=ranked_series,
            gene_sets=gene_sets,
            min_size=min_size,
            max_size=500,
            permutation_num=n_perm,
            seed=42,
            no_plot=True,
            verbose=False,
        )
        df = res.res2d[["Term", "NES", "FDR q-val"]].copy()
        df.columns = ["Term", "NES", "FDR"]
        df["NES"] = df["NES"].astype(float)
        df["FDR"] = df["FDR"].astype(float)
        return df
    except Exception as e:
        print(f"    GSEA error: {e}")
        return pd.DataFrame(columns=["Term", "NES", "FDR"])

# ── Build Hallmark gene sets ──
print("Loading Hallmark gene sets...")
try:
    hallmark_lib = gp.get_library("MSigDB_Hallmark_2020")
    print(f"  gseapy built-in: {len(hallmark_lib)} sets")
except Exception:
    hallmark_lib = {}
    print("  gseapy built-in failed")

hallmark_map = {}
for h in HALLMARK_SHOW:
    short = h.replace("HALLMARK_", "").replace("_", " ").title()
    for k in hallmark_lib:
        if short.lower().replace(" ", "") in k.lower().replace(" ", ""):
            hallmark_map[h] = k
            break
    if h not in hallmark_map:
        for k in hallmark_lib:
            if any(w in k.lower() for w in short.lower().split()[:2]):
                hallmark_map[h] = k
                break

print("  Hallmark mapping:")
for h in HALLMARK_SHOW:
    mapped = hallmark_map.get(h, "NOT FOUND")
    n = len(hallmark_lib.get(mapped, [])) if mapped != "NOT FOUND" else 0
    print(f"    {h} -> {mapped} ({n} genes)")

# Column order
hallmark_valid = [h for h in HALLMARK_SHOW if h in hallmark_map]
lineage_keys = list(LINEAGE_SETS.keys())
col_order = hallmark_valid + lineage_keys
col_labels = [HALLMARK_LABELS.get(h, h) for h in hallmark_valid] + \
             [LINEAGE_SETS[k]["label"] for k in lineage_keys]

# Assemble all gene sets
all_gene_sets = {}
for h in hallmark_valid:
    all_gene_sets[h] = hallmark_lib[hallmark_map[h]]
for k, v in LINEAGE_SETS.items():
    all_gene_sets[k] = v["genes"]

print(f"\n  Final gene sets ({len(all_gene_sets)}):")
for s in col_order:
    print(f"    {s}: {len(all_gene_sets[s])} genes")

# ── Run GSEA per cell line using FULL gene set ──
n_cols = len(col_order)
nes_matrix = np.full((len(CLS), n_cols), np.nan)
fdr_matrix = np.full((len(CLS), n_cols), np.nan)
overlap_full_matrix = np.zeros((len(CLS), n_cols), dtype=int)
overlap_hvg_matrix = np.zeros((len(CLS), n_cols), dtype=int)

for ci, cl in enumerate(CLS):
    print(f"\n  {cl.upper()}: loading data...")
    a = ad.read_h5ad(BENCH / "data" / cl / "replogle.h5ad")

    X_full = np.array(a.X)
    G_full = X_full.shape[1]
    all_gene_names = list(a.var_names)
    all_gene_set = set(all_gene_names)
    genes_obs = a.obs["gene"].astype(str).values
    ctrl_mean = X_full[genes_obs == "non-targeting"].mean(0)

    hvg_mask = a.var["highly_variable"]
    hvg_names = set(a.var_names[hvg_mask])

    print(f"    Full: {G_full} genes, HVG: {len(hvg_names)} genes")

    # Template on full genes
    deltas = []
    for pn in sorted(set(genes_obs) - {"non-targeting"}):
        if (genes_obs == pn).sum() >= 5:
            deltas.append(X_full[genes_obs == pn].mean(0) - ctrl_mean)
    D = np.array(deltas)
    T = D.mean(0)

    # Ranked gene list
    ranked = pd.Series(T, index=all_gene_names)
    ranked = ranked[~ranked.index.duplicated()]
    ranked = ranked.sort_values(ascending=False)

    # Gene overlap
    for si, sname in enumerate(col_order):
        sg = all_gene_sets[sname]
        overlap_full_matrix[ci, si] = sum(1 for g in sg if g in all_gene_set)
        overlap_hvg_matrix[ci, si] = sum(1 for g in sg if g in hvg_names)

    for k in lineage_keys:
        idx = col_order.index(k)
        print(f"    {k}: {overlap_full_matrix[ci, idx]} in full, {overlap_hvg_matrix[ci, idx]} in HVG")

    # GSEA
    print(f"    Running preranked GSEA on {G_full} genes...")
    gsea_df = run_preranked(ranked, all_gene_sets, min_size=5, n_perm=1000)

    for si, sname in enumerate(col_order):
        row = gsea_df[gsea_df["Term"] == sname]
        if len(row) > 0:
            nes_matrix[ci, si] = float(row["NES"].iloc[0])
            fdr_matrix[ci, si] = float(row["FDR"].iloc[0])

# ── Plot ──
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.linewidth": 0.6,
    "axes.edgecolor": C_TEXT,
    "xtick.color": C_TEXT,
    "ytick.color": C_TEXT,
    "text.color": C_TEXT,
})

valid_cols = [si for si in range(n_cols) if not np.all(np.isnan(nes_matrix[:, si]))]
plot_nes = nes_matrix[:, valid_cols]
plot_fdr = fdr_matrix[:, valid_cols]
plot_overlap_hvg = overlap_hvg_matrix[:, valid_cols]
plot_overlap_full = overlap_full_matrix[:, valid_cols]
plot_labels = [col_labels[si] for si in valid_cols]

fig, ax = plt.subplots(figsize=(0.85 * len(valid_cols) + 1.8, 2.8))

cmap = LinearSegmentedColormap.from_list("blue_white_terra",
    [C_NEG, "#8FADD4", C_ZERO, "#E8A882", C_POS])

vmax = np.nanmax(np.abs(plot_nes)) * 1.05
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(plot_nes, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(range(len(valid_cols)))
ax.set_xticklabels(plot_labels, fontsize=8.5, color=C_TEXT, ha="center")
ax.set_yticks(range(len(CLS)))
ax.set_yticklabels([CL_LABELS[cl] for cl in CLS], fontsize=10, color=C_TEXT)

# Gray background + asterisk for low HVG overlap; NES values; FDR dots
# Consistent FDR marker style
C_FDR_FACE = "white"
C_FDR_EDGE = C_TEXT
FDR_SIZE = 14

# Gray background + asterisk for low HVG overlap; NES values; FDR dots
for ci in range(len(CLS)):
    for si in range(len(valid_cols)):
        nes_val = plot_nes[ci, si]
        fdr_val = plot_fdr[ci, si]
        hvg_ov = plot_overlap_hvg[ci, si]

        if np.isnan(nes_val):
            continue

        if hvg_ov < HVG_OVERLAP_THRESHOLD:
            rect = Rectangle(
                (si - 0.5, ci - 0.5), 1, 1,
                facecolor=C_GRAY_BG,
                edgecolor="none",
                zorder=1.5
            )
            ax.add_patch(rect)

        text_color = "white" if abs(nes_val) > vmax * 0.55 and hvg_ov >= HVG_OVERLAP_THRESHOLD else C_TEXT

        label = f"{nes_val:.1f}"
        if hvg_ov < HVG_OVERLAP_THRESHOLD:
            label += "*"

        ax.text(
            si, ci, label,
            ha="center", va="center",
            fontsize=7.5,
            color=text_color,
            zorder=3
        )

        if fdr_val < 0.05:
            ax.scatter(
                si + 0.32, ci - 0.32,
                s=FDR_SIZE,
                marker="o",
                facecolors=C_FDR_FACE,
                edgecolors=C_FDR_EDGE,
                linewidths=0.55,
                zorder=4
            )
# Vertical separator between Hallmark and lineage
n_hallmark_valid = sum(1 for si in valid_cols if col_order[si] in hallmark_valid)
if 0 < n_hallmark_valid < len(valid_cols):
    ax.axvline(x=n_hallmark_valid - 0.5, color=C_TEXT, linewidth=0.5,
               linestyle="--", alpha=0.3)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Template enrichment NES", fontsize=9, color=C_TEXT)
cbar.ax.tick_params(labelsize=8, colors=C_TEXT)
cbar.outline.set_edgecolor(C_TEXT)
cbar.outline.set_linewidth(0.5)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

# Legend — consistent with markers in heatmap
legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor=C_FDR_FACE,
        markeredgecolor=C_FDR_EDGE,
        markeredgewidth=0.7,
        markersize=4.5,
        label="FDR < 0.05"
    )
]

# Add this only if low-HVG asterisks can appear in the plotted columns
if np.any(plot_overlap_hvg < HVG_OVERLAP_THRESHOLD):
    legend_elements.append(
        Line2D(
            [0], [0],
            marker="$*$",
            linestyle="None",
            color=C_TEXT,
            markersize=7,
            label=f"<{HVG_OVERLAP_THRESHOLD} genes in HVG"
        )
    )

fig.legend(
    handles=legend_elements,
    loc="center right",
    frameon=False,
    fontsize=7.5,
    bbox_to_anchor=(1.20, 0.5),
    labelspacing=1.0,
    handletextpad=0.6
)

plt.tight_layout(rect=[0, 0, 0.88, 1])
fig.savefig(OUT / "fig1d_template_gsea.svg", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "fig1d_template_gsea.pdf", dpi=300, bbox_inches="tight")
plt.close()

# ── Print summary ──
print(f"\n{'='*120}")
print("GSEA NES Summary (dot = FDR < 0.05, * = <10 genes in HVG)")
print(f"{'='*120}")
print(f"{'Signature':<25}", end="")
for cl in CLS:
    print(f"  {CL_LABELS[cl]:>22}", end="")
print()
print("-" * 120)

for si in valid_cols:
    sname = col_order[si]
    label = col_labels[si].replace("\n", " ")
    print(f"{label:<25}", end="")
    for ci in range(len(CLS)):
        nes = nes_matrix[ci, si]
        fdr = fdr_matrix[ci, si]
        ov_full = overlap_full_matrix[ci, si]
        ov_hvg = overlap_hvg_matrix[ci, si]
        if np.isnan(nes):
            print(f"  {'--':>22}", end="")
        else:
            fdr_mark = "•" if fdr < 0.05 else " "
            hvg_mark = "*" if ov_hvg < HVG_OVERLAP_THRESHOLD else " "
            print(f"  {nes:>5.2f}{fdr_mark}{hvg_mark} (full={ov_full:>2}, hvg={ov_hvg:>2})", end="")
    print()

# Save raw results
results = {
    "cls": CLS,
    "signatures": [col_order[si] for si in valid_cols],
    "sig_labels": [col_labels[si].replace("\n", " ") for si in valid_cols],
    "nes": plot_nes.tolist(),
    "fdr": plot_fdr.tolist(),
    "overlap_full": plot_overlap_full.tolist(),
    "overlap_hvg": plot_overlap_hvg.tolist(),
}
with open(OUT / "fig1d_gsea_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUT}")