"""
Figure: Pathway enrichment of DepMap-favored vs source-response-favored perturbations.

4-panel horizontal bar plot (one per target CL).
  left = source-response-favored pathways
  right = DepMap-favored pathways
  x-axis = signed log2(fold enrichment)

Loads results from depmap_vs_source_enrichment.py.
Usage: python3 fig_enrichment_barplot.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pickle

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
GO_DIR = BENCH / "data" / "go"
ENRICH_DIR = BENCH / "scripts" / "figure_enrichment"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure3"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"
C_DEPMAP = "#3572B0"
C_SOURCE = "#D26A3A"

FDR_THRESH = 0.10
MAX_TERMS_PER_SIDE = 7


def load_data():
    with open(ENRICH_DIR / "simple_enrichment_results.json") as f:
        simple = json.load(f)

    # Load GO annotations for redundancy filtering
    with open(GO_DIR / "go_annotations.pkl", "rb") as f:
        gene_to_bp = pickle.load(f)

    # Build term -> gene set (in perturbation universe)
    # Get perturbation universe from results
    with open(BENCH / "scripts" / "figure_source_vs_depmap" / "source_vs_depmap_results.json") as f:
        src_res = json.load(f)
    all_perts = set()
    for cl in CLS:
        for p in src_res[cl]["per_pert"]:
            all_perts.add(p["gene"])

    term_to_genes = defaultdict(set)
    for gene in all_perts:
        if gene in gene_to_bp:
            for term in gene_to_bp[gene]:
                term_to_genes[term].add(gene)

    return simple, term_to_genes


def remove_redundant(terms, term_to_genes, max_overlap=0.5):
    """Remove redundant GO terms by gene overlap. Keep better FDR."""
    if not terms:
        return terms

    # Sort by FDR (best first)
    sorted_terms = sorted(terms, key=lambda x: x["fdr"])
    kept = []
    kept_gene_sets = []

    for t in sorted_terms:
        t_genes = term_to_genes.get(t["term"], set())
        if not t_genes:
            continue

        redundant = False
        for kg in kept_gene_sets:
            overlap = len(t_genes & kg) / min(len(t_genes), len(kg))
            if overlap > max_overlap:
                redundant = True
                break

        if not redundant:
            kept.append(t)
            kept_gene_sets.append(t_genes)

        if len(kept) >= MAX_TERMS_PER_SIDE:
            break

    return kept


def make_plot(simple, term_to_genes):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.linewidth": 0.5,
        "axes.edgecolor": C_TEXT,
        "xtick.color": C_TEXT,
        "ytick.color": C_TEXT,
        "text.color": C_TEXT,
    })

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ci, cl in enumerate(CLS):
        ax = axes[ci]
        res = simple[cl]

        # Separate DepMap-favored and source-favored significant terms
        dep_sig = [r for r in res if r["fdr_depmap"] < FDR_THRESH and r["fold_depmap"] > 1]
        src_sig = [r for r in res if r["fdr_source"] < FDR_THRESH and r["fold_source"] > 1]

        # Add FDR field for redundancy filtering
        for r in dep_sig:
            r["fdr"] = r["fdr_depmap"]
        for r in src_sig:
            r["fdr"] = r["fdr_source"]

        # Remove redundant terms
        dep_kept = remove_redundant(dep_sig, term_to_genes)
        src_kept = remove_redundant(src_sig, term_to_genes)

        # Build items for plotting
        items = []

        # Source-favored (negative x)
        for r in reversed(src_kept):
            score = -np.log2(max(r["fold_source"], 1.01))
            fdr = r["fdr_source"]
            name = r["name"][:45]
            items.append({"name": name, "score": score, "fdr": fdr, "direction": "source"})

        # DepMap-favored (positive x)
        for r in dep_kept:
            score = np.log2(max(r["fold_depmap"], 1.01))
            fdr = r["fdr_depmap"]
            name = r["name"][:45]
            items.append({"name": name, "score": score, "fdr": fdr, "direction": "depmap"})

        if not items:
            ax.text(0.5, 0.5, "No significant\nenrichments",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=C_TEXT, alpha=0.5)
            ax.set_title(CL_LABELS[cl], fontsize=11, fontweight="bold", color=C_TEXT)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue

        y = np.arange(len(items))
        scores = [it["score"] for it in items]
        colors = [C_SOURCE if it["direction"] == "source" else C_DEPMAP for it in items]
        alphas = [0.9 if it["fdr"] < 0.05 else 0.55 for it in items]

        bars = ax.barh(y, scores, color=colors, edgecolor="white", height=0.7)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        # FDR annotations
        for i, it in enumerate(items):
            if it["fdr"] < 0.01:
                marker = "**"
            elif it["fdr"] < 0.05:
                marker = "*"
            else:
                marker = ""
            if marker:
                xpos = scores[i]
                ha = "left" if xpos >= 0 else "right"
                offset = 0.05 if xpos >= 0 else -0.05
                ax.text(xpos + offset, i, marker, ha=ha, va="center",
                        fontsize=8, fontweight="bold", color=C_TEXT)

        ax.axvline(0, color=C_TEXT, linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([it["name"] for it in items], fontsize=6.5)
        ax.set_title(CL_LABELS[cl], fontsize=11, fontweight="bold", color=C_TEXT, pad=8)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

        if ci == 0:
            ax.set_xlabel("Signed log\u2082 fold enrichment", fontsize=9, color=C_TEXT)

        # Side labels at top
        xlim = ax.get_xlim()
        ax.text(xlim[0] * 0.5, -0.8, "Source-favored", fontsize=7, color=C_SOURCE,
                ha="center", va="bottom", fontstyle="italic")
        ax.text(xlim[1] * 0.5, -0.8, "DepMap-favored", fontsize=7, color=C_DEPMAP,
                ha="center", va="bottom", fontstyle="italic")

    plt.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig3_enrichment_barplot.svg", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig3_enrichment_barplot.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUT / 'fig3_enrichment_barplot.svg'}")
    print(f"       + {OUT / 'fig3_enrichment_barplot.pdf'}")


def main():
    simple, term_to_genes = load_data()
    make_plot(simple, term_to_genes)


if __name__ == "__main__":
    main()