"""
Figure: Cross-cell-line pathway prediction.

Panel A — Absolute pathway recovery
Panel B — Model-specific gain/loss vs Ridge

Loads: figure_pathway_crosscl/crosscl_pathway_results.json
Run: python3 fig_pathway_crosscl.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BENCH = Path("/mnt/storage01/home/amolina/benchmark_2026")
DATA = BENCH / "scripts" / "figure_pathway_crosscl"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "figure_pathway"
OUT.mkdir(exist_ok=True)

CLS = ["k562", "rpe1", "hepg2", "jurkat"]

C_TEXT = "#1F2937"
C_GRID = "#E5E7EB"

MODEL_ORDER = ["Ridge (DepMap)", "MLP (DepMap)", "Training pertrubation mean", "MORPH", "STATE"]
MODEL_LABELS = ["Ridge\n(DepMap)", "MLP\n(DepMap)", "Direct\ntransfer", "MORPH", "STATE"]
MODEL_COLORS = {
    "Ridge (DepMap)": "#149E9A",
    "MLP (DepMap)": "#0D7377",
    "Training pertrubation mean": "#7AA6C2",
    "MORPH": "#6F5CC4",
    "STATE": "#8A94A6",
}

N_SHOW = 28


def main():
    with open(DATA / "crosscl_pathway_results.json") as f:
        raw = json.load(f)

    results = {}
    for key, val in raw.items():
        cl, pw, model = key.split("||")
        results[(cl, pw, model)] = val

    all_pathways = sorted(set(pw for _, pw, _ in results))

    # Average across CLs
    pw_scores = {}
    for pw in all_pathways:
        for model in MODEL_ORDER:
            vals = [results.get((cl, pw, model), np.nan) for cl in CLS]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                pw_scores[(pw, model)] = np.mean(vals)

    # Determine which models have data
    models_present = [m for m in MODEL_ORDER
                      if any(pw_scores.get((pw, m), None) is not None for pw in all_pathways)]
    labels_present = [MODEL_LABELS[MODEL_ORDER.index(m)] for m in models_present]

    # Sort by Training pertrubation mean descending
    sort_key = "Training pertrubation mean"
    pw_sort = {pw: pw_scores.get((pw, sort_key), 0) for pw in all_pathways}
    sorted_pw = sorted(all_pathways, key=lambda pw: -pw_sort[pw])

    show_pw = [pw for pw in sorted_pw
               if max(pw_scores.get((pw, m), 0) for m in models_present) > 0.05]
    show_main = show_pw[:N_SHOW]
    show_supp = show_pw

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "text.color": C_TEXT,
        "axes.linewidth": 0.5, "axes.edgecolor": C_TEXT,
        "xtick.color": C_TEXT, "ytick.color": C_TEXT,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    # ── Panel A: Absolute ──
    def plot_absolute(pathways, suffix):
        mat = np.zeros((len(pathways), len(models_present)))
        for pi, pw in enumerate(pathways):
            for mi, model in enumerate(models_present):
                mat[pi, mi] = pw_scores.get((pw, model), 0)

        fig_h = max(4, len(pathways) * 0.30 + 1.5)
        fig, ax = plt.subplots(figsize=(5.5, fig_h))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.85)

        ax.set_xticks(range(len(models_present)))
        ax.set_xticklabels(labels_present, fontsize=8, rotation=45,
                           ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(pathways)))
        ax.set_yticklabels(pathways, fontsize=7)

        for pi in range(len(pathways)):
            for mi in range(len(models_present)):
                v = mat[pi, mi]
                if v > 0.15:
                    color = "white" if v > 0.55 else C_TEXT
                    ax.text(mi, pi, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5, color=color)

        cb = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cb.set_label("Pathway activity prediction (r)", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(OUT / f"crosscl_pathway_absolute_{suffix}.svg", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"crosscl_pathway_absolute_{suffix}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    plot_absolute(show_main, "main")
    plot_absolute(show_supp, "supplement")
    print(f"Saved Panel A (main: {len(show_main)}, supplement: {len(show_supp)})")

    # ── Panel B: Difference vs Ridge ──
    ridge_model = "Ridge (DepMap)"
    diff_models = [m for m in models_present if m != ridge_model]
    diff_labels = [f"{MODEL_LABELS[MODEL_ORDER.index(m)]}\n\u2212 Ridge" for m in diff_models]

    def plot_diff(pathways, suffix):
        mat_diff = np.zeros((len(pathways), len(diff_models)))
        for pi, pw in enumerate(pathways):
            ridge_val = pw_scores.get((pw, ridge_model), 0)
            for di, dm in enumerate(diff_models):
                mat_diff[pi, di] = pw_scores.get((pw, dm), 0) - ridge_val

        fig_h = max(4, len(pathways) * 0.30 + 1.5)
        fig, ax = plt.subplots(figsize=(5, fig_h))
        vmax_d = max(abs(mat_diff.min()), abs(mat_diff.max()), 0.2)
        im = ax.imshow(mat_diff, aspect="auto", cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)

        ax.set_xticks(range(len(diff_models)))
        ax.set_xticklabels(diff_labels, fontsize=8, rotation=45,
                           ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(pathways)))
        ax.set_yticklabels(pathways, fontsize=7)

        for pi in range(len(pathways)):
            for di in range(len(diff_models)):
                v = mat_diff[pi, di]
                if abs(v) > 0.05:
                    color = "white" if abs(v) > vmax_d * 0.65 else C_TEXT
                    ax.text(di, pi, f"{v:+.2f}", ha="center", va="center",
                            fontsize=5.5, color=color)

        cb = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cb.set_label("\u0394r vs Ridge (DepMap)", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(OUT / f"crosscl_pathway_diff_{suffix}.svg", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"crosscl_pathway_diff_{suffix}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    plot_diff(show_main, "main")
    plot_diff(show_supp, "supplement")
    print(f"Saved Panel B")

    # Summary
    print(f"\n{'Pathway':<35} " + "  ".join(f"{m[:12]:>12}" for m in models_present))
    print("-" * (35 + 14 * len(models_present)))
    for pw in show_main:
        vals = [pw_scores.get((pw, m), 0) for m in models_present]
        best = max(vals)
        line = f"{pw:<35} "
        for v in vals:
            marker = "*" if v == best and v > 0.05 else " "
            line += f" {v:>11.3f}{marker}"
        print(line)

    print(f"\nAll saved to {OUT}")


if __name__ == "__main__":
    main()