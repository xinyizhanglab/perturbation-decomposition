"""Original plotting statements, with input/output locations supplied by the launcher."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

CELLS = ("k562", "rpe1", "hepg2", "jurkat")
LABELS = dict(zip(CELLS, ("K562", "RPE1", "HepG2", "Jurkat")))
SETTINGS = ("source_only", "target30")
MODELS = ("ridge_1to1", "ridge_3to1", "mlp_1to1", "mlp_3to1", "morph", "state", "biolord", "scgen")
MODEL_LABELS = dict(zip(MODELS, ("Ridge 1→1", "Ridge 3→1", "MLP 1→1", "MLP 3→1", "MORPH 1→1", "STATE 3→1", "BioLORD 3→1", "scGen 3→1")))
STEM = "component_alignment_all_models_v1"
def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    styles = {"source_only": ("#6F5CC4", "o", "No target perturbed cells"),
              "controls": ("#AEB6C2", "D", "Target-control fine-tuning"),
              "target30": ("#D26A3A", "o", "+30% target perturbations")}
    figure, axes = plt.subplots(4, 3, figsize=(12.8, 12.8), sharex="col", sharey=True)
    for row, target in enumerate(CELLS):
        for column, (metric, title) in enumerate((("alpha", "Cell-line effect (α)"),
                                                 ("beta", "Shared perturbation effect (β)"),
                                                 ("gamma", "Interaction effect (γ)"))):
            axis = axes[row, column]
            for index, model in enumerate(MODELS):
                entries = summary[model][target]
                if "source_only" in entries and "target30" in entries:
                    axis.plot([entries[s][metric] for s in SETTINGS], [index, index],
                              color="#CDD2D8", linewidth=1, zorder=1)
                for setting, entry in entries.items():
                    color, marker, _ = styles[setting]
                    axis.errorbar(entry[metric], index, xerr=entry[f"{metric}_std"],
                                  fmt=marker, color=color, ecolor=color, markersize=4.5,
                                  capsize=2, linewidth=.9, markeredgecolor="white", markeredgewidth=.4)
            axis.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
            axis.set_ylim(len(MODELS) - .5, -.5)
            axis.axvline(0, color="#6B7280", linewidth=.6)
            axis.grid(axis="x", color="#E5E7EB", linewidth=.6)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.set_title(title, fontsize=11)
            if column == 0:
                sizes = summary["state"][target]["source_only"]["test_sizes"]
                axis.set_ylabel(f"{LABELS[target]}\nn = {min(sizes)}–{max(sizes)} per repeat", fontsize=10)
            if row == 3:
                axis.set_xlabel("Mean component-alignment Pearson", fontsize=10)
    for column in range(3):
        lower, upper = axes[0, column].get_xlim()
        axes[0, column].set_xlim(max(-1., lower), min(1., upper))
    handles = [Line2D([0], [0], color=color, marker=marker, linestyle="none", markersize=6,
                      label=label) for color, marker, label in styles.values()]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.57, .008))
    figure.subplots_adjust(left=.17, right=.985, top=.96, bottom=.075, hspace=.26, wspace=.16)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(output / f"{STEM}.{extension}", dpi=250, bbox_inches="tight")
    plt.close(figure)
