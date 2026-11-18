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

CELL_LINES = ("k562", "rpe1", "hepg2", "jurkat")
LABELS = dict(zip(CELL_LINES, ("K562", "RPE1", "HepG2", "Jurkat")))
SETTINGS = ("source_only", "target30")
STEM = "fig4c_calibration_author_aligned_v2"
def plot(payload, output):
    rows = []
    for target in CELL_LINES:
        rows.append(("STATE", LABELS[target], [("source_only", payload["state"][target]["zeroshot"]),
                                                ("target30", payload["state"][target]["calibrated"])]))
    for source, target in (("k562", "rpe1"), ("rpe1", "k562"), ("hepg2", "jurkat"), ("jurkat", "hepg2")):
        entry = payload["morph"][f"{source}→{target}"]
        rows.append(("MORPH", f"{LABELS[source]}→{LABELS[target]}",
                     [(key, entry[name]) for key, name in (("source_only", "Source only"),
                      ("controls", "Fine-tune ctrl"), ("target30", "+30% target"))]))
    for model, label in (("biolord", "BioLORD"), ("scgen", "scGen")):
        for target in CELL_LINES:
            rows.append((label, LABELS[target], [(setting, payload[model][target][setting])
                                                 for setting in SETTINGS
                                                 if payload[model][target][setting].get("status") != "not_applicable"]))
    styles = {"source_only": ("#6F5CC4", "o", "No target perturbed cells"),
              "controls": ("#C9CED6", "D", "Target-control fine-tuning"),
              "target30": ("#D26A3A", "o", "+30% target perturbations")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 8.7), sharey=True)
    for axis, metric, title, xlabel in zip(axes, ("pref", "gamma"),
                                         ("Prediction performance", "Interaction recovery"),
                                         ("Perturbed-reference Pearson", "corr(δ̂, γ)")):
        for index, (_, _, entries) in enumerate(rows):
            axis.plot([entry[metric] for _, entry in entries], [index] * len(entries),
                      color="#C9CED6", linewidth=1.2, zorder=1)
            for setting, entry in entries:
                color, marker, _ = styles[setting]
                if entry.get(f"{metric}_std", 0) > 0:
                    axis.errorbar(entry[metric], index, xerr=entry[f"{metric}_std"], fmt="none",
                                  ecolor=color, capsize=2, linewidth=.9)
                axis.scatter(entry[metric], index, s=40, color=color, marker=marker,
                             edgecolor="white", linewidth=.5, zorder=3)
        # N/A is explicitly non-numeric: no orange point, line, or zero score.
        axis.text(.98, 13.5, "+30%: N/A", transform=axis.get_yaxis_transform(),
                  ha="right", va="center", color="#6B7280", fontsize=8)
        axis.set_ylim(len(rows) - .5, -.5)
        axis.set_yticks(range(len(rows)), [])
        axis.axvline(0, color="#1F2937", alpha=.4, linewidth=.6)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel, fontsize=10)
        axis.xaxis.grid(True, color="#E5E7EB", linewidth=.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for separator in (3.5, 7.5, 11.5):
            axis.axhline(separator, color="#E5E7EB", linewidth=.8)
    for index, (_, label, _) in enumerate(rows):
        axes[0].text(-.06, index, label, transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=8)
    for index, group in enumerate(("STATE", "MORPH", "BioLORD", "scGen")):
        axes[0].text(-.36, 1.5 + index * 4, group, rotation=90,
                     transform=axes[0].get_yaxis_transform(), ha="center", va="center", fontsize=9)
    handles = [Line2D([0], [0], marker=marker, color=color, linestyle="none", label=label,
                      markeredgecolor="white", markersize=6) for color, marker, label in styles.values()]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.55, .01))
    figure.subplots_adjust(left=.27, right=.98, top=.95, bottom=.11, wspace=.28)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(output / f"{STEM}.{extension}", dpi=250, bbox_inches="tight")
    plt.close(figure)
