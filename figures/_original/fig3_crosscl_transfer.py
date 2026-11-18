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
CELL_LABELS = ("K562", "RPE1", "HepG2", "Jurkat")
MODEL_STYLE = {
    "Ridge 1→1": ("#149E9A", "s", False, "Ridge (1 cell line in train)"),
    "Ridge 3→1": ("#149E9A", "s", True, "Ridge (3 cell lines in train)"),
    "MLP 1→1": ("#0D7377", "^", False, "MLP (1 cell line in train)"),
    "MLP 3→1": ("#0D7377", "^", True, "MLP (3 cell lines in train)"),
    "MORPH": ("#6F5CC4", "D", True, "MORPH"),
    "STATE": ("#8A94A6", "o", True, "STATE"),
    "BioLORD": ("#D98736", "P", True, "BioLORD (3 cell lines in train)*"),
    "scGen": ("#AA5C97", "D", True, "scGen (3 cell lines in train)"),
}
STEM = "fig3_crosscl_biolord_cellline_go_author_v1"
def plot(payload, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D


    plt.rcParams.update({"font.family": payload.get("font", "DejaVu Sans"), "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    figure.subplots_adjust(left=.11, right=.57, bottom=.23, top=.96)
    positions = np.arange(4) * 1.2
    offsets = np.linspace(-.38, .38, len(MODEL_STYLE))
    handles = []
    for index, (model, (color, marker, filled, label)) in enumerate(MODEL_STYLE.items()):
        label = model if model in ("BioLORD", "scGen") else label
        for ti, target in enumerate(CELL_LINES):
            entry = payload["results"][model][target]
            position = positions[ti] + offsets[index]
            if entry["std"] is not None:
                axis.errorbar(position, entry["mean"], yerr=entry["std"], fmt="none",
                              ecolor=color, capsize=2, linewidth=1)
            axis.scatter(position, entry["mean"], s=45, marker=marker,
                         facecolors=color if filled else "white", edgecolors="white" if filled else color,
                         linewidths=.5 if filled else 1.2, zorder=3)
        handles.append(Line2D([0], [0], marker=marker, color=color, linestyle="none", label=label,
                              markerfacecolor=color if filled else "white", markersize=6))
    axis.axhline(0, color="#1F2937", linewidth=.5, alpha=.4)
    axis.set_xticks(positions, CELL_LABELS, rotation=45, ha="right", rotation_mode="anchor")
    axis.set_xlim(positions[0] - .62, positions[-1] + .62)
    axis.set_ylabel("Perturbed-reference Pearson")
    axis.yaxis.grid(True, color="#E5E7EB", linewidth=.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(handles=handles, frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.04, .5))
    for extension in ("pdf", "png", "svg"):
        figure.savefig(output / f"{STEM}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)
