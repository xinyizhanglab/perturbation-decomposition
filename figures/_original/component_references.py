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

from unittest.mock import patch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from figures._original import component_base as frozen
COMPONENTS = ("alpha", "beta", "gamma")
def render(summary, references, output):
    """Reuse the exact renderer; overlay one black diamond in each panel."""
    original_savefig, original_ylabel = Figure.savefig, Axes.set_ylabel
    decorated = set()
    removed = []

    def without_sample_size(axis, label, *args, **kwargs):
        if "\nn = " in label:
            cell_line, annotation = label.split("\n", 1)
            if cell_line not in frozen.LABELS.values() or not annotation.endswith(" per repeat"):
                raise ValueError(f"Unexpected sample-size label: {label}")
            removed.append(annotation)
            label = cell_line
        return original_ylabel(axis, label, *args, **kwargs)

    def with_references(figure, *args, **kwargs):
        if figure not in decorated:
            if len(figure.axes) != 12:
                raise ValueError("Expected the frozen 4-by-3 component layout")
            # Retain existing limits unless extra room is needed for a reference.
            limits = [figure.axes[column].get_xlim() for column in range(3)]
            for row, target in enumerate(frozen.CELLS):
                for column, metric in enumerate(COMPONENTS):
                    axis = figure.axes[3 * row + column]
                    entry = references[target]
                    marker = axis.errorbar(
                        entry[metric], .985, xerr=entry[f"{metric}_std"],
                        transform=axis.get_xaxis_transform(), fmt="D", color="black",
                        ecolor="black", markersize=4.5, capsize=2, linewidth=.9,
                        markeredgecolor="white", markeredgewidth=.4, zorder=10,
                        clip_on=False, label="_nolegend_",
                    )
                    marker.lines[0].set_gid(f"perfect_prediction_reference_{target}_{metric}")
            for column, metric in enumerate(COMPONENTS):
                lower, upper = limits[column]
                padding = .04 * (upper - lower)
                for target in frozen.CELLS:
                    entry = references[target]
                    lower = min(lower, entry[metric] - entry[f"{metric}_std"] - padding)
                    upper = max(upper, entry[metric] + entry[f"{metric}_std"] + padding)
                figure.axes[column].set_xlim(max(-1., lower), min(1., upper))
            decorated.add(figure)
        return original_savefig(figure, *args, **kwargs)

    with patch.object(Axes, "set_ylabel", without_sample_size), patch.object(
            Figure, "savefig", with_references):
        frozen.plot(summary, output)
    if len(decorated) != 1 or len(removed) != 4:
        raise RuntimeError("Unexpected renderer behavior")
    svg = (output / f"{frozen.STEM}.svg").read_text()
    if svg.count('id="perfect_prediction_reference_') != 12 or "per repeat" in svg:
        raise RuntimeError("Missing references or unexpected sample-size text")
