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
from figures._original import component_base as frozen
from figures._original import component_references as reference_renderer
STEM = "component_matching_assembled_v1"
COMPONENTS = ("alpha", "beta", "gamma")
def render(summary, output):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    reference = {target: {**{m: 1. for m in COMPONENTS},
                          **{f"{m}_std": 0. for m in COMPONENTS}} for target in frozen.CELLS}
    original_xlabel, original_savefig = Axes.set_xlabel, Figure.savefig

    def xlabel(axis, label, *args, **kwargs):
        if label == "Mean component-alignment Pearson":
            label = "Matching-component Pearson"
        return original_xlabel(axis, label, *args, **kwargs)

    def savefig(figure, *args, **kwargs):
        # Reference renderer adds a black diamond at +1. Leave a little room to
        # its right so the marker remains inside the panel, with no extra text.
        for column in range(3):
            lower, _ = figure.axes[column].get_xlim()
            figure.axes[column].set_xlim(lower, 1.04)
        return original_savefig(figure, *args, **kwargs)

    with patch.object(frozen, "STEM", STEM), patch.object(Axes, "set_xlabel", xlabel), patch.object(
            Figure, "savefig", savefig):
        reference_renderer.render(summary, reference, output)
