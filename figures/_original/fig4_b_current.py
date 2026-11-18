"""Saved independent component comparison without target perturbation calibration."""
import copy
from pathlib import Path
import runpy
from unittest.mock import patch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from figures._original import component_base as frozen

_styles = runpy.run_path(str(Path(__file__).with_name("fig4_d_current.py")))
MAIN_STEM = "component_matching_no_target30_largefont_v8_grid"

def without_target30(summary):
    return {model: {target: {setting: copy.deepcopy(entry)
                            for setting, entry in settings.items() if setting != "target30"}
                    for target, settings in contexts.items()} for model, contexts in summary.items()}

def render_main(summary, output):
    original_ylabel, original_xlabel, original_legend = Axes.set_ylabel, Axes.set_xlabel, Figure.legend

    def ylabel(axis, label, *args, **kwargs):
        if "\nn = " in label:
            label = label.split("\n", 1)[0]
            if label not in frozen.LABELS.values():
                raise ValueError("Unexpected sample-size label")
        return original_ylabel(axis, label, *args, **kwargs)

    def xlabel(axis, label, *args, **kwargs):
        if label == "Mean component-alignment Pearson":
            label = "Matching-component Pearson"
        return original_xlabel(axis, label, *args, **kwargs)

    def legend(figure, *args, **kwargs):
        kwargs["handles"] = [handle for handle in kwargs["handles"]
                             if handle.get_label() != "+30% target perturbations"]
        kwargs["ncol"] = 2
        return original_legend(figure, *args, **kwargs)

    with patch.object(frozen, "STEM", MAIN_STEM), patch.object(Axes, "set_ylabel", ylabel), patch.object(
            Axes, "set_xlabel", xlabel), patch.object(Figure, "legend", legend):
        frozen.plot(without_target30(summary), output)

def plot(summary, output):
    original_savefig = Figure.savefig

    def savefig(figure, *args, **kwargs):
        _styles["style"](figure)
        return original_savefig(figure, *args, **kwargs)

    with patch.object(Figure, "savefig", savefig):
        render_main(summary, output)
