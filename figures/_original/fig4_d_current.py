"""Saved 2x2 interaction comparison with the recorded fonts and grid."""
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

_base = runpy.run_path(str(Path(__file__).with_name("component_base.py")))
frozen = SimpleNamespace(**{name: _base[name] for name in ("CELLS", "LABELS", "MODELS", "MODEL_LABELS")})
STYLES = {"source_only": ("#6F5CC4", "o", "No target perturbed cells"),
          "target30": ("#D26A3A", "o", "+30% target perturbations")}
STEM = "component_matching_gamma_target30_comparison_2x2_largefont_v8_grid"
FONT_SCALE = 1.4

def _base_enlarge(figure, scale=FONT_SCALE):
    """Scale each existing text artist once, before any output format is saved."""
    if getattr(figure, "_matching_largefont_applied", False):
        return
    for artist in figure.findobj(match=Text):
        artist.set_fontsize(artist.get_fontsize() * scale)
    if len(figure.axes) == 12:
        # The alpha panels' original dense numeric ticks touch at this size.
        # Keep their limits and plotted data, using fewer major labels.
        for axis in figure.axes:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            axis.tick_params(axis="x", labelsize=9 * scale)
    else:
        # Separate the enlarged bottom axis labels from the unchanged legend.
        figure.subplots_adjust(bottom=.15)
    figure._matching_largefont_applied = True

BASE_ENLARGE = _base_enlarge

def _layout_enlarge(figure):
    if getattr(figure, "_matching_extralarge_layout", False):
        return
    BASE_ENLARGE(figure, scale=2.)
    for axis in figure.axes:
        axis.set_xlabel("")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        axis.tick_params(axis="x", labelsize=18)
    if len(figure.axes) == 12:
        figure.set_size_inches(14., 14.)
        figure.axes[1].set_title("Shared perturbation\neffect (β)", fontsize=22)
        figure.subplots_adjust(left=.21, right=.985, top=.92, bottom=.12,
                               hspace=.28, wspace=.17)
        figure.supxlabel("Matching-component Pearson", x=.60, y=.065, fontsize=20)
    else:
        figure.set_size_inches(10.4, 8.)
        figure.subplots_adjust(left=.24, right=.985, top=.88, bottom=.19,
                               wspace=.17, hspace=.30)
        figure.supxlabel("Matching-component Pearson", x=.61, y=.095, fontsize=20)
    figure._matching_extralarge_layout = True

layout = SimpleNamespace(enlarge_text=_layout_enlarge)

def resize_text(figure):
    if getattr(figure, "_matching_medium_font_applied", False):
        return
    layout.enlarge_text(figure)
    for artist in figure.findobj(match=Text):
        artist.set_fontsize(artist.get_fontsize() * .85)
    for axis in figure.axes:
        axis.tick_params(axis="both", labelsize=15.3)
    figure._matching_medium_font_applied = True

previous = SimpleNamespace(approved=SimpleNamespace(resize_text=resize_text))

def style(figure):
    previous.approved.resize_text(figure)
    for axis in figure.axes:
        for line in (*axis.get_xgridlines(), *axis.get_ygridlines()):
            line.set_linewidth(1.5)

def _draw(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 6.8), sharex=True, sharey=True)
    for panel, target in enumerate(frozen.CELLS):
        axis = axes.flat[panel]
        for index, model in enumerate(frozen.MODELS):
            entries = summary[model][target]
            if all(setting in entries for setting in STYLES):
                axis.plot([entries[setting]["gamma"] for setting in STYLES], [index, index],
                          color="#CDD2D8", linewidth=1, zorder=1)
            for setting, (color, marker, _) in STYLES.items():
                if setting not in entries:
                    continue
                entry = entries[setting]
                axis.errorbar(entry["gamma"], index, xerr=entry["gamma_std"], fmt=marker,
                              color=color, ecolor=color, markersize=4.5, capsize=2, linewidth=.9,
                              markeredgecolor="white", markeredgewidth=.4)
        axis.set_yticks(range(len(frozen.MODELS)), [frozen.MODEL_LABELS[m] for m in frozen.MODELS])
        axis.set_ylim(len(frozen.MODELS) - .5, -.5)
        axis.axvline(0, color="#6B7280", linewidth=.6)
        axis.grid(axis="x", color="#E5E7EB", linewidth=.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(frozen.LABELS[target], fontsize=11)
        if panel >= 2:
            axis.set_xlabel("Matching-component Pearson", fontsize=10)
    lower, upper = axes[0, 0].get_xlim()
    axes[0, 0].set_xlim(max(-1., lower), min(1., upper))
    handles = [Line2D([0], [0], color=color, marker=marker, linestyle="none", markersize=6, label=label)
               for color, marker, label in STYLES.values()]
    figure.suptitle("Interaction effect (γ)", fontsize=11, y=.985)
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                  bbox_to_anchor=(.55, .005))
    figure.subplots_adjust(left=.17, right=.985, top=.91, bottom=.12, wspace=.15, hspace=.30)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(output / f"{STEM}.{extension}", dpi=250, bbox_inches="tight")
    plt.close(figure)

def plot(summary, output):
    original_savefig = Figure.savefig

    def savefig(figure, *args, **kwargs):
        style(figure)
        return original_savefig(figure, *args, **kwargs)

    with patch.object(Figure, "savefig", savefig):
        _draw(summary, output)
