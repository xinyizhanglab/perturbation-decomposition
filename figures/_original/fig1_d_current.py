"""Saved cell-line-template Hallmark drawing, including numeric NES labels."""
from types import SimpleNamespace
from unittest.mock import patch
from matplotlib.figure import Figure
import numpy as np

CELLS = ("k562", "rpe1", "hepg2", "jurkat")
TITLES = ("Cell-line template", "Cell-line deviation")
STEM = "fig1d_cell_line_template_mu_plus_alpha_numbers_v2"
constants = {'CL_LABELS': {'k562': 'K562', 'rpe1': 'RPE1', 'hepg2': 'HepG2', 'jurkat': 'Jurkat'}, 'C_TEXT': '#1F2937', 'C_NEG': '#3B5BA9', 'C_ZERO': '#FFFFFF', 'C_POS': '#D26A3A'}

def _render_matched(nes, fdr, labels, constants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    if nes.shape != fdr.shape or nes.shape[:2] != (2, 4) or not np.isfinite(nes).all():
        raise ValueError("Matched finite 2 x 4 x pathway matrices required")
    if not np.isfinite(fdr).all() or np.any((fdr < 0) | (fdr > 1)):
        raise ValueError("Invalid FDR matrix")
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.labelsize": 11,
                         "axes.linewidth": .6, "axes.edgecolor": constants["C_TEXT"],
                         "xtick.color": constants["C_TEXT"], "ytick.color": constants["C_TEXT"],
                         "text.color": constants["C_TEXT"], "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    cmap = LinearSegmentedColormap.from_list("blue_white_terra",
        [constants["C_NEG"], "#8FADD4", constants["C_ZERO"], "#E8A882", constants["C_POS"]])
    vmax = float(np.max(np.abs(nes)) * 1.05)
    vmax = vmax if vmax > 0 else 1.
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax)
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 3.25), sharey=True)
    figure.subplots_adjust(left=.07, right=.91, bottom=.20, top=.84, wspace=.16)
    images = []
    for ai, axis in enumerate(axes):
        images.append(axis.imshow(nes[ai], cmap=cmap, norm=norm, aspect="auto"))
        axis.set_xticks(range(len(labels)), labels, fontsize=8.5)
        axis.set_yticks(range(4), [constants["CL_LABELS"][c] for c in CELLS], fontsize=10)
        axis.set_title(TITLES[ai], fontsize=11, pad=9)
        for ci, si in np.argwhere(fdr[ai] < .05):
            axis.scatter(si + .32, ci - .32, s=14, marker="o", facecolors="white",
                         edgecolors=constants["C_TEXT"], linewidths=.55, zorder=4)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    color_axis = figure.add_axes([.93, .23, .012, .56])
    colorbar = figure.colorbar(images[0], cax=color_axis)
    colorbar.set_label("NES", fontsize=9, color=constants["C_TEXT"])
    colorbar.ax.tick_params(labelsize=8, colors=constants["C_TEXT"])
    colorbar.outline.set_edgecolor(constants["C_TEXT"])
    colorbar.outline.set_linewidth(.5)
    return figure, axes, vmax

matched = SimpleNamespace(render=_render_matched)

def _standalone(results, OUTPUT):
    columns = [results["all_gene_sets"].index(name) for name in results["plotted_gene_sets"]]
    nes = np.asarray(results["NES"], dtype=float)[:, :, columns]
    fdr = np.asarray(results["FDR"], dtype=float)[:, :, columns]
    figure, axes, vmax = matched.render(nes, fdr, results["plotted_labels"], constants)
    if vmax != results["shared_NES_scale"]["vmax"]:
        raise ValueError("Original shared NES scale changed")
    # Retain the left panel and its colorbar, with their original text/dot styling.
    axes[1].remove()
    figure.set_size_inches(6.6, 3.25)
    axes[0].set_position([.12, .20, .72, .64])
    figure.axes[-1].set_position([.88, .23, .023, .56])
    np.testing.assert_array_equal(axes[0].images[0].get_array(), nes[0])
    if len(axes[0].collections) != np.count_nonzero(fdr[0] < .05):
        raise ValueError("Significance dots changed")
    if axes[0].texts or figure.legends:
        raise ValueError("Extra annotations appeared")
    for extension in ("pdf", "svg", "png"):
        figure.savefig(OUTPUT / f"{STEM}.{extension}", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)


def plot(payload, output):
    original_savefig = Figure.savefig

    def savefig(figure, *args, **kwargs):
        if not getattr(figure, "_template_nes_numbers_added", False):
            axis = figure.axes[0]
            image = axis.images[0]
            values = np.asarray(image.get_array())
            if values.shape != (4, 6) or not np.isfinite(values).all():
                raise ValueError("Expected the saved four-cell-line/six-pathway NES heatmap")
            for (ci, si), value in np.ndenumerate(values):
                color = "white" if abs(value) > image.norm.vmax * .55 else "#1F2937"
                axis.text(si, ci, f"{value:.1f}", ha="center", va="center",
                          fontsize=8, color=color, zorder=3)
            if len(axis.texts) != 24:
                raise ValueError("Expected one NES label per cell")
            figure._template_nes_numbers_added = True
        return original_savefig(figure, *args, **kwargs)

    with patch.object(Figure, "savefig", savefig):
        _standalone(payload, output)
