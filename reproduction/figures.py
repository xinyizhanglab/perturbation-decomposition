"""Reference export, original plotting, and explicitly simplified summaries."""

import argparse
from pathlib import Path
import shutil

import numpy as np

from .catalog import (
    CELLS, FINAL_REFERENCE_PREFIXES, LIMITATIONS, REFERENCE_PREFIXES, RENDERABLE, figure_ids,
)
from .io import check_output_location, new_directory, read_json, record_file, verify, write_json


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures.style import apply_style
    apply_style()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.hashsalt": "perturbation-reproduction"})
    return plt


def _finish(axes):
    for ax in np.asarray(axes, dtype=object).ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="0.6", lw=.5)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)


def _points(ax, values, label, offset=0, errors=None, **kwargs):
    x = np.arange(len(values)) + offset
    y = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    marker = kwargs.pop("marker", "o")
    if errors is None:
        ax.plot(x, y, marker=marker, linestyle="none", label=label, **kwargs)
    else:
        error = np.asarray([0 if v is None else v for v in errors], dtype=float)
        ax.errorbar(x, y, yerr=error, fmt=marker, capsize=2, label=label, **kwargs)


def _cells(ax):
    ax.set_xticks(range(4), ["K562", "RPE1", "HepG2", "Jurkat"])


def _model_plot(data, plt):
    from figures.style import COLORS, MARKERS
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (model, entries) in enumerate(data.items()):
        _points(ax, [entries[c]["mean"] for c in CELLS], model, (i - (len(data)-1)/2)*.085,
                [entries[c].get("std") for c in CELLS],
                color=COLORS.get(model.split(" ")[0], "#555555"),
                marker=MARKERS.get(model.split(" ")[0], "o"),
                markerfacecolor="white" if "1→1" in model else COLORS.get(model, COLORS.get(model.split(" ")[0], "#555555")))
    _cells(ax)
    ax.set_ylabel("Perturbed-reference Pearson")
    ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    _finish([ax])
    return fig


def _calibration(data, plt):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    rows = [(model, target, entries) for model, targets in data.items()
            for target, entries in targets.items()]
    settings = sorted({setting for _, _, entries in rows for setting in entries})
    labels = [f"{model} / {target}" for model, target, _ in rows]
    for metric, ax in zip(("pref", "gamma"), axes):
        for i, setting in enumerate(settings):
            means, deviations = [], []
            for _, _, entries in rows:
                entry = entries.get(setting, {})
                means.append(entry.get(metric))
                deviations.append(entry.get(f"{metric}_std"))
            _points(ax, means, setting, (i-(len(settings)-1)/2)*.09, deviations)
        ax.set_ylabel("pref" if metric == "pref" else "Historical interaction attribution")
    axes[-1].set_xticks(range(len(labels)), labels, rotation=70, ha="right")
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Native calibration protocols; scGen target30 is not applicable")
    _finish(axes)
    return fig


def _components(data, plt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    groups = [(model, setting) for model, contexts in data.items()
              for setting in next(iter(contexts.values()))]
    for metric, ax in zip(("alpha", "beta", "gamma"), axes):
        for i, (model, setting) in enumerate(groups):
            entries = [data[model][c][setting] for c in CELLS]
            _points(ax, [e[metric] for e in entries], f"{model} / {setting}",
                    (i-(len(groups)-1)/2)*.045, [e[f"{metric}_std"] for e in entries])
        _cells(ax)
        ax.set_title(metric)
        ax.set_ylabel("Independent matching-component Pearson")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, fontsize=7)
    fig.suptitle("Balanced cross-target assemblies; five native-fit/cohort repeats, not CV")
    _finish(axes)
    return fig


def _render(data, kind, fields, plt):
    if kind == "gamma_matching":
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        models = list(data)
        for ax, cell in zip(axes.flat, CELLS):
            for index, model in enumerate(models):
                for setting, color in (("source_only", "#6F5CC4"), ("target30", "#D26A3A")):
                    entry = data[model][cell].get(setting)
                    if entry is not None:
                        ax.errorbar(entry["gamma"], index, xerr=entry["gamma_std"],
                                    fmt="o", color=color, capsize=2,
                                    label=setting if index == 0 else None)
            ax.set_yticks(range(len(models)), models)
            ax.set_title(cell)
            ax.set_xlabel("Matching-component Pearson")
            ax.axvline(0, color="0.6", lw=.5)
            ax.grid(axis="x", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(frameon=False)
        return fig
    if kind in ("models", "transfer"):
        return _model_plot(data["results"] if kind == "transfer" else data, plt)
    if kind == "calibration":
        return _calibration(data, plt)
    if kind == "components_source":
        selected = {model: {cell: {setting: entry for setting, entry in settings.items()
                                   if setting != "target30"}
                            for cell, settings in cells.items()}
                    for model, cells in data.items()}
        return _components(selected, plt)
    if kind == "components":
        return _components(data, plt)
    if kind == "fields":
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, field in enumerate(fields):
            _points(ax, [data[c][field] for c in CELLS], field, (i-.5)*.12)
        _cells(ax)
        ax.set_ylabel("Saved analysis value")
        ax.legend(frameon=False)
    elif kind == "heatmap":
        fig, ax = plt.subplots(figsize=(9, 4))
        if "analyses" in data:
            analysis = data["analyses"].index("cell_line_template")
            columns = [data["all_gene_sets"].index(name) for name in data["plotted_gene_sets"]]
            values = np.asarray(data["NES"], dtype=float)[analysis][:, columns]
            labels, contexts = data["plotted_labels"], data["contexts"]
            vmin, vmax = data["shared_NES_scale"]["vmin"], data["shared_NES_scale"]["vmax"]
        else:
            values, labels, contexts = data["nes"], data["sig_labels"], data["cls"]
            vmin, vmax = -3, 3
        image = ax.imshow(np.asarray(values, dtype=float), cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        ax.set_yticks(range(len(contexts)), contexts)
        fig.colorbar(image, ax=ax, label="Saved GSEA NES")
    elif kind == "control_space":
        fig, axes = plt.subplots(1, len(data), figsize=(9, 4))
        for ax, (rank, entries) in zip(np.atleast_1d(axes), data.items()):
            for i, name in enumerate(("template", "residual")):
                _points(ax, [entries[name][c] for c in CELLS], name, (i-.5)*.12)
            _cells(ax)
            ax.set_title(f"{rank} control PCs")
            ax.set_ylabel("Variance captured (%)")
            ax.legend(frameon=False)
        _finish(axes)
        return fig
    elif kind == "stacked":
        fig, ax = plt.subplots(figsize=(8, 3))
        for row, label in enumerate(("full_response", "template_removed")):
            left = 0
            for field in ("template_pct", "conserved_pct", "interaction_pct", "noise_pct"):
                value = data[label].get(field, 0)
                ax.barh(row, value, left=left, label=field if row == 0 else None)
                left += value
        ax.set_yticks([0, 1], ["Full response", "Projectively template-removed"])
        ax.set_xlabel("Saved variance fraction (%)")
        ax.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    elif kind == "ranks":
        fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
        ranks = data["ranks"]
        for ax, cell in zip(axes, CELLS):
            for model in ("ridge", "mlp"):
                values = [data[model][cell][str(r)] for r in ranks]
                ax.plot(ranks, [v["mean"] if isinstance(v, dict) else v for v in values],
                        "o-", label=model)
            ax.set_title(cell)
            ax.set_xscale("log")
            ax.set_xlabel("PCA rank")
        axes[0].set_ylabel("pref")
        axes[-1].legend(frameon=False)
        _finish(axes)
        return fig
    elif kind == "mechanism":
        fig, ax = plt.subplots(figsize=(9, 4))
        stages = list(data[CELLS[0]])
        for cell in CELLS:
            # Native field 'std' is standard Pearson, NOT a standard deviation.
            ax.plot(range(len(stages)), [data[cell][s]["pref"] for s in stages], "o-", label=cell)
        ax.set_xticks(range(len(stages)), [data[CELLS[0]][s]["label"] for s in stages],
                      rotation=25, ha="right")
        ax.set_ylabel("pref")
        ax.legend(frameon=False)
    elif kind == "landscape":
        fig, ax = plt.subplots(figsize=(7, 4))
        for cell in CELLS:
            x = data["target_properties"][cell]["erank_resid"]
            for field, marker in (("pref_seen", "o"), ("pref_double_unseen", "^")):
                ax.scatter(x, data[field][cell], marker=marker)
                ax.annotate(f"{cell} / {field}", (x, data[field][cell]), fontsize=7)
        ax.set_xlabel("Residual effective rank")
        ax.set_ylabel("pref")
    elif kind == "basis":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, field in zip(axes, ("mlp_matrix", "overlap_matrix")):
            image = ax.imshow(np.array(data[field], dtype=float), cmap="viridis")
            ax.set_xticks(range(4), data["cls"])
            ax.set_yticks(range(4), data["cls"])
            ax.set_title(field)
            fig.colorbar(image, ax=ax)
        return fig
    elif kind == "attribution":
        fig, axes = plt.subplots(1, len(fields), figsize=(12, 4))
        for ax, field in zip(np.atleast_1d(axes), fields):
            for i, (model, cells) in enumerate(data.items()):
                values = [np.asarray(cells[c][field], dtype=float) for c in CELLS]
                _points(ax, [v.mean() for v in values], model, (i-(len(data)-1)/2)*.09,
                        [v.std(ddof=0) if len(v) > 1 else 0 for v in values])
            _cells(ax)
            ax.set_title(f"Historical {field} attribution—not independent matching")
        axes[-1].legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
        _finish(axes)
        return fig
    elif kind == "gamma_reveal":
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, metric in zip(axes, ("beta", "gamma")):
            for i, setting in enumerate(("zeroshot", "target30")):
                _points(ax, [data[c][setting][metric] for c in CELLS], setting, (i-.5)*.12)
            _cells(ax)
            ax.set_title(f"Historical {metric} attribution")
            ax.legend(frameon=False)
        _finish(axes)
        return fig
    elif kind == "enrichment":
        fig, axes = plt.subplots(1, 4, figsize=(19, 5))
        for ax, cell in zip(axes, CELLS):
            rows = data[cell]
            if rows:
                ax.barh(range(len(rows)), [r["score"] for r in rows],
                        color=["#3572B0" if r["direction"] == "depmap" else "#D26A3A"
                               for r in rows])
                ax.set_yticks(range(len(rows)), [r["name"] for r in rows], fontsize=7)
            else:
                ax.text(.5, .5, "No significant enrichments", transform=ax.transAxes, ha="center")
            ax.axvline(0, color="0.6", lw=.5)
            ax.set_title(cell)
            ax.set_xlabel("Signed log2 fold enrichment")
        return fig
    else:
        raise ValueError(f"Unknown renderer: {kind}")
    _finish([ax])
    return fig


def render(root, destination, ids=None, mode="reference", font_dir=None):
    if mode == "original":
        from figures.original import render as render_original
        return render_original(root, destination, ids, font_dir)
    if font_dir is not None:
        raise ValueError("--font_dir applies only to --mode original.")
    if mode == "rerender":
        import warnings
        warnings.warn("--mode rerender is a legacy alias for summary, not original figure plotting. "
                      "Use reference for unchanged assets or original for original drawing code.",
                      FutureWarning, stacklevel=2)
        mode = "summary"
    root, destination = Path(root), Path(destination)
    manifest = verify(root)
    check_output_location(root, destination)
    if mode not in ("summary", "reference"):
        raise ValueError("Select reference, original, or summary mode")
    references = set(REFERENCE_PREFIXES) | set(FINAL_REFERENCE_PREFIXES)
    selected = list(ids) if ids else sorted(RENDERABLE if mode == "summary" else references)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate figure entrypoints are not permitted")
    unknown = set(selected) - set(figure_ids())
    if unknown:
        raise ValueError(f"Unknown figures: {sorted(unknown)}")
    for identifier in selected:
        if mode == "summary" and identifier not in RENDERABLE:
            raise ValueError(f"{identifier}: no summary renderer; use reference mode or consult coverage")
        if mode == "reference" and identifier not in references:
            raise ValueError(f"{identifier}: no exact reference asset available")
    output = new_directory(destination)
    provenance = {"release_version": manifest["version"], "mode": mode, "figures": {},
                  "note": "Reference mode copies existing assets without alteration. "
                          "Summary mode uses simplified diagnostic layouts, not the original figure design."}
    for identifier in selected:
        directory = new_directory(output / identifier)
        if mode == "summary":
            relative, kind, fields = RENDERABLE[identifier]
            data = read_json(root / relative)
            plt = _pyplot()
            figure = _render(data, kind, fields, plt)
            if manifest.get("example"):
                figure.suptitle("Synthetic example — not manuscript results")
            figure.tight_layout()
            for suffix in ("pdf", "svg", "png"):
                figure.savefig(directory / f"{identifier}.{suffix}", bbox_inches="tight", dpi=150,
                               metadata={"Creator": "perturbation-decomposition reproduction"})
            plt.close(figure)
            write_json(directory / "plot_data.json", data)
            provenance["figures"][identifier] = {"input": relative,
                "input_sha256": next(e["sha256"] for e in manifest["files"] if e["path"] == relative),
                "renderer": kind}
        else:
            final = FINAL_REFERENCE_PREFIXES
            prefixes = (final[identifier],) if identifier in final else tuple(
                "paper/" + p for p in REFERENCE_PREFIXES[identifier])
            entries = [e for e in manifest["files"] if e["path"].endswith((".svg", ".pdf"))
                       and any(e["path"].startswith(p) for p in prefixes)]
            if not entries:
                raise FileNotFoundError(f"No reference assets for {identifier}")
            for entry in entries:
                shutil.copyfile(root / entry["path"], directory / Path(entry["path"]).name)
            provenance["figures"][identifier] = {"copied_reference_files": entries}
    provenance["outputs"] = [record_file(output, p) for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "provenance.json", provenance)
    return output


def coverage():
    from figures.original import catalog as original_catalog
    from figures.rendering import PROFILE_PATH
    original = original_catalog()
    profiles = read_json(PROFILE_PATH)["figures"]
    return [{"id": identifier, "rerender": identifier in RENDERABLE,
             "summary": identifier in RENDERABLE,
             "original": identifier in original and profiles[identifier]["compatibility"] == "matched",
             "rendering_profile": profiles[identifier]["status"],
             "rendering_compatibility": profiles[identifier]["compatibility"],
             "reference": identifier in REFERENCE_PREFIXES or identifier in FINAL_REFERENCE_PREFIXES,
             "limitation": LIMITATIONS.get(identifier)}
            for identifier in figure_ids()]


def legacy_main(identifier):
    parser = argparse.ArgumentParser(description=f"Portable figure entrypoint: {identifier}")
    parser.add_argument("--root", required=True, type=Path, help="Verified artifact release")
    parser.add_argument("--out", required=True, type=Path, help="New output directory")
    parser.add_argument("--mode", choices=("reference", "original", "summary", "rerender"),
                        default="reference")
    parser.add_argument("--font_dir", type=Path)
    args = parser.parse_args()
    render(args.root, args.out, [identifier], args.mode, args.font_dir)
