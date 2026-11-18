"""Run preserved plotting routines on saved results, without fitting models."""

import argparse
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from reproduction.catalog import CELLS
from reproduction.io import (
    check_output_location, new_directory, read_json, record_file, safe_path,
    sha256, verify, write_json,
)


SOURCES = Path(__file__).with_name("_original")


def catalog():
    return read_json(SOURCES / "manifest.json")["figures"]


def available(paths):
    from figures.rendering import profile
    paths = set(paths)
    selected = []
    for name, entry in sorted(catalog().items()):
        rendering = profile(name, require_verified=False)
        if ((entry["input"] is None or entry["input"] in paths)
                and rendering["compatibility"] == "matched"
                and set(rendering["assets"]).issubset(paths)):
            selected.append(name)
    return selected


def _source_records(identifier):
    manifest = read_json(SOURCES / "manifest.json")
    entry = manifest["figures"][identifier]
    records = {}
    for name in [entry["source"], *entry["dependencies"]]:
        path = safe_path(SOURCES, name)
        record = manifest["sources"][name]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Original plotting source changed: {name}")
        records[name] = record
    return records


def _prepare(identifier, data):
    """Restore plotting variable names and JSON-erased scalar, array and key types."""
    from figures.saved_supplementary import SUPPORTED as SAVED, prepare as saved_prepare
    if identifier in SAVED:
        return saved_prepare(identifier, data)
    from figures.supplementary import SUPPORTED, prepare
    if identifier in SUPPORTED:
        return prepare(identifier, data)
    if identifier == "fig1_b":
        return {"template_var": {
            cell: np.dtype(data["source"]["inputs"][cell]["dtype"]).type(value)
            for cell, value in data["template_var"].items()
        }}
    if identifier == "fig1_dsup":
        cos2 = {cell: data[cell]["cos2"] for cell in CELLS}
        return {"cos2": cos2, "sorted_cls": sorted(CELLS, key=cos2.get, reverse=True)}
    if identifier == "fig1_d" and "analyses" in data:
        return {}
    if identifier == "fig1_d":
        return {
            "n_cols": len(data["signatures"]), "col_order": data["signatures"],
            "col_labels": [
                {"HALLMARK_G2M_CHECKPOINT": "G2M\ncheckpoint",
                 "HALLMARK_E2F_TARGETS": "E2F\ntargets",
                 "HALLMARK_P53_PATHWAY": "p53\npathway",
                 "HALLMARK_APOPTOSIS": "Apoptosis",
                 "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": "UPR",
                 "HALLMARK_OXIDATIVE_PHOSPHORYLATION": "OxPhos",
                 "LINEAGE_EPITHELIAL": "Epithelial\nidentity",
                 "LINEAGE_HEPATOCYTE": "Hepatocyte\nidentity",
                 "LINEAGE_TCELL": "T cell\nidentity"}[name]
                for name in data["signatures"]],
            "hallmark_valid": [name for name in data["signatures"] if name.startswith("HALLMARK_")],
            **{target: np.asarray(data[key], dtype=float) for target, key in (
                ("nes_matrix", "nes"), ("fdr_matrix", "fdr"),
                ("overlap_hvg_matrix", "overlap_hvg"), ("overlap_full_matrix", "overlap_full"))},
        }
    if identifier == "fig1_e":
        return {"results": {int(k): v for k, v in data.items()}}
    if identifier == "fig1_f":
        full, residual = data["full_response"], data["template_removed"]
        return {
            **{variable: full[key] for variable, key in (
                ("template_pct", "template_pct"), ("beta_pct", "conserved_pct"),
                ("gamma_pct", "interaction_pct"), ("noise_pct", "noise_pct"))},
            **{variable: residual[key] for variable, key in (
                ("beta_resid_pct", "conserved_pct"), ("gamma_resid_pct", "interaction_pct"),
                ("noise_resid_pct", "noise_pct"))},
        }
    if identifier == "fig1_suppl_decomp":
        labels = ("ANOVA variance\naccounting", "Centroid\nremoval", "Projective\nremoval")
        return {"methods": {label: data["methods"][label.replace("\n", " ")] for label in labels},
                "geom": data["geometry"]}
    if identifier in ("fig2_a", "fig3_c"):
        return {"results": {model: {cell: entry["values"] for cell, entry in cells.items()}
                            for model, cells in data.items()}}
    if identifier in ("fig2_d", "fig2_d2"):
        if not {"ranks", "ridge", "mlp", "morph_5foldcv"} <= data.keys():
            raise ValueError("The current paper Figure 2d needs its multi-model saved table, not older Ridge-only results.")
        def keys(entries):
            return {cell: {int(k) if k.isdigit() else k: v for k, v in values.items()}
                    for cell, values in entries.items()}
        return {"RANKS": data["ranks"], "ridge_results": keys(data["ridge"]),
                "mlp_results": keys(data["mlp"]), "morph_pref": data["morph_5foldcv"]}
    if identifier == "fig2_e":
        return {"all_results": data}
    if identifier == "fig2_supp2_2":
        return {"feature_results": data["feature_ablation"],
                "pls_results": {int(k): v for k, v in data["pls"].items()},
                "ridge_pca50": data["ridge_pca50"]}
    if identifier == "fig3_d":
        return {"target_props": data["target_properties"], "pref_seen": data["pref_seen"],
                "pref_du": data["pref_double_unseen"]}
    if identifier == "fig3_e":
        return {"n_cl": len(data["cls"]),
                "mlp_matrix": np.asarray(data["mlp_matrix"], dtype=float),
                "overlap_matrix": np.asarray(data["overlap_matrix"], dtype=float),
                "rho": data["spearman_rho"], "p_val": data["spearman_p"]}
    if identifier == "fig3_b":
        return {"results": {model: {target: entry["values"] for target, entry in targets.items()}
                            for model, targets in data["results"].items()}}
    if identifier in ("fig_tahoe_b_saturation", "fig_tahoe_c_controls"):
        return {"battery": data}
    return {"results": data}


def _font(family, font_dir=None):
    from matplotlib import font_manager
    if font_dir is not None:
        files = sorted(p for p in Path(font_dir).rglob("*")
                       if p.is_file() and p.suffix.lower() in (".ttf", ".otf", ".ttc"))
        if not files:
            raise ValueError(f"No font files found in {font_dir}")
        for path in files:
            font_manager.fontManager.addfont(str(path))
    try:
        path = Path(font_manager.findfont(
            font_manager.FontProperties(family=family, style="normal", weight="normal", stretch="normal"),
            fallback_to_default=False))
    except ValueError as exc:
        raise ValueError(
            f"Saved figure font {family!r} is unavailable. Install that font or "
            "provide --font_dir containing licensed font files; otherwise use --mode reference. "
            "No replacement font was selected."
        ) from exc
    return {"requested": family, "file": path.name, "sha256": sha256(path)}


def _worker(root, destination, identifier, font_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import __version__ as matplotlib_version
    from figures.rendering import (
        PROFILE_PATH, compare_outputs, profile, saved_font_families, saved_settings,
    )
    plt.rcdefaults()
    recipe = catalog()[identifier]
    sources = _source_records(identifier)
    rendering = profile(identifier, root)
    check_output_location(root, destination)
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("The plotting worker requires a new, empty output directory.")
    manifest = verify(root)
    paths = {record["path"] for record in manifest["files"]}
    relative = recipe["input"]
    if relative is not None and relative not in paths:
        raise ValueError(f"{identifier}: bundle does not declare required input {relative}")
    packaged = recipe.get("packaged_input")
    input_path = safe_path(SOURCES, packaged) if packaged else safe_path(root, relative) if relative else None
    from figures.saved_supplementary import load_plot_data
    data = load_plot_data(input_path) if input_path else {}
    namespace = {"__name__": "preserved_plot"}
    namespace.update(_prepare(identifier, data))
    if relative:
        path = safe_path(root, relative)
        namespace["DATA"] = path.parent if identifier == "fig_pathway_crosscl" else path
    family = rendering.get("font_check_family") or rendering["rc_overrides"]["font.family"][0]
    font = _font(family, font_dir)
    source = safe_path(SOURCES, recipe["source"])
    with tempfile.TemporaryDirectory(prefix="perturbation-drawing-") as temporary:
        drawing = Path(temporary)
        namespace["OUT"] = drawing
        # Bundles supply data, never executable plotting code.
        with saved_font_families(rendering.get("isolate_font_families")), \
                saved_settings(rendering["rc_overrides"]):
            namespace = runpy.run_path(str(source), init_globals=namespace, run_name="preserved_plot")
            if recipe["call"]:
                function = namespace[recipe["call"]]
                function() if recipe["call"] == "main" else function(data, drawing)
        plt.close("all")
        comparisons = compare_outputs(drawing, rendering)
        if not all(row["matched"] for row in comparisons):
            raise RuntimeError(f"{identifier}: regenerated drawing differs from its saved rendering "
                               "profile. No figure was copied to the output. Use --mode reference.")
        outputs = [p for p in sorted(drawing.iterdir()) if p.suffix in (".pdf", ".svg", ".png")]
        if not outputs:
            raise RuntimeError(f"{identifier}: original routine produced no figure files")
        for path in outputs:
            with path.open("rb") as source_file, (destination / path.name).open("xb") as output_file:
                shutil.copyfileobj(source_file, output_file)
    outputs = [p for p in sorted(destination.iterdir()) if p.suffix in (".pdf", ".svg", ".png")]
    write_json(destination / "plot_data.json", data if input_path else {
        "source": "Literal values preserved in the original plotting statements",
        "source_sha256": sources[recipe["source"]]["sha256"],
    })
    write_json(destination / "provenance.json", {
        "figure": identifier, "mode": "original",
        "input": f"package:figures._original/{packaged}" if packaged else relative,
        "input_sha256": sha256(input_path) if input_path else None,
        "plotting_sources": sources, "font": font, "matplotlib": matplotlib_version,
        "numpy": np.__version__, "analysis_recomputed": False,
        "runtime_defaults": "matplotlib.rcdefaults() before the per-figure saved settings",
        "rendering_profile_sha256": sha256(PROFILE_PATH),
        "saved_rc_settings": rendering["rc_overrides"], "rendering_comparisons": comparisons,
        "outputs": [record_file(destination, path) for path in outputs],
        "appearance": "Drawing matched the saved SVG profile; export metadata is not byte-identical. "
                      "Reference mode preserves the exact original files.",
    })


def render(root, destination, ids=None, font_dir=None):
    from figures.rendering import profile
    root, destination = Path(root).resolve(), Path(destination).resolve()
    manifest = verify(root)
    if manifest.get("example"):
        raise ValueError("Original manuscript routines require real saved results, not the synthetic example.")
    check_output_location(root, destination)
    paths = {record["path"] for record in manifest["files"]}
    recipes = catalog()
    selected = list(ids) if ids is not None else available(paths)
    profiles = {}
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one original figure, without duplicate entrypoints.")
    for identifier in selected:
        if identifier not in recipes:
            raise ValueError(f"{identifier}: original numerical plotting is not packaged; use --mode reference.")
        required = recipes[identifier]["input"]
        if required is not None and required not in paths:
            raise ValueError(f"{identifier}: missing declared saved input {required}. Install an updated figure bundle.")
        _source_records(identifier)
        profiles[identifier] = profile(identifier, root)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    for identifier in selected:
        family = profiles[identifier].get("font_check_family") or profiles[identifier]["rc_overrides"]["font.family"][0]
        _font(family, font_dir)
    output = new_directory(destination)
    records = {}
    for identifier in selected:
        directory = new_directory(output / identifier)
        with tempfile.TemporaryDirectory(prefix="perturbation-plot-cache-") as cache:
            command = [sys.executable, "-B", "-m", "figures.original", "--worker",
                       "--results_dir", str(root), "--out_dir", str(directory), "--figure", identifier]
            if font_dir is not None:
                command.extend(["--font_dir", str(Path(font_dir).resolve())])
            env = {**os.environ, "MPLCONFIGDIR": cache, "PYTHONDONTWRITEBYTECODE": "1",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            env.pop("MATPLOTLIBRC", None)
            completed = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
            if completed.returncode:
                raise RuntimeError(f"{identifier}: original plotting failed:\n{completed.stderr[-6000:]}")
        records[identifier] = read_json(directory / "provenance.json")
    write_json(output / "provenance.json", {
        "release_version": manifest["version"], "mode": "original",
        "analysis_recomputed": False, "figures": records,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--figure", required=True)
    parser.add_argument("--font_dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            _worker(args.results_dir, args.out_dir, args.figure, args.font_dir)
        else:
            render(args.results_dir, args.out_dir, [args.figure], args.font_dir)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
