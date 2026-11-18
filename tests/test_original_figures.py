"""Portability and file-preservation checks, not scientific reanalysis."""

import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from data.figure_inputs import TABLES, prepare
from figures.original import SOURCES, _font, _prepare, _source_records, _worker, available, catalog, render
from figures.rendering import build_profiles, profile, saved_settings
from reproduction.catalog import CELLS
from reproduction.io import record_file, verify, write_json


ROOT = Path(__file__).resolve().parents[1]


def bundle(root, files):
    root.mkdir()
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode())
    write_json(root / "manifest.json", {
        "schema_version": 1, "version": "test", "profile": "figures",
        "files": [record_file(root, path) for path in sorted(root.rglob("*")) if path.is_file()],
    })
    return root


def native_fixture(base):
    import matplotlib
    styles = ("Ridge 1→1", "Ridge 3→1", "MLP 1→1", "MLP 3→1",
              "MORPH", "STATE", "BioLORD", "scGen")
    data = {"font": "Arial", "results": {
        model: {cell: {"mean": .2, "std": .01} for cell in CELLS} for model in styles}}
    source = bundle(base / "source", {"final/figure3/results.json": data})
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        with saved_settings({"font.family": ["DejaVu Sans"], "font.sans-serif": ["DejaVu Sans"]}):
            namespace = runpy.run_path(str(SOURCES / "fig3_crosscl_transfer.py"))
            namespace["plot"](data, source / "final/figure3")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"] = [record_file(source, path) for path in sorted(source.rglob("*"))
                         if path.is_file() and path.name != "manifest.json"]
    (source / "manifest.json").write_text(json.dumps(manifest))
    profiles = build_profiles(source, catalog())
    for identifier in ("fig3_a", "fig3_crosscl_transfer"):
        profiles["figures"][identifier]["compatibility"] = "matched"
    registry = base / "rendering_profiles.json"
    write_json(registry, profiles)
    return source, registry


class OriginalFigureTests(unittest.TestCase):
    def test_shipped_sources_are_plotting_only_and_hash_checked(self):
        self.assertEqual(len(catalog()), 43)
        for identifier in catalog():
            _source_records(identifier)
        for path in SOURCES.glob("*.py"):
            source = path.read_text()
            self.assertNotIn("/mnt/", source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = ([n.name for n in node.names] if isinstance(node, ast.Import)
                               else [node.module or ""])
                    self.assertFalse(any(m.split(".")[0] in
                                         {"torch", "sklearn", "anndata", "scipy", "pickle"}
                                         for m in modules), str(path))
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, {"fit", "fit_transform", "read_h5ad"}, str(path))

    def test_available_original_requires_declared_input(self):
        self.assertNotIn("fig_pathway_crosscl", available([]))
        entry = profile("fig_pathway_crosscl")
        self.assertIn("fig_pathway_crosscl", available([entry["input"], *entry["assets"]]))
        self.assertNotIn("fig3_morph_double_unseen_combo", available([]))

    def test_bundle_export_preserves_existing_figure_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = bundle(base / "source", {"paper/figure_pathway/existing.pdf": b"unchanged-vector"})
            benchmark = base / "benchmark"
            for relative in TABLES:
                path = benchmark / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"saved": 1}\n')
            before = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source.rglob("*") if p.is_file()}
            output = prepare(source, benchmark, base / "new-bundle")
            self.assertEqual((output / "paper/figure_pathway/existing.pdf").read_bytes(), b"unchanged-vector")
            self.assertEqual(before, {
                str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob("*") if p.is_file()})
            self.assertEqual(len(verify(output)["files"]), 4)
            with self.assertRaises(FileExistsError):
                prepare(source, benchmark, output)

    def test_bundle_export_names_missing_table_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = bundle(base / "source", {"plot.pdf": b"existing"})
            with self.assertRaisesRegex(FileNotFoundError, "crosscl_pathway_results"):
                prepare(source, base / "benchmark", base / "out")
            self.assertFalse((base / "out").exists())

    def test_default_cli_copies_reference_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = bundle(base / "source", {
                "paper/figure_pathway/crosscl_pathway_absolute_main.pdf": b"exact-paper-bytes"})
            result = subprocess.run([
                sys.executable, "-B", "-m", "figures.fig_pathway_crosscl",
                "--results_dir", str(source), "--out_dir", str(base / "out")],
                cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            copied = base / "out/fig_pathway_crosscl/crosscl_pathway_absolute_main.pdf"
            self.assertEqual(copied.read_bytes(), b"exact-paper-bytes")
            self.assertEqual(json.loads((base / "out/provenance.json").read_text())["mode"], "reference")

    def test_missing_original_input_does_not_fall_back_to_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = bundle(base / "source", {"other.json": {}})
            with self.assertRaisesRegex(ValueError, "missing declared saved input"):
                render(source, base / "out", ["fig_pathway_crosscl"])
            self.assertFalse((base / "out").exists())

    def test_rank_entrypoint_explicitly_uses_the_archived_current_paper_variant(self):
        self.assertEqual(catalog()["fig2_d"]["alias_of"], "fig2_d2")
        self.assertEqual(catalog()["fig2_d"]["source"], catalog()["fig2_d2"]["source"])
        self.assertEqual(profile("fig2_d")["assets"], profile("fig2_d2")["assets"])
        with self.assertRaisesRegex(ValueError, "not older Ridge-only"):
            _prepare("fig2_d", {"k562": {"1": .1, "full": .2}})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = bundle(base / "source", {"other.json": {}})
            with self.assertRaisesRegex(ValueError, "missing declared saved input"):
                render(source, base / "out", ["fig2_d"])
            self.assertFalse((base / "out").exists())

    def test_missing_font_is_actionable_not_a_substitution(self):
        with patch("matplotlib.font_manager.findfont", side_effect=ValueError("missing")):
            with self.assertRaisesRegex(ValueError, "No replacement font"):
                _font("Missing original font")

    def test_original_plot_source_keeps_original_sizes_and_labels(self):
        source = (SOURCES / "fig_pathway_crosscl.py").read_text()
        self.assertIn('figsize=(5.5, fig_h)', source)
        self.assertIn('figsize=(5, fig_h)', source)
        self.assertIn('N_SHOW = 28', source)
        self.assertIn('"font.family": "Arial"', source)
        self.assertNotIn("apply_style", source)

    def test_original_native_plot_needs_only_saved_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, registry = native_fixture(base)
            result = base / "out"
            result.mkdir()
            with patch("figures.rendering.PROFILE_PATH", registry):
                _worker(source, result, "fig3_crosscl_transfer")
            stem = result / "fig3_crosscl_biolord_cellline_go_author_v1"
            for suffix in (".pdf", ".svg", ".png"):
                self.assertTrue(stem.with_suffix(suffix).is_file())
            provenance = json.loads((result / "provenance.json").read_text())
            self.assertFalse(provenance["analysis_recomputed"])
            self.assertEqual(provenance["font"]["requested"], "DejaVu Sans")
            self.assertTrue(all(row["matched"] for row in provenance["rendering_comparisons"]))
            with patch("figures.rendering.PROFILE_PATH", registry):
                with self.assertRaisesRegex(ValueError, "empty output directory"):
                    _worker(source, result, "fig3_crosscl_transfer")

    def test_mismatched_drawing_is_not_copied_to_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, registry = native_fixture(base)
            output = base / "out"
            output.mkdir()
            with patch("figures.rendering.PROFILE_PATH", registry), patch(
                    "figures.rendering.compare_outputs", return_value=[{"matched": False}]):
                with self.assertRaisesRegex(RuntimeError, "No figure was copied"):
                    _worker(source, output, "fig3_crosscl_transfer")
            self.assertEqual(list(output.iterdir()), [])

    def test_pathway_uses_saved_font_not_the_source_request(self):
        self.assertEqual(catalog()["fig_pathway_crosscl"]["source_requested_font"], "Arial")
        self.assertEqual(profile("fig_pathway_crosscl")["rc_overrides"]["font.family"], ["DejaVu Sans"])


    def test_current_figure3a_alias_keeps_all_eight_series(self):
        recipe = catalog()["fig3_a"]
        canonical = catalog()["fig3_crosscl_transfer"]
        self.assertEqual(recipe["alias_of"], "fig3_crosscl_transfer")
        for name, value in canonical.items():
            self.assertEqual(recipe[name], value)
        self.assertEqual(recipe["input"], "final/figure3/results.json")
        self.assertEqual(profile("fig3_a")["alias_of"], "fig3_crosscl_transfer")
        for name, value in profile("fig3_crosscl_transfer").items():
            self.assertEqual(profile("fig3_a")[name], value)
        namespace = runpy.run_path(str(SOURCES / recipe["source"]))
        self.assertEqual(tuple(namespace["MODEL_STYLE"]), (
            "Ridge 1→1", "Ridge 3→1", "MLP 1→1", "MLP 3→1",
            "MORPH", "STATE", "BioLORD", "scGen",
        ))

    def test_current_figure3a_reference_cli_selects_final_not_historical_panels(self):
        from figures.render import available as available_figures
        from figures.rendering import reference_entries
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = bundle(base / "source", {
                "final/figure3/current.svg": b"current-full-comparison-svg",
                "final/figure3/current.pdf": b"current-full-comparison-pdf",
                "paper/figure3/fig3a_1to1_transfer.svg": b"old-one-source",
                "paper/figure3/fig3b_3to1_transfer.svg": b"old-three-source",
            })
            before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
            self.assertIn("fig3_a", available_figures(source, "reference"))
            self.assertEqual(
                {entry["path"] for entry in reference_entries("fig3_a", verify(source))},
                {"final/figure3/current.svg", "final/figure3/current.pdf"},
            )
            result = subprocess.run([
                sys.executable, "-B", "-m", "figures.fig3_a",
                "--results_dir", str(source), "--out_dir", str(base / "out")],
                cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = base / "out/fig3_a"
            self.assertEqual({p.name for p in output.iterdir()}, {"current.pdf", "current.svg"})
            self.assertEqual((output / "current.svg").read_bytes(), b"current-full-comparison-svg")
            self.assertEqual((output / "current.pdf").read_bytes(), b"current-full-comparison-pdf")
            self.assertEqual(before, {p: p.read_bytes() for p in source.rglob("*") if p.is_file()})

    def test_current_figure3a_does_not_fall_back_to_old_panel(self):
        from figures.render import available as available_figures
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = bundle(base / "source", {
                "paper/figure3/fig3a_1to1_transfer.svg": b"old-one-source",
                "paper/figure3/fig3b_3to1_transfer.svg": b"old-three-source",
            })
            self.assertNotIn("fig3_a", available_figures(source, "reference"))
            self.assertNotIn("fig3_a", available_figures(source, "original"))
            result = subprocess.run([
                sys.executable, "-B", "-m", "figures.fig3_a",
                "--results_dir", str(source), "--out_dir", str(base / "out")],
                cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unavailable in reference mode", result.stderr)
            self.assertFalse((base / "out").exists())
            with self.assertRaisesRegex(ValueError, "missing declared saved input"):
                render(source, base / "original", ["fig3_a"])
            self.assertFalse((base / "original").exists())

    def test_current_figure3a_redraws_all_series_from_saved_table(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, registry = native_fixture(base)
            output = base / "out"
            output.mkdir()
            with patch("figures.rendering.PROFILE_PATH", registry):
                _worker(source, output, "fig3_a")
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(provenance["figure"], "fig3_a")
            self.assertEqual(provenance["input"], "final/figure3/results.json")
            self.assertFalse(provenance["analysis_recomputed"])
            self.assertTrue(all(row["matched"] for row in provenance["rendering_comparisons"]))
            data = json.loads((output / "plot_data.json").read_text())
            self.assertEqual(len(data["results"]), 8)
            self.assertIn("BioLORD", data["results"])
            self.assertIn("scGen", data["results"])


if __name__ == "__main__":
    unittest.main()
