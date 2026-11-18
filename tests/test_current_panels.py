"""Current paper variants use frozen tables and preserved drawing statements."""

import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data.current_figures import plan, prepare
from figures.original import SOURCES, _prepare, catalog
from figures.rendering import profile
from reproduction.catalog import CELLS, FINAL_REFERENCE_PREFIXES, RENDERABLE
from reproduction.download import download, pack
from reproduction.io import record_file, sha256, verify, write_json


def fixture_bundle(root, files):
    root.mkdir()
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    write_json(root / "manifest.json", {
        "schema_version": 1, "version": "test", "profile": "figures",
        "files": [record_file(root, path) for path in root.rglob("*") if path.is_file()],
    })
    return root


class CurrentPanelTests(unittest.TestCase):
    def test_current_panels_bind_the_new_sources_and_inputs(self):
        for name, source, prefix in (
            ("fig1_d", "fig1_d_current.py", "final/hallmark/"),
            ("fig4_c", "fig4_d_current.py", "final/figure4d/"),
            ("fig4_d", "fig4_d_current.py", "final/figure4d/"),
        ):
            with self.subTest(name=name):
                self.assertEqual(catalog()[name]["source"], source)
                self.assertEqual(catalog()[name]["call"], "plot")
                self.assertEqual(FINAL_REFERENCE_PREFIXES[name], prefix)
                self.assertTrue(all(path.startswith(prefix) for path in profile(name)["assets"]))
                self.assertEqual(profile(name)["compatibility"], "matched")
        self.assertEqual(RENDERABLE["fig1_d"][0], "final/hallmark/results.json")
        self.assertEqual(RENDERABLE["fig4_d"][:2], ("final/components/results.json", "gamma_matching"))
        self.assertEqual(catalog()["fig4_d"]["alias_of"], "fig4_c")
        self.assertEqual(profile("fig4_d")["assets"], profile("fig4_c")["assets"])
        self.assertNotIn("alias_of", catalog()["fig4_calibration"])
        self.assertEqual(catalog()["fig4_calibration"]["input"], "final/figure4c/results.json")

    def test_new_hallmark_payload_does_not_use_the_old_adapter(self):
        data = {"analyses": ["cell_line_template", "alpha_only"]}
        self.assertEqual(_prepare("fig1_d", data), {})
        self.assertEqual(data, {"analyses": ["cell_line_template", "alpha_only"]})

    def test_hallmark_drawing_keeps_nes_labels_fdr_dots_and_shared_scale(self):
        from matplotlib.figure import Figure
        import matplotlib
        values = np.arange(48, dtype=float).reshape(2, 4, 6) / 10 - 2
        vmax = float(np.max(np.abs(values)) * 1.05)
        data = {
            "all_gene_sets": list("abcdef"), "plotted_gene_sets": list("abcdef"),
            "plotted_labels": list("abcdef"), "NES": values.tolist(),
            "FDR": np.full((2, 4, 6), .01).tolist(), "shared_NES_scale": {"vmax": vmax},
        }
        before = copy.deepcopy(data)
        seen = []

        def capture(figure, *args, **kwargs):
            axis = figure.axes[0]
            seen.append((len(figure.axes), len(axis.texts), len(axis.collections),
                         axis.images[0].norm.vmax, axis.get_title()))

        with matplotlib.rc_context(), patch.object(Figure, "savefig", capture):
            namespace = runpy.run_path(str(SOURCES / "fig1_d_current.py"))
            namespace["plot"](data, Path("/not-written"))
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(record == (2, 24, 24, vmax, "Cell-line template") for record in seen))
        self.assertEqual(data, before)

    def test_clean_calibration_does_not_invent_scgen_target30(self):
        from matplotlib.figure import Figure
        import matplotlib
        metric = {"pref": .2, "gamma": .1}
        missing = {"status": "not_applicable", "pref": None, "gamma": None}
        data = {
            "state": {cell: {"zeroshot": metric.copy(), "calibrated": metric.copy()} for cell in CELLS},
            "morph": {pair: {setting: metric.copy() for setting in
                            ("Source only", "Fine-tune ctrl", "+30% target")} for pair in
                      ("k562→rpe1", "rpe1→k562", "hepg2→jurkat", "jurkat→hepg2")},
            "biolord": {cell: {"source_only": metric.copy(), "target30": metric.copy()} for cell in CELLS},
            "scgen": {cell: {"source_only": metric.copy(), "target30": missing.copy()} for cell in CELLS},
        }
        before, seen = copy.deepcopy(data), []

        def capture(figure, *args, **kwargs):
            seen.append([[text.get_text() for text in axis.texts] for axis in figure.axes])

        with matplotlib.rc_context(), patch.object(Figure, "savefig", capture):
            namespace = runpy.run_path(str(SOURCES / "fig4_c_current.py"))
            namespace["plot"](data, Path("/not-written"))
        self.assertEqual(len(seen), 3)
        self.assertTrue(all("+30%: N/A" not in texts for axes in seen for texts in axes))
        self.assertEqual(data, before)

    def test_interaction_drawing_keeps_the_recorded_layout_fonts_and_grid(self):
        from matplotlib.figure import Figure
        import matplotlib
        models = ("ridge_1to1", "ridge_3to1", "mlp_1to1", "mlp_3to1",
                  "morph", "state", "biolord", "scgen")
        data = {
            model: {cell: {setting: {"gamma": .1, "gamma_std": .01}
                           for setting in (("source_only",) if model == "scgen"
                                           else ("source_only", "target30"))}
                    for cell in CELLS}
            for model in models
        }
        before, seen = copy.deepcopy(data), []

        def capture(figure, *args, **kwargs):
            seen.append({
                "titles": [axis.get_title() for axis in figure.axes],
                "size": figure.get_size_inches().tolist(),
                "grid": {line.get_linewidth() for axis in figure.axes for line in axis.get_xgridlines()},
                "labels": [text.get_text() for text in figure.axes[0].get_yticklabels()],
            })

        with matplotlib.rc_context(), patch.object(Figure, "savefig", capture):
            namespace = runpy.run_path(str(SOURCES / "fig4_d_current.py"))
            namespace["plot"](data, Path("/not-written"))
        self.assertEqual(len(seen), 3)
        for record in seen:
            self.assertEqual(record["titles"], ["K562", "RPE1", "HepG2", "Jurkat"])
            self.assertEqual(record["size"], [10.4, 8.])
            self.assertEqual(record["grid"], {1.5})
            self.assertEqual(len(record["labels"]), 8)
        self.assertEqual(data, before)

    def test_current_figure_export_is_read_only_and_installable(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = fixture_bundle(base / "source", {
                "final/components/results.json": b'{"saved": true}',
                "paper/old.svg": b"original-vector",
            })
            research = base / "research"
            research.mkdir()
            (research / "new.svg").write_bytes(b"new-vector")
            spec = base / "current.json"
            write_json(spec, {
                "schema_version": 1, "version": "figures-test",
                "required_inputs": {
                    "final/components/results.json": sha256(source / "final/components/results.json")},
                "files": {"new.svg": {"path": "final/current/new.svg",
                                      "sha256": sha256(research / "new.svg")}},
            })
            before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
            with patch("data.current_figures.CATALOG", spec):
                plan(source, research)
                self.assertFalse((base / "output").exists())
                output = prepare(source, research, base / "output")
                with self.assertRaises(FileExistsError):
                    prepare(source, research, output)
                with self.assertRaisesRegex(ValueError, "read-only research"):
                    prepare(source, research, research / "forbidden")
            self.assertEqual(before, {path: path.read_bytes() for path in source.rglob("*") if path.is_file()})
            self.assertEqual((output / "paper/old.svg").read_bytes(), b"original-vector")
            index = pack(output, base / "figures.tar.gz")
            installed = download(index, base / "installed", base / "figures.tar.gz")
            self.assertEqual(verify(installed)["version"], "figures-test")
            self.assertEqual((installed / "final/current/new.svg").read_bytes(), b"new-vector")

    def test_changed_saved_inputs_are_refused_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = fixture_bundle(base / "source", {"saved.json": b"{}"})
            research = base / "research"
            research.mkdir()
            (research / "new.svg").write_bytes(b"changed")
            spec = base / "current.json"
            write_json(spec, {
                "schema_version": 1, "version": "test", "required_inputs": {},
                "files": {"new.svg": {"path": "final/new.svg", "sha256": "0" * 64}},
            })
            with patch("data.current_figures.CATALOG", spec):
                with self.assertRaisesRegex(ValueError, "changed saved figure input"):
                    prepare(source, research, base / "output")
            self.assertFalse((base / "output").exists())


if __name__ == "__main__":
    unittest.main()
