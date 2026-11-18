"""Check supplementary drawings against the archived references."""

from contextlib import redirect_stdout
import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from figures.original import SOURCES, _prepare, _source_records, _worker, catalog
from figures.rendering import profile
from figures.saved_supplementary import SUPPORTED, load_plot_data, restore
from reproduction.io import record_file, write_json


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reproduction_files/figures-v4.tar.gz"
PANELS = {
    "fig1_supp_hallmark": ("fig1_supplementary_full", "s1d_"),
    "fig1_supp_noise": ("fig1_supplementary_full", "s1e_"),
    "fig2_supp_deciles": ("fig2_supp2_1", "s2a_"),
    "fig2_supp_scatter": ("fig2_supp2_1", "s2c_"),
    "fig3_supp1_slopes": ("fig3_supp1", "s3b_"),
    "fig3_supp1_calibration": ("fig3_supp1", "s3c_"),
}


def make_inputs(root, identifier):
    root.mkdir(parents=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for name in profile(identifier)["assets"]:
            output = root / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.extractfile(name).read())
    write_json(root / "manifest.json", {
        "schema_version": 1, "version": "test", "profile": "figures",
        "files": [record_file(root, root / name) for name in sorted(profile(identifier)["assets"])],
    })
    return root


class SavedSupplementaryTests(unittest.TestCase):
    def test_saved_snapshots_have_bound_hashes_and_unchanged_reference_profiles(self):
        manifest = json.loads((SOURCES / "manifest.json").read_text())
        for identifier, (parent, prefix) in PANELS.items():
            with self.subTest(identifier=identifier):
                recipe = catalog()[identifier]
                self.assertIsNone(recipe["input"])
                self.assertIn(recipe["packaged_input"], recipe["dependencies"])
                _source_records(identifier)
                entry = profile(identifier)
                previous = profile(parent, require_verified=False)
                self.assertEqual(entry["rc_overrides"], previous["rc_overrides"])
                self.assertEqual(entry["assets"], {
                    name: record for name, record in previous["assets"].items()
                    if Path(name).name.startswith(prefix)
                })
                path = SOURCES / recipe["packaged_input"]
                raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),
                                 manifest["sources"][path.name]["origin_sha256"])

    def test_scatter_arrays_keep_every_original_point_and_dtype(self):
        recipe = catalog()["fig2_supp_scatter"]
        data = load_plot_data(SOURCES / recipe["packaged_input"])
        curves = _prepare("fig2_supp_scatter", data)["curves"]
        for cell, values in curves.items():
            for key in ("dep_flat", "full_flat", "resid_flat"):
                expected = data["captured"][cell][key]
                self.assertEqual(values[key].dtype, np.dtype(expected["dtype"]))
                self.assertEqual(list(values[key].shape), expected["shape"])
                np.testing.assert_array_equal(values[key], np.asarray(expected["values"], dtype=expected["dtype"]))
            self.assertTrue(np.isfinite(values["rho_full"]))
            self.assertTrue(np.isfinite(values["rho_resid"]))

    def test_scalar_types_survive_json_roundtrip(self):
        value = restore({"kind": "scalar", "dtype": "<f4", "value": .1})
        self.assertIsInstance(value, np.float32)
        self.assertEqual(value, np.float32(.1))
        array = restore({
            "kind": "array", "dtype": "<f8", "shape": [2],
            "values": [1, {"kind": "nonfinite", "value": "nan"}],
        })
        self.assertTrue(np.isnan(array[1]))
        self.assertEqual(array.dtype, np.dtype("float64"))

    def test_unsafe_or_malformed_array_descriptors_fail(self):
        for value in (
            {"kind": "array", "dtype": "O", "shape": [1], "values": [0]},
            {"kind": "array", "dtype": "<f4", "shape": [3], "values": [1, 2]},
            {"kind": "nonfinite", "value": "not-a-number"},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                restore(value)

    def test_compressed_input_is_size_limited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json.gz"
            path.write_bytes(gzip.compress(b"[" + b"0," * 100 + b"0]"))
            with patch("figures.saved_supplementary.MAX_SNAPSHOT_BYTES", 64):
                with self.assertRaisesRegex(ValueError, "size limit"):
                    load_plot_data(path)

    def test_complete_supplementary3_reuses_the_individual_panel_tables(self):
        recipe = catalog()["fig3_supp1"]
        data = load_plot_data(SOURCES / recipe["packaged_input"])
        for identifier, key in (("fig3_supp1_slopes", "slopes"),
                                ("fig3_supp1_calibration", "calibration")):
            child = catalog()[identifier]
            self.assertEqual(data[key], load_plot_data(SOURCES / child["packaged_input"]))
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            table = json.load(archive.extractfile("paper/figure3_supplementary/fig3_supp_results.json"))
        self.assertEqual(data["tables"], table)
        self.assertEqual(len(profile("fig3_supp1")["assets"]), 10)
        self.assertEqual(profile("fig3_supp1")["compatibility"], "matched")

    def test_redraws_and_complete_supplementary3_match_without_model_dependencies(self):
        comparisons = []
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for identifier in SUPPORTED:
                with self.subTest(identifier=identifier):
                    source = make_inputs(base / identifier / "inputs", identifier)
                    output = base / identifier / "output"
                    output.mkdir()
                    with patch.dict(sys.modules, {
                        "sklearn": None, "torch": None, "gseapy": None, "anndata": None,
                    }), redirect_stdout(StringIO()):
                        _worker(source, output, identifier)
                    provenance = json.loads((output / "provenance.json").read_text())
                    self.assertFalse(provenance["analysis_recomputed"])
                    self.assertTrue(all(row["matched"] for row in provenance["rendering_comparisons"]))
                    comparisons.extend(provenance["rendering_comparisons"])
                    recipe = catalog()[identifier]
                    self.assertEqual(json.loads((output / "plot_data.json").read_text()),
                                     load_plot_data(SOURCES / recipe["packaged_input"]))
        self.assertEqual(sum(row["format"] == "svg" for row in comparisons), 11)
        self.assertEqual(sum(row["format"] == "pdf" for row in comparisons), 11)

    def test_tampered_snapshot_is_rejected_before_use(self):
        identifier = "fig2_supp_scatter"
        name = catalog()[identifier]["packaged_input"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / name).write_bytes(b"changed")
            with patch("figures.original.SOURCES", path):
                with self.assertRaisesRegex(ValueError, "packaged saved plotting input changed"):
                    profile(identifier)

    def test_readme_lists_every_new_panel_and_the_complete_figure_command(self):
        text = (ROOT / "README.md").read_text()
        for identifier in SUPPORTED:
            self.assertIn(f"figures.{identifier} --mode original", text)
        for label in ("recovered", "candidate", "job 292551"):
            self.assertNotIn(label, text.lower())


if __name__ == "__main__":
    unittest.main()
