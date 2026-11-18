"""Saved-table adapters and guarded input recovery; no research-data fitting."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data.main_figure_inputs import CELLS, main, recover
from figures.original import SOURCES, _prepare, catalog
from figures.rendering import profile


class MainFigureRecoveryTests(unittest.TestCase):
    def test_new_main_panels_use_packaged_existing_json_results(self):
        for identifier in ("fig1_b", "fig3_b", "fig_tahoe_b_saturation", "fig_tahoe_c_controls"):
            with self.subTest(identifier=identifier):
                recipe = catalog()[identifier]
                source = recipe["packaged_input"]
                self.assertIn(source, recipe["dependencies"])
                self.assertIsNone(recipe["input"])
                self.assertTrue((SOURCES / source).is_file())
                self.assertEqual(profile(identifier)["compatibility"], "matched")

    def test_figure1b_preserves_recovered_values_and_source_scalar_types(self):
        data = json.loads((SOURCES / "fig1_b_saved.json").read_text())
        prepared = _prepare("fig1_b", data)["template_var"]
        self.assertEqual(set(prepared), set(CELLS))
        for cell, value in prepared.items():
            self.assertEqual(float(value), data["template_var"][cell])
            self.assertEqual(str(value.dtype), data["source"]["inputs"][cell]["dtype"])
        self.assertTrue(data["source"]["paper_drawing_match_verified"])
        self.assertFalse(data["source"]["models_trained"])

    def test_figure1c_stays_unsupported_until_a_saved_drawing_matches(self):
        self.assertNotIn("fig1_c", catalog())
        with self.assertRaisesRegex(ValueError, "not rendering-compatible"):
            profile("fig1_c")

    def test_transfer_values_are_forwarded_without_changing_the_saved_cohort(self):
        data = json.loads((SOURCES / "fig3_b_saved.json").read_text())
        result = _prepare("fig3_b", data)["results"]
        for model, targets in data["results"].items():
            for target, entry in targets.items():
                self.assertEqual(result[model][target], entry["values"])
        self.assertFalse(data["source"]["numeric_values_changed"])

    def test_tahoe_table_is_not_recomputed_by_the_plot_adapter(self):
        data = json.loads((SOURCES / "tahoe_testing_battery_saved.json").read_text())
        for identifier in ("fig_tahoe_b_saturation", "fig_tahoe_c_controls"):
            self.assertIs(_prepare(identifier, data)["battery"], data)

    def test_packaged_table_hash_is_bound_to_the_saved_rendering_profile(self):
        recipe = catalog()["fig3_b"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / recipe["packaged_input"]).write_text('{"tampered": true}')
            with patch("figures.original.SOURCES", base):
                with self.assertRaisesRegex(ValueError, "packaged saved plotting input changed"):
                    profile("fig3_b")

    def test_recovery_preview_does_not_read_processed_data_or_fit_pca(self):
        stream = StringIO()
        with patch("data.main_figure_inputs.recover") as run, redirect_stdout(stream):
            self.assertEqual(main(["--data_dir", "/not-installed", "--out_dir", "/not-created"]), 0)
        self.assertFalse(json.loads(stream.getvalue())["execute"])
        run.assert_not_called()

    def test_recovery_requires_an_allocation_and_explicit_pickle_trust(self):
        with patch("benchmarks._runtime.require_compute", side_effect=RuntimeError("allocation")):
            with self.assertRaisesRegex(RuntimeError, "allocation"):
                recover("unused", "unused-output", True)
        with patch("benchmarks._runtime.require_compute"):
            with self.assertRaisesRegex(ValueError, "trust_processed"):
                recover("unused", "unused-output")

    def test_recovery_preserves_processed_inputs_and_records_that_figures_are_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "data"
            data.mkdir()
            before = {}
            for cell in CELLS:
                path = data / f"{cell}.pkl"
                with path.open("wb") as stream:
                    pickle.dump({"deltas": {"a": np.asarray([1., 2., 3.], dtype=np.float32),
                                            "b": np.asarray([2., 1., 4.], dtype=np.float32)}}, stream)
                before[cell] = path.read_bytes()
            with patch("benchmarks._runtime.require_compute"), \
                    patch("data.main_figure_inputs._effective_rank", return_value=np.float32(1.2)) as rank, \
                    patch("data.main_figure_inputs.version", return_value="test-fixture"):
                output = recover(data, base / "output", True)
            self.assertEqual(rank.call_count, 8)
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertFalse(provenance["models_trained"])
            self.assertFalse(provenance["paper_rendering_match_verified"])
            self.assertFalse(provenance["rng_seed_changed"])
            for cell in CELLS:
                self.assertEqual(before[cell], (data / f"{cell}.pkl").read_bytes())


if __name__ == "__main__":
    unittest.main()
