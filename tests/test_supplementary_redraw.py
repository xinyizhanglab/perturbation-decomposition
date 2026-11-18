"""Redraw checks use released small numerical tables, not models or raw cells."""

import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from figures.original import _prepare, _source_records, _worker, catalog
from figures.rendering import profile
from figures.supplementary import SUPPORTED, prepare
from reproduction.catalog import CELLS
from reproduction.io import record_file, write_json


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reproduction_files/figures-v4.tar.gz"
S1 = "paper/figure_1_supplementary/fig1_supp_results.json"
S3 = "paper/figure3_supplementary/fig3_supp_results.json"
S4 = "paper/figure4_supplementary/fig4_supp_results.json"


def contents(name):
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        return archive.extractfile(name).read()


def make_inputs(root, identifier):
    entry = profile(identifier)
    names = {entry["input"], *entry["assets"]}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents(name))
    write_json(root / "manifest.json", {
        "schema_version": 1, "version": "supplementary-test", "profile": "figures",
        "files": [record_file(root, root / name) for name in sorted(names)],
    })
    return root


class SupplementaryRedrawTests(unittest.TestCase):
    def test_saved_tables_and_reference_fonts_are_reused_without_new_downloads(self):
        for identifier in SUPPORTED:
            with self.subTest(identifier=identifier):
                recipe = catalog()[identifier]
                entry = profile(identifier)
                self.assertIn(recipe["input"], (S1, S3, S4))
                self.assertEqual(entry["input"], recipe["input"])
                self.assertEqual(entry["compatibility"], "matched")
                self.assertEqual(entry["rc_overrides"]["font.family"], ["DejaVu Sans"])
                self.assertEqual(entry["rc_overrides"]["svg.fonttype"], "path")
                self.assertEqual(entry["rc_overrides"]["pdf.fonttype"], 3)
                _source_records(identifier)

    def test_partial_profiles_are_exact_subsets_of_the_complete_saved_figures(self):
        for identifier, parent in (
            ("fig1_supp_alignment", "fig1_supplementary_full"),
            ("fig1_supp_anova", "fig1_supplementary_full"),
            ("fig3_supp1_heatmaps", "fig3_supp1"),
            ("fig3_supp1_basis", "fig3_supp1"),
            ("fig3_supp1_baselines", "fig3_supp1"),
        ):
            full = profile(parent, require_verified=False)
            entry = profile(identifier)
            self.assertEqual(len(entry["assets"]), 2)
            self.assertEqual(entry["rc_overrides"], full["rc_overrides"])
            for name, record in entry["assets"].items():
                self.assertEqual(record, full["assets"][name])

    def test_feature_matrices_preserve_cell_and_feature_order(self):
        data = json.loads(contents(S4))
        prepared = prepare("fig4_supplementary", data)
        for group, variable in (("s4a_within_features", "within"), ("s4a_cross_features", "cross")):
            features = data[group]["features"]
            self.assertEqual(prepared[variable.upper() + "_FEATS"], features)
            expected = [[data[group]["values"][cell][feature] for feature in features]
                        for cell in ("K562", "RPE1", "HepG2", "Jurkat")]
            np.testing.assert_array_equal(prepared[variable + "_results"], expected)

    def test_original_float32_variance_fractions_are_restored_without_changing_json(self):
        data = json.loads(contents(S4))
        before = copy.deepcopy(data)
        prepared = _prepare("fig4_supplementary", data)
        self.assertEqual(data, before)
        for model, contexts in prepared["within_decomp"].items():
            for cell, values in contexts.items():
                for key in ("t_frac", "e_frac"):
                    self.assertIsInstance(values[key], np.float32)
                    self.assertEqual(values[key], np.float32(data["s4b_within_decomp"][model][cell][key]))

    def test_supplementary1_uses_saved_scalars_not_anova_or_pca_recalculation(self):
        data = json.loads(contents(S1))
        aligned = _prepare("fig1_supp_alignment", data)
        self.assertEqual(aligned["cos_delta_pc1"], [data["s1b"]["cos_delta_pc1"][c] for c in CELLS])
        anova = _prepare("fig1_supp_anova", data)
        self.assertEqual(anova["cross_ab"], data["s1f"]["cross_dot_products"]["alpha_beta"])
        self.assertEqual(anova["gamma_per_cl"], [data["s1f"]["gamma_per_cl"][c] for c in CELLS])

    def test_supplementary3_preserves_missing_diagonals_and_saved_statistics(self):
        data = json.loads(contents(S3))
        prepared = _prepare("fig3_supp1_basis", data)
        self.assertTrue(np.isnan(np.diag(prepared["mlp_matrix"])).all())
        np.testing.assert_array_equal(prepared["overlap_matrix"],
                                      np.asarray(data["s3d"]["overlap_matrix"], dtype=float))
        self.assertEqual(prepared["rho"], data["s3d"]["spearman_rho"])
        self.assertEqual(prepared["p_val"], data["s3d"]["spearman_p"])
        self.assertIs(_prepare("fig3_supp1_baselines", data)["baseline_results"], data["s3e"])

    def test_all_eleven_redrawn_panels_match_archived_svg_geometry_and_pdf_settings(self):
        comparisons = []
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for identifier in SUPPORTED:
                with self.subTest(identifier=identifier):
                    source = make_inputs(base / identifier / "inputs", identifier)
                    output = base / identifier / "output"
                    output.mkdir()
                    with redirect_stdout(StringIO()):
                        _worker(source, output, identifier)
                    provenance = json.loads((output / "provenance.json").read_text())
                    self.assertFalse(provenance["analysis_recomputed"])
                    self.assertEqual(provenance["mode"], "original")
                    self.assertTrue(all(row["matched"] for row in provenance["rendering_comparisons"]))
                    comparisons.extend(provenance["rendering_comparisons"])
                    self.assertEqual(json.loads((output / "plot_data.json").read_text()),
                                     json.loads((source / profile(identifier)["input"]).read_text()))
        self.assertEqual(sum(row["format"] == "svg" for row in comparisons), 11)
        self.assertEqual(sum(row["format"] == "pdf" for row in comparisons), 11)

    def test_changed_saved_input_is_rejected_even_with_a_rebuilt_local_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = make_inputs(base / "inputs", "fig3_supp1_baselines")
            path = source / S3
            data = json.loads(path.read_text())
            data["s3e"]["Ridge"]["k562"]["Within-CL full"] += .1
            path.write_text(json.dumps(data))
            manifest = json.loads((source / "manifest.json").read_text())
            manifest["files"] = [record_file(source, source / row["path"]) for row in manifest["files"]]
            (source / "manifest.json").write_text(json.dumps(manifest))
            output = base / "output"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "plotting input differs"):
                _worker(source, output, "fig3_supp1_baselines")
            self.assertEqual(list(output.iterdir()), [])

    def test_incomplete_full_figures_are_not_silently_marked_as_complete_redraws(self):
        for identifier in ("fig1_c", "fig1_supplementary_full", "fig2_supp2_1",
                           "fig_tahoe_panels"):
            with self.subTest(identifier=identifier):
                self.assertNotIn(identifier, catalog())
                self.assertNotEqual(profile(identifier, require_verified=False)["compatibility"], "matched")
        with self.assertRaisesRegex(ValueError, "Unknown supplementary"):
            prepare("not-a-figure", {})


if __name__ == "__main__":
    unittest.main()
