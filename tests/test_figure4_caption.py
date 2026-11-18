"""Figure 4's three panel selections follow the manuscript caption."""

import copy
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

from figures.original import SOURCES, catalog
from figures.rendering import profile
from reproduction.catalog import CELLS, FINAL_REFERENCE_PREFIXES, RENDERABLE


class Figure4CaptionTests(unittest.TestCase):
    def test_paper_panels_use_within_source_and_interaction_comparisons(self):
        expected = {
            "fig4_a": ("fig4_a_current.py", "paper/figure4/fig4a_within_decomp_results.json"),
            "fig4_b": ("fig4_b_current.py", "final/components/results.json"),
            "fig4_c": ("fig4_d_current.py", "final/components/results.json"),
        }
        for identifier, (source, data) in expected.items():
            with self.subTest(identifier=identifier):
                self.assertEqual(catalog()[identifier]["source"], source)
                self.assertEqual(catalog()[identifier]["input"], data)
                self.assertEqual(catalog()[identifier]["call"], "plot")
                self.assertEqual(profile(identifier)["compatibility"], "matched")
        self.assertEqual(RENDERABLE["fig4_b"][1], "components_source")
        self.assertEqual(RENDERABLE["fig4_c"][1], "gamma_matching")
        self.assertEqual(catalog()["fig4_d"]["alias_of"], "fig4_c")
        self.assertEqual(profile("fig4_d")["assets"], profile("fig4_c")["assets"])
        self.assertEqual(FINAL_REFERENCE_PREFIXES["fig4_d"], FINAL_REFERENCE_PREFIXES["fig4_c"])
        self.assertNotEqual(catalog()["fig4_calibration"]["input"], catalog()["fig4_c"]["input"])
        self.assertEqual(catalog()["fig4_calibration"]["source"], "fig4_c_current.py")

    def test_within_panel_uses_saved_template_residual_values_and_clear_labels(self):
        import matplotlib
        from matplotlib.figure import Figure
        payload = {
            model: {cell: {"alpha": [.8, .85], "eps": [.2, .25]} for cell in CELLS}
            for model in ("Ridge", "MLP", "MORPH")
        }
        before, seen = copy.deepcopy(payload), []

        def capture(figure, *args, **kwargs):
            seen.append([text.get_text() for text in figure.axes[0].get_legend().get_texts()])

        with matplotlib.rc_context(), patch.object(Figure, "savefig", capture):
            module = runpy.run_path(str(SOURCES / "fig4_a_current.py"))
            module["plot"](payload, Path("/not-written"))
        self.assertEqual(len(seen), 3)
        for labels in seen:
            self.assertEqual(labels, [
                "Ridge: cell-line template", "Ridge: residual",
                "MLP: cell-line template", "MLP: residual",
                "MORPH: cell-line template", "MORPH: residual",
            ])
        self.assertEqual(payload, before)

    def test_cross_context_panel_excludes_target30_but_retains_morph_controls(self):
        import matplotlib
        from matplotlib.figure import Figure
        models = ("ridge_1to1", "ridge_3to1", "mlp_1to1", "mlp_3to1",
                  "morph", "state", "biolord", "scgen")
        payload = {}
        for model in models:
            payload[model] = {}
            for cell in CELLS:
                settings = ["source_only"]
                if model != "scgen":
                    settings.append("target30")
                if model == "morph":
                    settings.append("controls")
                payload[model][cell] = {
                    setting: {
                        "alpha": .2, "beta": .3, "gamma": .1,
                        "alpha_std": .01, "beta_std": .01, "gamma_std": .01,
                        "test_sizes": [40, 40],
                    } for setting in settings
                }
        before, seen = copy.deepcopy(payload), []

        def capture(figure, *args, **kwargs):
            seen.append({
                "n_axes": len(figure.axes),
                "legend": [text.get_text() for text in figure.legends[0].get_texts()],
                "references": [artist.get_gid() for artist in figure.findobj()
                               if str(artist.get_gid()).startswith("perfect_prediction_reference_")],
                "grid": {line.get_linewidth() for axis in figure.axes for line in axis.get_xgridlines()},
            })

        with matplotlib.rc_context(), patch.object(Figure, "savefig", capture):
            module = runpy.run_path(str(SOURCES / "fig4_b_current.py"))
            selected = module["without_target30"](payload)
            self.assertTrue(all("target30" not in settings for cells in selected.values()
                                for settings in cells.values()))
            self.assertTrue(all("controls" in selected["morph"][cell] for cell in CELLS))
            module["plot"](payload, Path("/not-written"))
        self.assertEqual(len(seen), 3)
        for record in seen:
            self.assertEqual(record["n_axes"], 12)
            self.assertEqual(record["legend"], [
                "No target perturbed cells", "Target-control fine-tuning"])
            self.assertEqual(record["references"], [])
            self.assertEqual(record["grid"], {1.5})
        self.assertEqual(payload, before)


if __name__ == "__main__":
    unittest.main()
