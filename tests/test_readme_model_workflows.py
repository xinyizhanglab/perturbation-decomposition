"""Check the README's model capability claims and non-executing training examples."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
def documented_modules(paths=None):
    commands = []
    for path in paths or [README]:
        for block in re.findall(r"```bash\n(.*?)```", path.read_text(), flags=re.S):
            logical = ""
            for line in block.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                logical += stripped[:-1] + " " if stripped.endswith("\\") else stripped
                if stripped.endswith("\\"):
                    continue
                parts = shlex.split(logical)
                logical = ""
                if "-m" in parts:
                    index = parts.index("-m")
                    commands.append((parts[index + 1], parts[index + 2:]))
                elif "train.py" in parts and path.parent.name in {"biolord", "scgen"}:
                    index = parts.index("train.py")
                    commands.append((f"models.{path.parent.name}.train", parts[index + 1:]))
    return commands


def preview(module, arguments, output):
    arguments = list(arguments)
    if "--out_dir" in arguments:
        arguments[arguments.index("--out_dir") + 1] = str(output)
    return subprocess.run(
        [sys.executable, "-B", "-m", module, *arguments], cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True, capture_output=True)


class ReadmeModelWorkflowTests(unittest.TestCase):
    def test_model_capabilities_are_visible_before_the_quick_start(self):
        introduction = README.read_text().split("## Quick start", 1)[0]
        for term in ("DepMap Ridge/MLP", "MORPH", "STATE", "scGPT", "BioLORD", "scGen",
                     "inference", "retraining"):
            self.assertIn(term, introduction)

    def test_each_model_has_separate_saved_weight_and_training_capabilities(self):
        models = README.read_text().split("## Models\n", 1)[1].split("## Metrics\n", 1)[0]
        self.assertIn("| Model | 1. Saved predictions | 2. Inference | 3. Retraining |", models)
        for model in ("Ridge (DepMap)", "MLP (DepMap)", "MORPH", "STATE", "scGPT", "BioLORD", "scGen"):
            self.assertIn(f"| {model} | Yes |", models)
        self.assertNotIn("Historical scGen weights", models)
        self.assertIn("New fits only", models)
        self.assertIn("not fits to train", models)
        routes = ["**1. Saved predictions**", "**2. Inference**", "**3. Retraining**"]
        positions = [models.index(route) for route in routes]
        self.assertEqual(positions, sorted(positions))
        figures = README.read_text().split("## Pre-computed results\n", 1)[1].split("## Models\n", 1)[0]
        self.assertIn("Regenerate figures from saved results", figures)
        self.assertNotIn("-m models.", figures)

    def test_data_installation_is_model_independent_and_precedes_the_analyses(self):
        text = README.read_text()
        headings = ["## Data\n", "## Pre-computed results\n", "## Main analyses\n", "## Models\n"]
        self.assertEqual([text.index(value) for value in headings],
                         sorted(text.index(value) for value in headings))
        data = text.split("## Data\n", 1)[1].split("## Pre-computed results\n", 1)[0]
        self.assertIn("--bundles processed", data)
        self.assertNotIn("--model", data)
        self.assertIn("data/processed/hallmark.gmt", text)

    def test_decomposition_and_hallmark_have_saved_input_routes_without_models(self):
        text = README.read_text()
        analyses = text.split("## Main analyses\n", 1)[1].split("## Models\n", 1)[0]
        for module in ("figures.fig1_f", "figures.fig1_d", "figures.fig_pathway_crosscl", "figures.fig1_e"):
            with self.subTest(module=module):
                self.assertIn(f"-m {module} --mode original", analyses)
        self.assertNotIn("-m models.", analyses)
        self.assertNotIn("-m benchmarks.", analyses)
        saved = text.split("## Pre-computed results\n", 1)[1].split("## Main analyses\n", 1)[0]
        self.assertIn("--all --mode original", saved)
        self.assertNotIn("--mode reference", saved)
        self.assertIn("not the paper's split-half noise-corrected calculation", analyses)

    def test_readme_uses_plain_file_names_and_action_based_plot_instructions(self):
        text = README.read_text()
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        prose = re.sub(r"`[^`]*`", "", prose)
        self.assertNotRegex(prose.lower(), r"\bbundles?\b|\b(?:original|reference|summary) mode\b")
        self.assertNotIn("BUNDLE_DIR", text)
        self.assertNotIn("INPUT_BUNDLES", text)
        self.assertIn("RESULTS_DOWNLOADS=", text)
        self.assertIn("DATA_DOWNLOADS=", text)
        saved = text.split("## Pre-computed results\n", 1)[1].split("## Main analyses\n", 1)[0]
        self.assertIn("**Regenerate figures from saved results**", saved)
        self.assertIn("--all --mode original", saved)
        self.assertIn("not every paper figure", saved)

    def test_reference_commands_match_saved_figure_coverage(self):
        from reproduction.figures import coverage
        expected = {row["id"] for row in coverage() if row["reference"] and not row["original"]}
        text = README.read_text()
        exported = set(re.findall(r"-m figures\.(\w+) --mode reference", text))
        self.assertEqual(exported, expected)
        for removed in (
                "**Not yet regenerable**",
                "**Optional: copy existing paper figures for comparison**",
                "**Saved figures without a supported redraw**",
                "**copy saved figure only**",
                "Historical scGen weights",
                "MORPH calibration weights were not retained",
                "exact redraw remains unavailable",
                "reproducibility gaps"):
            self.assertNotIn(removed, text)

    def test_models_are_selected_without_long_ids_or_interpreter_paths(self):
        text = README.read_text()
        self.assertIn("models.run --model ridge --target all --saved", text)
        self.assertIn("models.run --model ridge --target all \\", text)
        self.assertIn("models.run --model scgen --target k562", text)
        self.assertIn("| Model | Default experiment |", text)
        self.assertNotIn("--id ", text)
        self.assertNotIn("ID=", text)
        self.assertNotIn("PYTHON=", text)
        self.assertNotRegex(text, r"models/\w+/\.venv/bin/python")
        self.assertIn("Three-source transfer, source-only", text)
        self.assertIn("complete saved binary-fit cohort", text)
        self.assertIn("/path/to/new-binary-fit/checkpoint", text)

    def test_workflow_instructions_do_not_link_out_to_other_files(self):
        links = re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", README.read_text())
        self.assertFalse([link for link in links if not link.startswith(("https://", "http://", "#"))])

    def test_documented_baseline_training_examples_accept_dry_run(self):
        selected = [(module, arguments) for module, arguments in
                    documented_modules()
                    if module in {
                        "benchmarks.within_cl", "benchmarks.cross_cl",
                        "benchmarks.norman_doubles", "benchmarks.tahoe_drugs",
                        "benchmarks.pathway_prediction"}]
        self.assertEqual({module for module, _ in selected}, {
            "benchmarks.within_cl", "benchmarks.cross_cl", "benchmarks.norman_doubles",
            "benchmarks.tahoe_drugs", "benchmarks.pathway_prediction"})
        with tempfile.TemporaryDirectory() as directory:
            for index, (module, arguments) in enumerate(selected):
                with self.subTest(module=module, arguments=arguments):
                    output = Path(directory) / f"run{index}"
                    result = preview(module, [*arguments, "--dry_run"], output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads(result.stdout)
                    self.assertIn("missing_inputs", plan)
                    self.assertFalse(output.exists())

    def test_documented_native_training_commands_are_plans(self):
        native = {"models.biolord.train", "models.scgen.train", "models.morph.train",
                  "models.state.train", "models.scgpt.train"}
        selected = [(module, arguments) for module, arguments in
                    documented_modules() if module in native]
        self.assertEqual({module for module, _ in selected}, native)
        with tempfile.TemporaryDirectory() as directory:
            for index, (module, arguments) in enumerate(selected):
                with self.subTest(module=module, arguments=arguments):
                    self.assertNotIn("--execute", arguments)
                    output = Path(directory) / f"run{index}"
                    result = preview(module, arguments, output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads(result.stdout)
                    self.assertFalse(plan["execute"])
                    if module == "models.scgen.train" and "--save_checkpoints" in arguments:
                        self.assertTrue(plan["save_scgen_checkpoints"])
                        self.assertEqual(plan["checkpoint_tasks"], "0")
                        self.assertIn("64", plan["command"])
                    self.assertFalse(output.exists())

    def test_documented_prediction_panel_produces_separate_component_figures(self):
        from decomposition.run import run
        from figures.components import render
        from models.inference_io import write_predictions
        text = README.read_text()
        manifest = json.loads(re.search(r"<<'JSON'\n(.*?)\nJSON", text, flags=re.S).group(1))
        self.assertEqual(set(manifest["contexts"]), {"k562", "rpe1", "hepg2", "jurkat"})
        rng = np.random.default_rng(7)
        genes = np.asarray([f"g{i}" for i in range(7)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for context in manifest["contexts"]:
                observed = rng.normal(size=(4, 7))
                write_predictions(root / context,
                    {"test_perts": np.asarray(["p0", "p1", "p2", "p3"]),
                     "feature_names": genes, "true_deltas": observed},
                    observed + .1 * rng.normal(size=observed.shape), genes,
                    {"model": "ridge", "target": context})
            path = root / "panel.json"
            path.write_text(json.dumps(manifest))
            run(root / "components", manifest=path)
            render(root / "components", root / "figures")
            for suffix in ("pdf", "svg", "png"):
                self.assertTrue((root / f"figures/components.{suffix}").is_file())
        normalized = " ".join(text.split())
        self.assertIn("new component plots, not replacements for the archived paper panels", normalized)
        self.assertIn("does not automatically rebuild every original paper panel", normalized)


if __name__ == "__main__":
    unittest.main()
