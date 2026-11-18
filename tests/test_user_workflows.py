"""Public entrypoint, configuration, and example-workflow tests."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from data.artifacts import create_example, prepare_processed, validate_data
from data.check import validate_config
from figures.render import available
from reproduction.io import verify


ROOT = Path(__file__).resolve().parents[1]


def cli(module, *arguments, cwd=ROOT):
    return subprocess.run([sys.executable, "-B", "-m", module, *map(str, arguments)],
                          cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/perturbation-test-mpl")})


class UserWorkflowTests(unittest.TestCase):
    def test_core_help_entrypoints(self):
        for module in ("data.artifacts", "data.check", "figures.render",
                       "data.process", "data.prepare_release",
                       "models.biolord.train", "models.scgen.train",
                       "models.biolord.predict", "models.scgen.predict",
                       "benchmarks.within_cl", "benchmarks.cross_cl",
                       "benchmarks.norman_doubles", "benchmarks.tahoe_drugs",
                       "benchmarks.pathway_prediction"):
            with self.subTest(module=module):
                result = cli(module, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_documented_example_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = cli("data.artifacts", "example", "--out_dir", base / "inputs")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = verify(base / "inputs")
            self.assertTrue(manifest["example"])
            result = cli("figures.render", "--results_dir", base / "inputs", "--all",
                         "--out_dir", base / "plots")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(available(base / "inputs", "rerender"), ["fig1_dsup"])
            svg = (base / "plots/fig1_dsup/fig1_dsup.svg").read_text()
            self.assertIn("Synthetic example", svg)
            self.assertTrue((base / "plots/fig1_dsup/fig1_dsup.png").is_file())

    def test_example_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example"
            create_example(path)
            with self.assertRaises(FileExistsError):
                create_example(path)

    def test_figure_error_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            result = cli("figures.render", "--results_dir", directory, "--all")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("data.artifacts install", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_file_validation_names_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "k562.pkl"):
                validate_data(directory, "genetic")

    def test_processed_packaging_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source/norman"
            source.mkdir(parents=True)
            for name in ("norman.pkl", "depmap_embeddings.pkl", "split_info.json"):
                (source / name).write_bytes(b"fixture")
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            with patch("benchmarks._runtime.require_compute"):
                output = prepare_processed(base / "source", base / "bundle", "norman")
            self.assertEqual(verify(output)["profile"], "processed")
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})

    def test_config_loads_all_datasets(self):
        config = validate_config(ROOT / "configs/datasets.yaml")
        self.assertEqual(len(config["datasets"]), 6)
        self.assertEqual(config["datasets"]["nadig_hepg2"]["name"], "HepG2")

    def test_misindented_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("datasets:\n  hepg2:\n  name: HepG2\nmodels: {}\n")
            with self.assertRaises(ValueError):
                validate_config(path)

    def test_duplicate_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("datasets: {}\ndatasets: {}\nmodels: {}\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_config(path)

    def test_benchmark_previews_do_not_write_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            for module in ("within_cl", "cross_cl", "norman_doubles", "tahoe_drugs", "pathway_prediction"):
                with self.subTest(module=module):
                    result = cli("benchmarks." + module, "--dry_run", "--out_dir", output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    data = json.loads(result.stdout)
                    self.assertIn("missing_inputs", data)
                    self.assertFalse(output.exists())

    def test_unsupported_pathway_mode_errors(self):
        result = cli("benchmarks.pathway_prediction", "--setting", "within_cl", "--dry_run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not implemented", result.stderr)

    def test_biolord_training_plan_uses_model_entrypoint(self):
        result = cli("models.biolord.train", "--stage", "train", "--task", 0,
                     "--data_dir", "/datasets/raw", "--reference_dir", "/datasets/references",
                     "--out_dir", "/outputs/new")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertFalse(plan["execute"])
        self.assertTrue(plan["command"][2].endswith("biolord_cellline_go_v1.py"))
        self.assertIn("/outputs/new/logs", plan["command"])

    def test_scgen_training_plan_checks_shards(self):
        result = cli("models.scgen.train", "--data_dir", "/datasets/raw",
                     "--reference_dir", "/datasets/references", "--workers", 4, "--shard", 4)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("shard", result.stderr)

    def test_native_execution_requires_allocation(self):
        with patch.dict(os.environ, {}, clear=True):
            result = cli("models.biolord.train", "--stage", "prepare", "--execute",
                         "--data_dir", "/datasets/raw", "--reference_dir", "/datasets/references",
                         "--out_dir", "/outputs/new")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SLURM", result.stderr)


if __name__ == "__main__":
    unittest.main()
