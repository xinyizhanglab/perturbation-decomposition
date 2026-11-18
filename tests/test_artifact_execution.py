"""Packaging and launch-plan tests; no native training or large checkpoints."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data.onehot import ordered_labels
from models.artifacts import bundle_run
from models.upstream_train import stage_and_run
from reproduction.io import verify, write_json


ROOT = Path(__file__).resolve().parents[1]


class ArtifactExecutionTests(unittest.TestCase):
    def test_onehot_axis_is_recorded_not_alphabetical(self):
        result = ordered_labels({"A": np.asarray([0, 1]), "Z": np.asarray([1, 0])})
        self.assertEqual(result, ["Z", "A"])

    def test_non_basis_map_rejected(self):
        with self.assertRaises(ValueError):
            ordered_labels({"A": np.asarray([0., 0.]), "B": np.asarray([1., 0.])})

    def test_new_run_bundle_keeps_checkpoint_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkpoint = base / "source"
            checkpoint.mkdir()
            write_json(checkpoint / "model.json", {"example": True})
            (checkpoint / "weights.npz").write_bytes(b"fixture")
            with patch("benchmarks._runtime.require_compute"):
                output = bundle_run("ridge", checkpoint, base / "bundle")
            self.assertTrue((output / "checkpoint/model.json").is_file())
            self.assertEqual(verify(output)["profile"], "model")
            self.assertEqual((checkpoint / "weights.npz").read_bytes(), b"fixture")

    def test_new_run_bundle_rejects_source_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                bundle_run("ridge", root, root / "bundle")

    def test_state_launch_stages_data_and_controls_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "inputs"
            data.mkdir()
            (data / "config.toml").write_text('path="' + str(data / "dataset.h5ad") + '"\n')
            (data / "dataset.h5ad").write_bytes(b"fixture")
            binary = base / "native/.venv/bin/python"
            binary.parent.mkdir(parents=True)
            binary.write_text("fixture")
            binary.with_name("state").write_text("fixture")
            config = base / "training.json"
            write_json(config, {"arguments": {"data.kwargs.toml_config_path": "{data_dir}/config.toml"}})
            args = argparse.Namespace(model="state", out_dir=base / "out", data_dir=data,
                source_dir=None, pretrained_dir=None, native_python=binary, config=config, driver=None)
            before = (data / "config.toml").read_bytes()
            with patch("benchmarks._runtime.require_compute"), patch("subprocess.call", return_value=0) as run:
                self.assertEqual(stage_and_run(args), 0)
            command = run.call_args.args[0]
            self.assertIn("output_dir=" + str((base / "out").resolve()), command)
            self.assertIn("models.training_runtime", command)
            self.assertIn("use_wandb=false", command)
            self.assertEqual((data / "config.toml").read_bytes(), before)
            self.assertIn(str((base / "out/data").resolve()), (base / "out/data/config.toml").read_text())

    def test_public_workflow_help(self):
        for module in ("models.predict", "models.artifacts", "models.runtime", "data.inference",
                       "data.controls", "data.onehot", "decomposition.run", "figures.components",
                       "benchmarks.pipeline", "models.morph.train", "models.state.train",
                       "models.scgpt.train", "models.morph.infer", "models.state.infer",
                       "models.scgpt.infer", "models.biolord.infer", "models.scgen.infer"):
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, "-B", "-m", module, "--help"],
                                        cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_scgen_retention_plan_does_not_train(self):
        result = subprocess.run([sys.executable, "-B", "-m", "models.scgen.train",
            "--stage", "train", "--shard", "0", "--save_checkpoints", "--checkpoint_tasks", "0,4",
            "--data_dir", "/datasets/raw", "--reference_dir", "/datasets/references",
            "--out_dir", "/outputs/new"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertTrue(plan["save_scgen_checkpoints"])
        self.assertEqual(plan["checkpoint_tasks"], "0,4")
        self.assertFalse(plan["execute"])


if __name__ == "__main__":
    unittest.main()
