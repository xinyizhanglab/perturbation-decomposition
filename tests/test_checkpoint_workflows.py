"""Portable checkpoints, truth-free inference, and component-pipeline fixtures."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.pipeline import run as pipeline
from data.inference import prepare
from decomposition.run import assemble, run as components
from models.artifacts import discover, package
from models.checkpoints import Predictor, save_arrays, save_ensemble, save_fitted
from models.inference_io import write_predictions
from models.predict import predict_checkpoint, saved_predictions
from reproduction.io import read_json, sha256, verify, write_json


SKLEARN = importlib.util.find_spec("sklearn") is not None
TORCH = importlib.util.find_spec("torch") is not None
FIT_ALLOWED = not Path("/mnt/storage01/home").is_dir() or bool(os.environ.get("SLURM_JOB_ID"))


def ridge_fixture(path, axis="example-axis", target=None, scale=1.):
    arrays = {"pca_components": np.eye(3), "pca_mean": np.zeros(3),
              "ridge_coef": np.eye(3)*scale, "ridge_intercept": np.asarray([.1, .2, .3])}
    return save_arrays(arrays, {"kind": "ridge", "target": target}, path,
                       ["g1", "g2", "g3"], axis)


class CheckpointTests(unittest.TestCase):
    def test_numpy_ridge_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = ridge_fixture(Path(directory) / "fit")
            model = Predictor(path)
            x = np.asarray([[1., 2., 3.]])
            np.testing.assert_array_equal(model(x, "example-axis"), x + [.1, .2, .3])

    def test_wrong_descriptor_axis_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Predictor(ridge_fixture(Path(directory) / "fit"))
            with self.assertRaisesRegex(ValueError, "Descriptor-axis"):
                model(np.ones((2, 3)), "different")

    def test_weights_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = ridge_fixture(Path(directory) / "fit")
            (path / "weights.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                Predictor(path)

    def test_numpy_mlp_checkpoint(self):
        arrays = {"pca_components": np.eye(2), "pca_mean": np.zeros(2),
                  "layer0_weight": np.eye(2), "layer0_bias": np.zeros(2),
                  "layer1_weight": np.asarray([[2., 3.]]), "layer1_bias": np.asarray([1.])}
        with tempfile.TemporaryDirectory() as directory:
            save_arrays(arrays, {"kind": "mlp", "n_layers": 2}, Path(directory) / "fit", ["g"], "axis")
            model = Predictor(Path(directory) / "fit")
            np.testing.assert_array_equal(model([[-2, 4]], "axis"), [[13.]])

    def test_truth_free_prediction_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkpoint = ridge_fixture(base / "checkpoint", target="k562")
            inputs = base / "inputs.npz"
            np.savez(inputs, test_perts=["a", "b"], features=np.ones((2, 3)),
                     descriptor_axis_id="example-axis")
            result = predict_checkpoint(argparse.Namespace(
                checkpoint=checkpoint, inputs=inputs, out_dir=base / "predictions",
                model="ridge", seed=0, device="cpu", target="k562"))
            with np.load(result / "predictions.npz") as saved:
                self.assertNotIn("true_deltas", saved)
            self.assertFalse(read_json(result / "metadata.json")["has_evaluation_truth"])

    def test_wrong_model_family_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkpoint = ridge_fixture(base / "checkpoint")
            np.savez(base / "inputs.npz", test_perts=["a"], features=np.ones((1, 3)),
                     descriptor_axis_id="example-axis")
            with self.assertRaisesRegex(ValueError, "model family"):
                predict_checkpoint(argparse.Namespace(
                    checkpoint=checkpoint, inputs=base/"inputs.npz", out_dir=base/"out",
                    model="mlp", seed=0, device="cpu", target=None))

    @unittest.skipUnless(SKLEARN and FIT_ALLOWED, "analysis environment and compute test context required")
    def test_fitted_ridge_roundtrip(self):
        from models.ridge import train_ridge
        rng = np.random.RandomState(2)
        x, y, test = rng.randn(24, 5), rng.randn(24, 3), rng.randn(4, 5)
        fn, _ = train_ridge(x, y, n_pca=3)
        with tempfile.TemporaryDirectory() as directory:
            save_fitted(fn, Path(directory)/"fit", ["a", "b", "c"], "axis")
            np.testing.assert_allclose(Predictor(Path(directory)/"fit")(test, "axis"),
                                       fn(test), atol=1e-12, rtol=1e-12)

    @unittest.skipUnless(SKLEARN and TORCH and FIT_ALLOWED, "PyTorch/analysis environment and compute context required")
    def test_small_mlp_roundtrip_and_ensemble(self):
        from models.ridge import train_mlp
        rng = np.random.RandomState(3)
        x, y = rng.randn(16, 4).astype("f"), rng.randn(16, 3).astype("f")
        models = [train_mlp(x, y, n_pca=3, hidden_dims=(4, 4), epochs=2,
                            batch_size=8, device="cpu", seed=i)[0] for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ensemble"
            save_ensemble(models, path, ["a", "b", "c"], "axis")
            expected = sum(model(x[:2]) / 2 for model in models)
            np.testing.assert_allclose(Predictor(path)(x[:2], "axis"), expected, atol=1e-6, rtol=1e-5)


class InferenceInputTests(unittest.TestCase):
    def test_processed_input_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "source"
            data.mkdir()
            with (data/"cell.pkl").open("wb") as stream:
                pickle.dump({"deltas": {"a": np.ones(3)}, "ctrl_mean": np.zeros(3),
                             "hvg_names": ["g1", "g2", "g3"]}, stream)
            with (data/"depmap.pkl").open("wb") as stream:
                pickle.dump({"a": np.ones(4)}, stream)
            write_json(data/"perts.json", ["a"])
            output = prepare(data/"cell.pkl", data/"depmap.pkl", data/"perts.json", base/"out")
            with np.load(output/"inputs.npz") as saved:
                self.assertEqual(saved["descriptor_axis_id"].item(), "sha256:"+sha256(data/"depmap.pkl"))
                self.assertNotIn("controls", saved)
                np.testing.assert_array_equal(saved["true_deltas"], np.ones((1, 3)))

    def test_saved_export_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base/"source"
            source.mkdir()
            path = source/"p.npz"
            np.savez(path, test_perts=["a"], predictions=np.zeros((1, 3)))
            result = saved_predictions(path, base/"copy")
            self.assertFalse(result["has_truth"])
            self.assertEqual(path.read_bytes(), (base/"copy/predictions.npz").read_bytes())

    def test_no_fake_truth_inserted(self):
        with tempfile.TemporaryDirectory() as directory:
            result = write_predictions(Path(directory)/"out", {"test_perts": np.asarray(["a"])},
                                       np.ones((1, 2)), ["g1", "g2"], {"model": "example"})
            with np.load(result/"predictions.npz") as saved:
                self.assertNotIn("true_deltas", saved)
                self.assertNotIn("train_mean", saved)


class ArtifactDiscoveryTests(unittest.TestCase):
    def test_discovery_does_not_claim_runtime_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = {name: base/name for name in ("agop", "benchmark", "release")}
            for path in roots.values():
                path.mkdir()
            registry = discover(roots)
            scgen = [r for r in registry["records"] if r["model"] == "scgen"]
            self.assertTrue(all(not r["weights_available"] for r in scgen))
            self.assertTrue(all(r["checkpoint_status"] == "not_saved_in_original_run" for r in scgen))
            self.assertTrue(all(r["runtime_validation"] == "not_established_by_file_discovery" for r in scgen))

    def test_checkpoint_bundle_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base/"source"
            source.mkdir()
            (source/"model.pt").write_bytes(b"fixture weights")
            registry = {"records": [{"id": "test", "weights_available": True, "assets": [{
                "root": "example", "path": "model.pt", "destination": "weights/model.pt",
                "role": "weights", "available": True}]}]}
            with patch("benchmarks._runtime.require_compute"):
                out = package(registry, "test", {"example": source}, base/"out", "weights")
            self.assertEqual(verify(out)["profile"], "model")
            self.assertEqual((source/"model.pt").read_bytes(), b"fixture weights")

    def test_missing_weights_not_replaced_with_saved_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            registry = {"records": [{"id": "scgen", "weights_available": False, "assets": []}]}
            with self.assertRaisesRegex(ValueError, "no saved checkpoint"):
                package(registry, "scgen", {"example": base/"source"}, base/"out", "weights")


class DecompositionWorkflowTests(unittest.TestCase):
    def fixture(self, base):
        rng = np.random.RandomState(6)
        files = {}
        for context in ("a", "b"):
            path = base / "inputs" / context
            path.mkdir(parents=True)
            truth = rng.randn(3, 4)
            np.savez(path/"predictions.npz", test_perts=["p1", "p2", "p3"],
                     feature_names=["g1", "g2", "g3", "g4"], predictions=truth+.1,
                     true_deltas=truth)
            files[context] = str(path/"predictions.npz")
        write_json(base/"panel.json", {"contexts": ["a", "b"], "predictions": files})
        return base/"panel.json"

    def test_labeled_manifest_to_components(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = self.fixture(base)
            out = components(base/"analysis", manifest=manifest)
            self.assertTrue((out/"predictions_components.npz").is_file())
            self.assertLess(read_json(out/"summary.json")["results"]["predictions"]["reconstruction_error"], 1e-12)

    def test_duplicate_context_archive_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = self.fixture(base)
            spec = read_json(manifest)
            spec["predictions"]["b"] = spec["predictions"]["a"]
            manifest.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                assemble(manifest)

    def test_full_checkpoint_replay_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model_entries = {}
            rng = np.random.RandomState(4)
            for index, context in enumerate(("a", "b")):
                path = base / "sources" / context
                path.mkdir(parents=True)
                ridge_fixture(path/"fit", target=context, scale=1.+index*.1)
                values = rng.randn(4, 3)
                np.savez(path/"input.npz", test_perts=["p1", "p2", "p3", "p4"],
                         feature_names=["g1", "g2", "g3"], features=values,
                         true_deltas=values+.1*index, descriptor_axis_id="example-axis")
                model_entries[context] = {"model": "ridge", "checkpoint": str(path/"fit"),
                                          "inputs": str(path/"input.npz")}
            write_json(base/"spec.json", {"contexts": ["a", "b"], "models": model_entries})
            out = pipeline(base/"spec.json", base/"pipeline")
            self.assertTrue((out/"figures/components.svg").is_file())
            self.assertTrue((out/"components/predictions_components.npz").is_file())
            self.assertFalse(read_json(out/"provenance.json")["training"])


if __name__ == "__main__":
    unittest.main()
