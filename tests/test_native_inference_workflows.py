"""Native adapter protocol checks without external packages or GPU inference."""

import argparse
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from models import native_inference
from models.upstream_train import _arguments
from reproduction.io import write_json


class Tensor:
    def __init__(self, values):
        self.value = np.asarray(values)

    def __setitem__(self, key, value):
        self.value[key] = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class StateAdapterTests(unittest.TestCase):
    def setUp(self):
        compatibility = patch("models.state.compat.install",
                              return_value={"recorded_utils_sha256": "synthetic-test-source"})
        compatibility.start()
        self.addCleanup(compatibility.stop)

    def fixture(self):
        data = {"controls": np.arange(12).reshape(4, 3).astype("f"),
                "feature_names": np.asarray(["a", "b", "c"]),
                "perturbation_labels": np.asarray(["p0", "p1"]),
                "test_perts": np.asarray(["p1"])}
        mapping = {"p0": np.asarray([1., 0.]), "p1": np.asarray([0., 1.])}

        class Model:
            @classmethod
            def load_from_checkpoint(cls, *args, **kwargs):
                return cls()

            def eval(self):
                return self

            def to(self, device):
                return self

            def forward(self, batch, padded=False):
                offset = batch["pert_emb"].value[:, 1:2]
                return Tensor(batch["ctrl_cell_emb"].value+offset)

        names = ("state", "state.tx", "state.tx.models", "state.tx.models.state_transition")
        modules = {name: ModuleType(name) for name in names}
        modules[names[-1]].StateTransitionPerturbationModel = Model
        torch = SimpleNamespace(load=lambda *a, **k: mapping, long=np.int64,
            as_tensor=lambda values, **k: Tensor(values),
            zeros=lambda *shape, **k: Tensor(np.zeros(shape, dtype=k.get("dtype", np.float32))),
            inference_mode=nullcontext)
        args = argparse.Namespace(trust_checkpoint=True, device="cpu", seed=2,
                                  cell_set_len=2, control_batches=2, perturbation_map="map.pt")
        return data, args, modules, torch

    def test_real_controls_and_recorded_onehot_map(self):
        data, args, modules, torch = self.fixture()
        with patch.dict(sys.modules, modules), patch("models.native_inference._torch", return_value=torch):
            prediction, genes, meta = native_inference.state("checkpoint", data, args)
        values = []
        for batch in range(2):
            idx = np.random.RandomState(2000+batch).choice(4, 2, replace=False)
            values.append((data["controls"][idx]+1).mean(0))
        np.testing.assert_allclose(prediction[0], np.mean(values, 0)-data["controls"].mean(0))
        self.assertEqual(genes.tolist(), ["a", "b", "c"])
        self.assertEqual(meta["control_batches"], 2)

    def test_reordered_onehot_labels_rejected(self):
        data, args, modules, torch = self.fixture()
        data["perturbation_labels"] = data["perturbation_labels"][::-1]
        with patch.dict(sys.modules, modules), patch("models.native_inference._torch", return_value=torch):
            with self.assertRaisesRegex(ValueError, "ordering"):
                native_inference.state("checkpoint", data, args)

    def test_synthetic_or_insufficient_controls_rejected(self):
        data, args, modules, torch = self.fixture()
        data["controls"] = data["controls"][:1]
        with patch.dict(sys.modules, modules), patch("models.native_inference._torch", return_value=torch):
            with self.assertRaises(ValueError):
                native_inference.state("checkpoint", data, args)

    def test_checkpoint_trust_is_explicit(self):
        data, args, _, _ = self.fixture()
        args.trust_checkpoint = False
        with self.assertRaisesRegex(ValueError, "trust"):
            native_inference.state("checkpoint", data, args)


class NativeContractTests(unittest.TestCase):
    def test_biolord_requires_prepared_inputs(self):
        args = argparse.Namespace(trust_checkpoint=False)
        with self.assertRaisesRegex(ValueError, "requires"):
            native_inference.biolord("checkpoint", {}, args)

    def test_scgen_rejects_wrong_binary_task_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root/"protocol.json", {
                "protocol": "scgen_binary_target_controls_real_cell_labels_native_all_v1",
                "target": "k562", "perturbation": "A"})
            data = {"test_perts": np.asarray(["B"])}
            with self.assertRaisesRegex(ValueError, "recorded target"):
                native_inference.scgen(root, data,
                    argparse.Namespace(trust_checkpoint=True, target="k562"))

    def test_morph_architecture_is_source_identical(self):
        root = Path(__file__).resolve().parents[1]/"models/morph"
        source = (root/"benchmark_model.py").read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
        text = ast.get_source_segment(source, node)
        recorded = json.loads((root/"architecture_source.json").read_text())
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), recorded["class_sha256"])

    def test_upstream_arguments_rebase_only_configured_roots(self):
        values = _arguments({"dataset": "k562", "path": "{data_dir}/x.h5ad", "enabled": True},
                            {"{data_dir}": "/staged/data"}, {"output_dir"})
        self.assertEqual(dict(values)["path"], "/staged/data/x.h5ad")
        self.assertEqual(dict(values)["enabled"], "true")

    def test_upstream_arguments_reject_output_and_secrets(self):
        for values in ({"output_dir": "/original"}, {"api_key": "not-a-key"}):
            with self.assertRaises(ValueError):
                _arguments(values, {}, {"output_dir"})

    def test_scgen_export_is_optional_and_after_prediction(self):
        root = Path(__file__).resolve().parents[1]
        source = (root/"models/_context_generalization/scripts/scgen_author_aligned_v1.py").read_text()
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_predict")
        code = ast.get_source_segment(source, fn)
        self.assertLess(code.index("model.predict("), code.index("model.save("))
        self.assertIn("checkpoint_dir=None", code)
        self.assertIn("save_anndata=True", code)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch environment required")
    def test_small_mean_conditioned_morph_inference(self):
        import torch
        from models.morph.benchmark_model import MORPHModel
        torch.set_num_threads(1)
        torch.manual_seed(9)
        model = MORPHModel(3, 2).eval()
        data = {"test_perts": np.asarray(["p"]), "feature_names": np.asarray(["a", "b", "c"]),
                "descriptors": np.ones((1, 2), dtype="f"), "conditioning_mean": np.arange(3, dtype="f")}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/"model.pt"
            torch.save(model.state_dict(), checkpoint)
            args = argparse.Namespace(device="cpu", seed=0, architecture="benchmark_mean_conditioned")
            predicted, _, _ = native_inference.morph(checkpoint, data, args)
            with torch.inference_mode():
                expected = model(torch.as_tensor(data["conditioning_mean"][None]),
                                 torch.as_tensor(data["descriptors"])).numpy()-data["conditioning_mean"]
            np.testing.assert_array_equal(predicted, expected)


if __name__ == "__main__":
    unittest.main()
