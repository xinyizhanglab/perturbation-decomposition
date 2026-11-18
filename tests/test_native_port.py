"""Test portable path safety and the BioLORD export-axis fix without training."""

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np


class Tensor:
    """Minimal interface used by the export helper; not a replacement model."""
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape, self.ndim, self.dtype = self.data.shape, self.data.ndim, self.data.dtype

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.data, axis))

    def expand(self, n, unused):
        return Tensor(np.broadcast_to(self.data, (n, self.data.shape[-1])))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class NativePortTests(unittest.TestCase):
    def export(self, data, n_observations=1):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "models/_context_generalization/scripts/biolord_cellline_go_v1.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "expression_mean")
        function.decorator_list = []
        namespace = {"np": np, "GO_KEY": "go", "torch": SimpleNamespace(
            as_tensor=lambda values, **kwargs: Tensor(values),
            isfinite=lambda values: np.isfinite(values.data))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<native-export>", "exec"), namespace)
        model = SimpleNamespace(
            device="cpu", adata=SimpleNamespace(n_vars=7),
            module=SimpleNamespace(eval=lambda: None, get_expression=lambda tensors: (Tensor(data), None)))
        return namespace["expression_mean"](model, {"go": Tensor(np.zeros((n_observations, 3)))}, np.ones(3))

    def test_singleton_keeps_gene_vector(self):
        np.testing.assert_array_equal(self.export(np.arange(7)), np.arange(7))

    def test_multiple_observations_mean(self):
        values = np.arange(14).reshape(2, 7)
        np.testing.assert_array_equal(self.export(values, 2), values.mean(axis=0))

    def test_scalar_rejected(self):
        with self.assertRaises(ValueError):
            self.export(np.asarray(2.))

    def test_wrong_gene_count_rejected(self):
        with self.assertRaises(ValueError):
            self.export(np.arange(6))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            self.export(np.full(7, np.nan))

    def test_multiple_observations_cannot_use_squeezed_vector(self):
        with self.assertRaises(ValueError):
            self.export(np.arange(7), 2)

    def test_native_cli_rejects_original_output_override(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([
            sys.executable, "-B", str(root / "models/_context_generalization/run.py"),
            "--model", "biolord", "--reference-root", "/example/original",
            "--data-root", "/example/benchmark/data", "--results-root", "/example/new",
            "--", "train", "--root", "/example/original/do-not-write"],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--root must remain", result.stderr)


if __name__ == "__main__":
    unittest.main()
