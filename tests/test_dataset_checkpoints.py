"""Dataset-specific export plumbing with fake estimators, never model training."""

from pathlib import Path
import pickle
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.tahoe_drugs import CLS_TAHOE, run_compound_cv, run_source_transfer
from data.inference import prepare
from models.checkpoints import Predictor
from reproduction.io import write_json


class PCA:
    def __init__(self, n, **kwargs):
        self.n = n

    def fit(self, values):
        self.components_ = np.eye(values.shape[1])[:self.n]
        self.mean_ = np.zeros(values.shape[1])
        return self

    def transform(self, values):
        return values @ self.components_.T


class Ridge:
    def __init__(self, **kwargs):
        pass

    def fit(self, values, responses):
        self.coef_ = np.zeros((responses.shape[1], values.shape[1]))
        self.intercept_ = responses.mean(0)
        return self

    def predict(self, values):
        return values @ self.coef_.T + self.intercept_


def estimators():
    modules = {name: ModuleType(name) for name in
               ("sklearn", "sklearn.decomposition", "sklearn.linear_model")}
    modules["sklearn.decomposition"].PCA = PCA
    modules["sklearn.linear_model"].Ridge = Ridge
    return modules


class DatasetCheckpointTests(unittest.TestCase):
    def test_tahoe_compound_export_is_predictable(self):
        rng = np.random.RandomState(1)
        responses = {context: {} for context in CLS_TAHOE}
        responses["C32"] = {(f"d{i}", 1.): rng.randn(3) for i in range(30)}
        features = {"Morgan": {f"d{i}": rng.randn(4) for i in range(30)}}
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", estimators()):
            root = Path(directory)
            options = {"root": root, "genes": ["a", "b", "c"], "axis_ids": {"Morgan": "morgan-axis"}}
            run_compound_cv(responses, features, root, options)
            path = root / "checkpoints/compound/C32/Morgan/fold0"
            predictor = Predictor(path)
            with np.load(path / "predictions.npz", allow_pickle=False) as saved:
                labels = saved["test_perts"].astype(str)
                import json
                values = np.asarray([features["Morgan"][json.loads(name)[0]] for name in labels])
                np.testing.assert_allclose(predictor(values, "morgan-axis"), saved["predictions"])

    def test_tahoe_source_export_preserves_source_definition(self):
        rng = np.random.RandomState(2)
        responses = {context: {} for context in CLS_TAHOE}
        for context in ("C32", "HOP62"):
            responses[context] = {(f"d{i}", 1.): rng.randn(3) for i in range(30)}
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", estimators()):
            root = Path(directory)
            run_source_transfer(responses, root, (1,), {"root": root, "genes": ["a", "b", "c"], "axis_ids": {}})
            predictor = Predictor(root / "checkpoints/source/C32/rank1/fold0")
            self.assertEqual(predictor.metadata["feature_builder"], "mean_available_source_response")
            self.assertNotIn("C32", predictor.metadata["source_contexts"])

    def test_norman_pair_sum_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            with (source / "data.pkl").open("wb") as stream:
                pickle.dump({"deltas": {"A+B": np.ones(3)}, "ctrl_mean": np.zeros(3)}, stream)
            with (source / "features.pkl").open("wb") as stream:
                pickle.dump({"A": np.asarray([1., 2.]), "B": np.asarray([3., 4.])}, stream)
            write_json(source / "genes.json", ["g1", "g2", "g3"])
            write_json(source / "test.json", ["A+B"])
            out = prepare(source / "data.pkl", source / "features.pkl", source / "test.json",
                          root / "out", pair_sum=True, gene_names=source / "genes.json")
            with np.load(out / "inputs.npz") as saved:
                np.testing.assert_array_equal(saved["features"], [[4., 6.]])
                self.assertTrue(saved["descriptor_axis_id"].item().endswith(":pair_sum"))


if __name__ == "__main__":
    unittest.main()
