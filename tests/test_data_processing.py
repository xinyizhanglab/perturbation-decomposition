"""Small raw-input fixtures for canonical processed-file output."""

from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from data.process import aligned_feature_names, process_genetic


class Column:
    def __init__(self, values):
        self.values = np.asarray(values)

    def astype(self, dtype):
        return Column(self.values.astype(dtype))


class Variables:
    columns = ("highly_variable",)

    def __getitem__(self, key):
        return np.asarray([True, False, True, True])


def raw_fixture():
    return SimpleNamespace(
        shape=(16, 4), var_names=np.asarray(["g1", "g2", "g3", "g4"]), var=Variables(),
        obs={"gene": Column(["non-targeting"]*6 + ["p1"]*5 + ["p2"]*5)},
        obsm={"X_hvg": np.arange(48).reshape(16, 3)})


class DataProcessingTests(unittest.TestCase):
    def test_canonical_filename_and_fields(self):
        data = raw_fixture()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"anndata": SimpleNamespace(read_h5ad=lambda path: data)}):
                result = process_genetic("example.h5ad", directory, "k562")
            path = Path(directory) / "k562.pkl"
            self.assertTrue(path.is_file())
            with path.open("rb") as stream:
                saved = pickle.load(stream)
            self.assertEqual(saved["hvg_names"], ["g1", "g3", "g4"])
            self.assertEqual(saved["dataset"], "k562")
            expected = np.asarray(data.obsm["X_hvg"], dtype=np.float32)
            np.testing.assert_array_equal(result["deltas"]["p1"],
                                          expected[6:11].mean(0)-expected[:6].mean(0))

    def test_missing_controls_rejected(self):
        data = raw_fixture()
        data.obs["gene"] = Column(["p1"]*16)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"anndata": SimpleNamespace(read_h5ad=lambda path: data)}):
                with self.assertRaisesRegex(ValueError, "No control"):
                    process_genetic("example.h5ad", directory, "k562")
            self.assertFalse((Path(directory) / "k562.pkl").exists())

    def test_ambiguous_gene_order_rejected(self):
        data = raw_fixture()
        data.var = SimpleNamespace(columns=())
        with self.assertRaisesRegex(ValueError, "gene order"):
            aligned_feature_names(data, 3)

    def test_explicit_gene_order(self):
        self.assertEqual(aligned_feature_names(raw_fixture(), 3, ["x", "y", "z"]), ["x", "y", "z"])

    def test_duplicate_genes_rejected(self):
        with self.assertRaises(ValueError):
            aligned_feature_names(raw_fixture(), 3, ["x", "x", "z"])

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "k562.pkl"
            path.write_bytes(b"preserve")
            with patch.dict("sys.modules", {"anndata": SimpleNamespace()}):
                with self.assertRaises(FileExistsError):
                    process_genetic("example.h5ad", directory, "k562")
            self.assertEqual(path.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
