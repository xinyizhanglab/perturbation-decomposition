"""Small export/recipe fixtures; no source datasets or checkpoint inference."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data.distribution import _cached, _manifest, read_legacy_prediction
from models.release import prepared_inputs, registry as bundle_registry, verify_selected
from models.token_policy import recorded_policy, validate_policy
from models.inference_io import prediction_labels
from reproduction.io import write_json


class DistributionTests(unittest.TestCase):
    def test_existing_identifier_schemas_are_accepted_without_ambiguity(self):
        np.testing.assert_array_equal(prediction_labels({"perturbations": ["A", "B"]}), ["A", "B"])
        with self.assertRaisesRegex(ValueError, "aliases disagree"):
            prediction_labels({"test_perts": ["A"], "perturbations": ["B"]})

    def test_legacy_string_axes_require_trust_and_export_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "out"
            source.mkdir()
            output.mkdir()
            path = source / "old.npz"
            values = np.arange(6, dtype=np.float32).reshape(2, 3)
            np.savez(path, test_perts=np.asarray(["A", "B"], dtype=object),
                     predictions=values, true_deltas=values + 1, train_mean=np.zeros(3))
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "trust_legacy_arrays"):
                read_legacy_prediction(path, False)
            registry = {"records": [{"id": "scgpt/within/test/seed0", "model": "scgpt",
                        "assets": [{"root": "benchmark", "path": "old.npz", "role": "saved"}]}]}
            records = _cached(registry, {"benchmark": source}, output, True)
            with np.load(output / records[0]["saved"], allow_pickle=False) as archive:
                self.assertEqual(archive["test_perts"].tolist(), ["A", "B"])
                np.testing.assert_array_equal(archive["predictions"], values)
                np.testing.assert_array_equal(archive["true_deltas"], values + 1)
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_unexpected_legacy_objects_are_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            np.savez(path, test_perts=["A"], predictions=[[1]], mystery=np.asarray([{}], dtype=object))
            with self.assertRaisesRegex(ValueError, "Unexpected object"):
                read_legacy_prediction(path, True)

    def test_missing_gene_axes_are_attached_from_an_explicit_verified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "out"
            source.mkdir()
            output.mkdir()
            np.savez(source / "old.npz", test_perts=["A"], predictions=[[1., 2.]])
            index = {"records": [{"id": "state/example", "model": "state", "target": "k562",
                "assets": [{"root": "release", "path": "old.npz", "role": "saved"}]}]}
            records = _cached(index, {"release": source}, output, True,
                              {("genetic", "k562"): np.asarray(["g1", "g0"])})
            with np.load(output / records[0]["saved"], allow_pickle=False) as archive:
                self.assertEqual(archive["feature_names"].tolist(), ["g1", "g0"])
                np.testing.assert_array_equal(archive["predictions"], [[1., 2.]])

    def test_input_recipe_materializes_only_packaged_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "inputs.npz", test_perts=["A", "B"], feature_names=["g0", "g1"],
                     true_deltas=np.ones((2, 2)))
            np.savez(root / "controls.npz", feature_names=["g0", "g1"],
                     controls=np.arange(8).reshape(4, 2), control_mean=[3, 4])
            destination = root / "materialized.npz"
            prepared_inputs(root, {"inputs": "inputs.npz", "controls": "controls.npz"},
                            destination, limit_perts=1, limit_controls=2)
            with np.load(destination, allow_pickle=False) as archive:
                self.assertEqual(archive["test_perts"].tolist(), ["A"])
                self.assertEqual(archive["controls"].shape, (2, 2))
                np.testing.assert_array_equal(archive["control_mean"], [3, 4])

    def test_input_recipe_rejects_gene_order_mismatch_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "inputs.npz", test_perts=["A"], feature_names=["a", "b"])
            np.savez(root / "controls.npz", feature_names=["b", "a"],
                     controls=[[1, 2]], control_mean=[1, 2])
            with self.assertRaisesRegex(ValueError, "gene axes"):
                prepared_inputs(root, {"inputs": "inputs.npz", "controls": "controls.npz"}, root / "out.npz")
            with self.assertRaises(ValueError):
                prepared_inputs(root, {"inputs": "../outside.npz"}, root / "out.npz")

    def test_token_policy_records_the_existing_pad_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = root / "vocab.json"
            write_json(vocab, {"A": 0, "<pad>": 1})
            policy = recorded_policy(vocab, ["A", "UNKNOWN"])
            self.assertEqual(policy["default_index"], 1)
            self.assertEqual(policy["unknown_genes"], ["UNKNOWN"])
            path = root / "policy.json"
            write_json(path, policy)
            self.assertEqual(validate_policy(path, vocab, ["A", "UNKNOWN"]), policy)
            changed = {**policy, "default_index": 0}
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "default_index"):
                validate_policy(path, vocab, ["A", "UNKNOWN"])

    def test_packaged_registry_and_selected_archive_need_no_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "bundle"
            source.mkdir()
            output.mkdir()
            np.savez(source / "old.npz", test_perts=["A"], predictions=[[1., 2.]])
            index = {"records": [{"id": "ridge/example", "model": "ridge",
                "assets": [{"root": "release", "path": "old.npz", "role": "saved"}]}]}
            records = _cached(index, {"release": source}, output, True)
            write_json(output / "registry.json", {"records": records})
            _manifest(output, "full", {"release": source})
            registry, files = bundle_registry(output)
            verify_selected(output, registry["records"][0], files, saved=True)
            self.assertEqual(registry["records"][0]["saved"], "saved/ridge/example/predictions.npz")
            (output / records[0]["saved"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_selected(output, registry["records"][0], files, saved=True)


if __name__ == "__main__":
    unittest.main()
