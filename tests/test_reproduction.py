"""Synthetic, CPU-only tests. No source datasets, checkpoints or native models."""

import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import numpy as np

from reproduction.catalog import CELLS, figure_ids
from reproduction.decomposition import (
    compare, decompose, reconstruct, remove_template,
    second_moment_fractions, validate_decomposition,
)
from reproduction.download import download, extract, pack
from reproduction.figures import coverage, render
from reproduction.io import check_output_location, record_file, safe_path, sha256, verify, write_json
from reproduction.metrics import evaluate, gene_correlations
from reproduction.predictions import Predictions, aggregate_binary, native_archive
from reproduction.release import build


def metadata(**updates):
    return {"model": "biolord", "target": "k562", "setting": "source_only",
            "repeat": 0, "protocol": "synthetic", "reference_definition": "explicit_test_vector",
            "control_subtraction": "observed", "repeat_kind": "test", **updates}


def prediction():
    rng = np.random.default_rng(3)
    observed = rng.normal(size=(4, 7))
    return Predictions(np.asarray(["p0", "p1", "p2", "p3"]),
                       np.asarray([f"g{i}" for i in range(7)]),
                       observed + rng.normal(size=(4, 7))*.1,
                       observed, observed.mean(axis=0), metadata())


def bundle(root, files):
    root.mkdir()
    for name, payload in files.items():
        write_json(root / name, payload)
    write_json(root / "manifest.json", {"schema_version": 1, "version": "test",
        "profile": "figures", "files": [record_file(root, root / p) for p in files]})
    return root


class PredictionTests(unittest.TestCase):
    def test_model_adapter_rejects_invalid_biolord_exports(self):
        from models.biolord import from_recovered_archive
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "metrics.json", {"prediction_recovery": {"retrained": True}})
            with self.assertRaises(ValueError):
                from_recovered_archive(root / "predictions.npz", metadata())

    def test_model_adapter_accepts_recovered_biolord(self):
        from models.biolord import from_recovered_archive
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "metrics.json", {
                "target": "k562", "seed": 0, "protocol": "synthetic",
                "prediction_recovery": {"retrained": False, "weights_unchanged": True}})
            np.savez(root / "predictions.npz", test_perts=["a"], feature_names=["x", "y"],
                     predictions=[[1, 2]], true_deltas=[[1, 2]], train_mean=[0, 0])
            result = from_recovered_archive(root / "predictions.npz", metadata())
            self.assertEqual(result.predicted.shape, (1, 2))

    def test_model_adapter_accepts_completed_binary_fit(self):
        from models.scgen import from_binary_archives
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "metadata.json", {
                "target": "k562", "seed": 0, "protocol": "synthetic",
                "old_truth_exact_match": True})
            np.savez(root / "predictions.npz", test_perts=["a"], feature_names=["x", "y"],
                     predictions=[[1, 2]], true_deltas=[[1, 2]])
            result = from_binary_archives([root / "predictions.npz"], metadata(model="scgen"))
            self.assertEqual(result.metadata["model"], "scgen")

    def test_model_adapter_rejects_incomplete_binary_fit(self):
        from models.scgen import from_binary_archives
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "metadata.json", {
                "target": "k562", "seed": 0, "protocol": "synthetic",
                "old_truth_exact_match": False})
            with self.assertRaises(ValueError):
                from_binary_archives([root / "predictions.npz"], metadata(model="scgen"))

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            original = prediction()
            original.save(Path(directory) / "fit")
            loaded = Predictions.load(Path(directory) / "fit")
            np.testing.assert_array_equal(loaded.predicted, original.predicted)
            self.assertEqual(loaded.metadata, original.metadata)

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit"
            prediction().save(path)
            with self.assertRaises(FileExistsError):
                prediction().save(path)

    def test_duplicate_perturbations(self):
        p = prediction()
        p.perturbations[0] = p.perturbations[1]
        with self.assertRaises(ValueError):
            p.validate()

    def test_duplicate_features(self):
        p = prediction()
        p.features[0] = p.features[1]
        with self.assertRaises(ValueError):
            p.validate()

    def test_nonfinite(self):
        p = prediction()
        p.predicted[0, 0] = np.inf
        with self.assertRaises(ValueError):
            p.validate()

    def test_wrong_axes(self):
        p = prediction()
        p.predicted = p.predicted.T
        with self.assertRaises(ValueError):
            p.validate()

    def test_reference_shape(self):
        p = prediction()
        p.reference = np.asarray(1.)
        with self.assertRaises(ValueError):
            p.validate()

    def test_scgen_target30_forbidden(self):
        p = prediction()
        p.metadata.update(model="scgen", setting="target30")
        with self.assertRaises(ValueError):
            p.validate()

    def test_metadata_required(self):
        p = prediction()
        del p.metadata["reference_definition"]
        with self.assertRaises(ValueError):
            p.validate()

    def test_native_gene_identity_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            np.savez(path, test_perts=["a"], predictions=[[1, 2]], true_deltas=[[1, 2]])
            with self.assertRaises(KeyError):
                native_archive(path, metadata())

    def test_binary_complete_cohort_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(2):
                path = Path(directory) / f"{i}.npz"
                np.savez(path, test_perts=[f"p{i}"], feature_names=["a", "b", "c"],
                         predictions=[[i, i+1, i+2]], true_deltas=[[i+1, i+2, i+3]])
                paths.append(path)
            result = aggregate_binary(paths, metadata(model="scgen"))
            np.testing.assert_array_equal(result.reference, [1.5, 2.5, 3.5])

    def test_binary_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, names in enumerate((["a", "b"], ["b", "a"])):
                path = Path(directory) / f"{i}.npz"
                np.savez(path, test_perts=[f"p{i}"], feature_names=names,
                         predictions=[[0, 1]], true_deltas=[[0, 1]])
                paths.append(path)
            with self.assertRaises(ValueError):
                aggregate_binary(paths, metadata(model="scgen"))


class MetricTests(unittest.TestCase):
    def test_explicit_reference_matches_numpy(self):
        p = prediction()
        actual = evaluate(p)
        expected = np.mean([np.corrcoef(y-p.reference, yhat-p.reference)[0, 1]
                            for y, yhat in zip(p.observed, p.predicted)])
        self.assertAlmostEqual(actual["pref"], expected, places=14)

    def test_constant_undefined(self):
        self.assertTrue(np.isnan(gene_correlations(np.ones(3), np.arange(3))))

    def test_constant_legacy_zero(self):
        self.assertEqual(gene_correlations(np.ones(3), np.arange(3), "legacy_zero"), 0)

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            gene_correlations(np.ones(3), np.ones(3), "guess")

    def test_gene_axis(self):
        x = np.random.default_rng(4).normal(size=(3, 5, 7))
        np.testing.assert_allclose(gene_correlations(x, x), np.ones((3, 5)), atol=1e-14)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            gene_correlations([np.nan, 0], [1, 0])


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        self.tensor = np.random.default_rng(6).normal(size=(4, 9, 17))

    def test_reconstruction(self):
        parts = decompose(self.tensor)
        np.testing.assert_allclose(reconstruct(parts), self.tensor, atol=1e-14)

    def test_centering(self):
        self.assertLess(validate_decomposition(self.tensor, decompose(self.tensor)), 1e-14)

    def test_energy_fractions(self):
        parts = decompose(self.tensor)
        self.assertAlmostEqual(sum(second_moment_fractions(parts).values()), 1.)
        self.assertAlmostEqual(sum(np.mean(v**2) for v in parts.values()), np.mean(self.tensor**2))

    def test_perfect_predictions(self):
        scores, pairs = compare(self.tensor.copy(), self.tensor)
        for value in pairs.values():
            np.testing.assert_allclose(value, 1, atol=1e-14)
        np.testing.assert_allclose(scores["gamma"], 1, atol=1e-14)

    def test_independent_prediction_decomposition(self):
        _, pairs = compare(np.zeros_like(self.tensor), self.tensor)
        for value in pairs.values():
            self.assertTrue(np.isnan(value).all())

    def test_shift_changes_mu_only(self):
        first = decompose(self.tensor)
        second = decompose(self.tensor+np.arange(17)[None, None, :])
        for key in ("alpha", "beta", "gamma"):
            np.testing.assert_allclose(first[key], second[key], atol=1e-14)

    def test_missing_data_rejected(self):
        self.tensor[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            decompose(self.tensor)

    def test_incomplete_shape_rejected(self):
        with self.assertRaises(ValueError):
            decompose(self.tensor[:1])

    def test_template_methods_distinct(self):
        template = np.arange(17.)
        projected = remove_template(self.tensor, template, "project")
        np.testing.assert_allclose(projected @ template, 0, atol=1e-11)
        np.testing.assert_allclose(remove_template(self.tensor, template, "subtract"),
                                   self.tensor-template)

    def test_zero_template_rejected(self):
        with self.assertRaises(ValueError):
            remove_template(self.tensor, np.zeros(17), "project")

    def test_strict_means_propagate_undefined(self):
        x = self.tensor.copy()
        x[0] = 0
        scores, _ = compare(np.zeros_like(x), x)
        self.assertTrue(np.isnan(scores["beta"]))


class ArtifactTests(unittest.TestCase):
    def test_output_cannot_be_inside_original_project(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = bundle(base / "bundle", {"provenance.json": {
                "sources_before": {"release": {"path": str(base / "original")}}}})
            with self.assertRaises(ValueError):
                check_output_location(root, base / "original/new-output")
            self.assertFalse((base / "original").exists())

    def test_output_cannot_be_inside_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = bundle(Path(directory) / "bundle", {"a.json": {}})
            with self.assertRaises(ValueError):
                check_output_location(root, root / "new-output")

    def test_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in ("../bad", "/bad", "a/../../bad"):
                with self.assertRaises(ValueError):
                    safe_path(directory, path)

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            (root / "escape").symlink_to(Path(directory), target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_path(root, "escape/file")

    def test_checksum_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = bundle(Path(directory) / "bundle", {"a.json": {"x": 1}})
            verify(root)
            (root / "a.json").write_text('{"x": 2}\n')
            with self.assertRaises(ValueError):
                verify(root)

    def test_manifest_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = bundle(Path(directory) / "bundle", {"a.json": {}})
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["files"] *= 2
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify(root)

    def test_pack_extract_local_download(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = bundle(base / "bundle", {"a.json": {"x": 1}})
            index = pack(root, base / "release.tar.gz")
            result = download(index, base / "downloaded", base / "release.tar.gz")
            self.assertEqual(verify(result)["version"], "test")

    def test_download_requires_published_url_or_local_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = bundle(base / "bundle", {"a.json": {}})
            index = pack(root, base / "release.tar.gz")
            with self.assertRaises(ValueError):
                download(index, base / "downloaded")

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = base / "bad.tar"
            with tarfile.open(archive, "w") as stream:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                stream.addfile(member, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                extract(archive, base / "out", sha256(archive))
            self.assertFalse((base / "escape").exists())

    def test_archive_link_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = base / "bad.tar"
            with tarfile.open(archive, "w") as stream:
                member = tarfile.TarInfo("link")
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                stream.addfile(member)
            with self.assertRaises(ValueError):
                extract(archive, base / "out", sha256(archive))

    def test_bad_checksum_before_destination_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive = base / "bad.tar"
            archive.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                extract(archive, base / "out", "0"*64)
            self.assertFalse((base / "out").exists())

    def test_export_cannot_write_into_original(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = {}
            for key in ("release", "agop", "benchmark"):
                roots[key] = base / key
                roots[key].mkdir()
            with self.assertRaises(ValueError):
                build(roots, roots["release"] / "new-output")


class RenderingTests(unittest.TestCase):
    def test_complete_inventory(self):
        self.assertEqual(len(figure_ids()), 48)
        self.assertEqual(len(coverage()), 48)

    def test_small_numeric_render(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = bundle(base / "bundle", {
                "paper/figure1/fig1dsup_identity_variance_results.json":
                    {cell: {"pct": i} for i, cell in enumerate(CELLS)}})
            result = render(root, base / "out", ["fig1_dsup"], mode="summary")
            self.assertTrue((result / "fig1_dsup/fig1_dsup.svg").is_file())
            self.assertTrue((result / "provenance.json").is_file())

    def test_unsupported_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = bundle(base / "bundle", {"a.json": {}})
            with self.assertRaises(ValueError):
                render(root, base / "out", ["fig1_b"], mode="summary")
            self.assertFalse((base / "out").exists())

    def test_no_native_model_import_in_core(self):
        package = Path(__file__).resolve().parents[1] / "reproduction"
        forbidden = {"torch", "anndata", "biolord", "scgen"}
        for path in package.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    self.assertFalse(forbidden & {v.name.split(".")[0] for v in node.names}, path)
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], forbidden, path)

    def test_cli_help(self):
        result = subprocess.run([sys.executable, "-B", "-m", "reproduction", "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_plan_does_not_load_model(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = [sys.executable, "-B", str(root / "models/_context_generalization/run.py"),
                    "--model", "scgen", "--reference-root", str(base / "references"),
                    "--data-root", str(base / "raw/data"), "--results-root", str(base / "new")]
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((base / "new").exists())

    def test_vendored_source_checksums(self):
        root = Path(__file__).resolve().parents[1] / "models/_context_generalization"
        for name, entry in json.loads((root / "source_manifest.json").read_text()).items():
            self.assertEqual(sha256(root / "source_snapshot" / name), entry["original_sha256"])
            self.assertEqual(sha256(root / "scripts" / name), entry["portable_sha256"])


if __name__ == "__main__":
    unittest.main()
