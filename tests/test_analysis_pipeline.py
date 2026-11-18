"""Small end-to-end fixtures for export → verify → analysis → regression."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reproduction.analyze import component_scores, matching
from reproduction.catalog import CELLS, CONTEXT, MATCHING
from reproduction.decomposition import compare
from reproduction.io import read_json, record_file, sha256, write_json
from reproduction.release import enrichment_plot_data, export_matching


class EndToEndMatching(unittest.TestCase):
    def make_fixture(self, base):
        agop = base / "agop-perturbation"
        original = agop / MATCHING
        native = agop / CONTEXT / "results/component_alignment_all_models_v1_retry1"
        (native / "inputs").mkdir(parents=True)
        original.mkdir(parents=True)
        rng = np.random.default_rng(27)
        truth = rng.normal(size=(4, 3, 7)).astype(np.float32)
        perturbations, features = np.asarray(["a", "b", "c"]), np.asarray([f"g{i}" for i in range(7)])
        np.savez(native / "inputs/anova.npz", perturbations=perturbations,
                 feature_names=features, true_deltas=truth)
        assemblies, panel_manifest, expected, repeat_scores, pair_arrays = [], {}, {}, {}, {}
        for repeat in range(5):
            panel_manifest[str(repeat)] = perturbations.tolist()
            for model in ("biolord", "scgen"):
                prediction = truth + rng.normal(size=truth.shape)*.3
                scores, pairs = compare(prediction, truth)
                for metric, values in pairs.items():
                    pair_arrays[f"{model}__source_only__repeat{repeat}__{metric}"] = values
                sources = []
                for index, context in enumerate(CELLS):
                    path = agop / "saved" / model / f"repeat{repeat}" / f"{context}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(path, feature_names=features, test_perts=perturbations,
                             predictions=prediction[index])
                    sources.append({"path": str(path), "sha256": sha256(path), "context": context})
                    repeat_scores.setdefault((model, context), []).append({
                        "alpha": scores["alpha"][index], "beta": scores["beta"], "gamma": scores["gamma"][index]})
                assemblies.append({"model": model, "setting": "source_only", "repeat": repeat,
                                   "sources": sources})
        for (model, context), rows in repeat_scores.items():
            entry = {"n_repeats": 5, "test_sizes": [3]*5}
            for metric in ("alpha", "beta", "gamma"):
                values = np.asarray([r[metric] for r in rows])
                entry[metric], entry[f"{metric}_std"] = float(values.mean()), float(values.std(ddof=0))
                entry[f"{metric}_n_defined_repeats"] = int(np.isfinite(values).sum())
            expected.setdefault(model, {})[context] = {"source_only": entry}
        write_json(original / "assembly_manifest.json", {
            "panels": panel_manifest, "features": features.tolist(), "contexts": list(CELLS),
            "assemblies": assemblies})
        output = base / "bundle"
        output.mkdir()
        write_json(output / "final/components/results.json", expected)
        write_json(output / "final/components/assembly_manifest.json", {
            "contexts": list(CELLS), "panels": panel_manifest})
        np.savez_compressed(output / "final/components/component_correlations.npz", **pair_arrays)
        return {"agop": agop, "benchmark": base / "benchmark", "release": base / "release"}, output

    def export_fixture(self, base):
        sources, output = self.make_fixture(base)
        provenance = {}
        export_matching(sources, output, provenance)
        write_json(output / "manifest.json", {"schema_version": 1, "version": "synthetic",
            "profile": "analysis", "files": [record_file(output, p) for p in sorted(output.rglob("*")) if p.is_file()]})
        return sources, output

    def test_independent_export_and_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sources, bundle = self.export_fixture(base)
            original_hashes = {p: sha256(p) for p in sources["agop"].rglob("*") if p.is_file()}
            result = matching(bundle, base / "computed")
            self.assertTrue(read_json(result / "validation.json")["all_reference_metrics_match"])
            self.assertEqual(original_hashes, {p: sha256(p) for p in original_hashes})

    def test_mutated_source_export_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sources, output = self.make_fixture(base)
            path = next((sources["agop"] / "saved").rglob("k562.npz"))
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                export_matching(sources, output, {})

    def test_saved_pair_score_reaggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, bundle = self.export_fixture(base)
            result = component_scores(bundle, base / "scores")
            validation = read_json(result / "validation.json")
            self.assertEqual(validation["n_reference_statistics_checked"], 48)
            self.assertFalse(validation["ANOVA_tensors_recomputed"])

    def test_permuted_axes_rejected_even_with_fresh_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, bundle = self.export_fixture(base)
            path = bundle / "matched/scgen/source_only/repeat0.npz"
            with np.load(path) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            arrays["feature_names"] = arrays["feature_names"][::-1]
            np.savez(path, **arrays)
            manifest = read_json(bundle / "manifest.json")
            manifest["files"] = [record_file(bundle, bundle/e["path"]) for e in manifest["files"]]
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Panel order"):
                matching(bundle, base / "out")


class HistoricalEnrichmentSelection(unittest.TestCase):
    def test_matches_original_term_selection_including_shared_row_behavior(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "tests/fixtures/enrichment_filter.py").read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "remove_redundant")
        namespace = {"MAX_TERMS_PER_SIDE": 7}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<historical-filter>", "exec"), namespace)
        annotations = {"g1": ["t1"], "g2": ["t1", "t2"], "g3": ["t3"]}
        responses = {cell: {"per_pert": [{"gene": g} for g in annotations]} for cell in CELLS}
        simple = {cell: [
            {"term": "t1", "name": "first", "fold_depmap": 2, "fold_source": 2,
             "fdr_depmap": .001, "fdr_source": .08},
            {"term": "t2", "name": "second", "fold_depmap": 2, "fold_source": 1,
             "fdr_depmap": .01, "fdr_source": .9},
            {"term": "t3", "name": "third", "fold_depmap": 1, "fold_source": 3,
             "fdr_depmap": .9, "fdr_source": .01}] for cell in CELLS}
        before = json.dumps(simple, sort_keys=True)
        result = enrichment_plot_data(annotations, simple, responses)
        self.assertEqual(before, json.dumps(simple, sort_keys=True))
        term_genes = {"t1": {"g1", "g2"}, "t2": {"g2"}, "t3": {"g3"}}
        for cell in CELLS:
            rows = [dict(r) for r in simple[cell]]
            dep = [r for r in rows if r["fdr_depmap"] < .1 and r["fold_depmap"] > 1]
            src = [r for r in rows if r["fdr_source"] < .1 and r["fold_source"] > 1]
            for row in dep:
                row["fdr"] = row["fdr_depmap"]
            for row in src:
                row["fdr"] = row["fdr_source"]
            expected = list(reversed(namespace["remove_redundant"](src, term_genes))) + namespace["remove_redundant"](dep, term_genes)
            self.assertEqual([r["term"] for r in result[cell]], [r["term"] for r in expected])


if __name__ == "__main__":
    unittest.main()
