"""Tiny, CPU-only fixtures for model-input packaging and frozen-data staging."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from data.training_inputs import (
    BIOLORD, CALIBRATION, CELLS, CONTEXT, DATASETS, LEGACY_CALIBRATION, PROFILES, SCGEN,
    _copy, export, for_model, install, plan, selected_bundles,
)
from models.prepared_inputs import PROTOCOLS, SCRIPTS, reference_scgen, stage
from reproduction.io import record_file, sha256


ROOT = Path(__file__).resolve().parents[1]


def put(root, name, data=b"recorded input bytes"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def put_json(root, name, value):
    return put(root, name, json.dumps(value, sort_keys=True) + "\n")


def frozen(root, calibration=False):
    source = root / "inputs"
    source.mkdir(parents=True)
    if calibration:
        names = ["legacy_manifest.json", "original_figure4c.json", "anova.npz",
                 "go_features.npz", "scgen_manifest.jsonl", "k562/seed0/source_only.h5ad"]
    else:
        names = ["original_figure3.json", "go_features.npz", "go_coverage.json"]
        names += [f"{cell}/{name}" for cell in CELLS for name in
                  ("training.h5ad", "evaluation.npz", "membership.json")]
    for name in names:
        put(source, name, "unchanged " + name)
    artifacts = {name: sha256(source / name) for name in names}
    contract = json.loads((SCRIPTS.parent / "source_manifest.json").read_text())
    metadata = {"protocol": PROTOCOLS["calibration" if calibration else "biolord"], "artifacts": artifacts}
    filename = "manifest.json" if calibration else "provenance.json"
    if calibration:
        metadata["source_sha256"] = {name: contract[name]["original_sha256"]
                                     for name in ("common.py", "figure4c_author_v2.py")}
        metadata.update(legacy_root="/recorded/source", scgen_root="/recorded/scgen")
    else:
        metadata["runner_sha256"] = contract["biolord_cellline_go_v1.py"]["original_sha256"]
    put_json(source, filename, metadata)
    put(source, "prepare_pass", b"")
    return metadata


def sources(base):
    roots = {name: base / name for name in ("agop", "benchmark", "release")}
    for root in roots.values():
        root.mkdir()
    agop, bench, release = (roots[name] for name in ("agop", "benchmark", "release"))
    put(release, "data/processed/genetic/k562.pkl")
    put(release, "data/processed/hallmark.gmt")
    paths = []
    for cell, dataset in DATASETS.items():
        put(bench, f"data/{cell}/replogle.h5ad", f"{cell} cells, unchanged order and dtype")
        put_json(bench, f"data/{cell}/split_info.json", {"test": ["B"], "train": ["A"]})
        text = f'[datasets]\n{cell} = "{bench}/data/{cell}"\n[fewshot]\ntest = ["B"]\n'
        put(bench, f"runs/STATE_zeroshot/{cell}/seed0/config.toml", text)
        for seed in range(5):
            put(bench, f"runs/STATE_crossCL/{cell}/seed{seed}/config.toml", text)
        morph = put(bench, f"morph_crosscl/data/{cell}_hvg.h5ad", f"{cell} original float64 representation")
        paths.append((cell + "_hvg", str(morph)))
        put(bench, f"morph_crosscl/data/{cell}_hvg_splits.csv", "name,test\nfold0,B\n")
        put(agop, f"morph_pipeline/outputs/{dataset}/pert_embeddings.pkl")
        for name in ("perturb_processed.h5ad", "train_perts.txt", "test_perts.txt", "data_pyg/cell_graphs.pkl"):
            put(bench, f"data/gears/{dataset}/{name}", f"original {dataset} {name}")
        put(bench, f"data/gears/{dataset}/splits/seed0.pkl")
    effect = put(bench, "morph_crosscl/data/depmap_gene_effect.pkl")
    norman = put(bench, "runs/MORPH_norman/norman_hvg.h5ad")
    norman_effect = put(bench, "runs/MORPH_norman/norman_depmap.pkl")
    put(bench, "runs/MORPH_norman/norman_splits.csv")
    paths.append(("norman_hvg", str(norman)))
    put(agop, "morph_pipeline/MORPH/data/scdata_file_path.csv",
        "dataset,file_path\n" + "".join(f"{name},{path}\n" for name, path in paths))
    put(agop, "morph_pipeline/MORPH/data/perturb_embed_file_path.csv",
        f"representation_type,dataset_name,file_path,note\nDepMap_GeneEffect,,{effect},\n"
        f"DepMap_GeneEffect,norman_hvg,{norman_effect},\n")
    for name in ("args.json", "config.json", "vocab.json", "best_model.pt"):
        put(agop, f"data/raw/scgpt_pretrained/{name}")
    put(bench, "scripts/run_scgpt_benchmark.py", "# recorded driver; not executed\n")
    frozen(agop / BIOLORD)
    frozen(agop / CALIBRATION, calibration=True)
    task = {"target": "k562", "perturbation": "A", "sources": ["rpe1", "hepg2", "jurkat"]}
    put_json(agop, f"{CONTEXT}/manifests/scgen_3to1.jsonl", task)
    for seed in range(5):
        for cell in CELLS:
            put(agop, f"{CONTEXT}/results/biolord/three_to_one_corrected_v2/seed{seed}/{cell}/predictions.npz")
        put(agop, f"{CONTEXT}/results/scgen/three_to_one_5seed/seed{seed}/k562/A/predictions.npz")
        for name in ("predictions.npz", "metadata.json", "membership.npz"):
            put(agop, f"{SCGEN}/seed{seed}/k562/A/{name}")
    put_json(agop, f"{SCGEN}/inputs/manifest.jsonl", task)
    put_json(agop, f"{SCGEN}/report/results.json", {"unchanged": True})
    put(agop, "data/biolord_cellline_go_v1/gene2go.pkl")
    put_json(agop, "scripts/paper_figures_june/figure3_crosscl_biolord_corrected_v5_compact/"
             "fig3_crosscl_transfer_biolord_corrected_v5_compact_results.json", {"unchanged": True})
    put_json(agop, f"{LEGACY_CALIBRATION}/inputs/manifest.json", {"splits": {"test": ["B"]}})
    return roots


def hashes(roots):
    return {str(path): sha256(path) for root in roots.values() for path in root.rglob("*") if path.is_file()}


class TrainingInputBundleTests(unittest.TestCase):
    def fixture(self, base):
        roots = sources(base)
        specification = plan(roots)
        self.assertFalse(specification["missing"])
        return roots, specification

    def test_plan_covers_every_model_without_loading_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            roots, specification = self.fixture(Path(directory))
            self.assertEqual(set(specification["profiles"]), set(PROFILES))
            self.assertEqual(specification["profiles"]["genetic"]["file_count"], 33)
            self.assertEqual(specification["profiles"]["biolord"]["generated"], {})
            self.assertFalse(specification["arrays_transformed"])
            for model in ("ridge", "mlp", "morph", "state", "scgpt", "biolord", "scgen"):
                self.assertTrue(for_model(model))
            self.assertEqual(for_model("scgpt", "k562"), ["scgpt-k562"])
            self.assertTrue(all(root.is_dir() for root in roots.values()))

    def test_missing_inputs_are_reported_and_execution_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in ("agop", "benchmark", "release")}
            for path in paths.values():
                path.mkdir()
            specification = plan(paths)
            self.assertTrue(specification["missing"])
            with patch("benchmarks._runtime.require_compute"):
                with self.assertRaisesRegex(ValueError, "missing"):
                    export(specification, root / "out")
            self.assertFalse((root / "out").exists())

    def test_export_install_and_frozen_staging_need_no_research_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots, specification = self.fixture(root)
            before = hashes(roots)
            with patch("benchmarks._runtime.require_compute"):
                output = export(specification, root / "archives", workers=2, compression_level=1)
            self.assertEqual(hashes(roots), before)
            catalog = json.loads((output / "inputs.json").read_text())
            for item in catalog["bundles"].values():
                self.assertTrue((output / item["archive"]).is_file())
            for path in roots.values():
                path.rename(path.with_name(path.name + "-not-mounted"))
            installed = root / "installed"
            targets = install(output / "inputs.json", installed,
                              ["processed", "genetic", "morph", "scgpt-k562", "biolord", "calibration"])
            self.assertEqual(len(targets), 8)
            gene = installed / "inputs/genetic"
            self.assertIn("{data_dir}/k562", (gene / "config.toml").read_text())
            self.assertIn('test = ["B"]', (gene / "config.toml").read_text())
            morph = installed / "inputs/morph"
            self.assertIn("{data_dir}/norman/norman_hvg.h5ad", (morph / "scdata_file_path.csv").read_text())
            self.assertNotIn(str(roots["benchmark"]), (morph / "scdata_file_path.csv").read_text())
            graph = installed / "inputs/scgpt/replogle_k562_essential/data_pyg/cell_graphs.pkl"
            self.assertEqual(graph.read_text(), "original replogle_k562_essential data_pyg/cell_graphs.pkl")
            self.assertTrue((installed / "inputs/scgpt-pretrained/driver.py").is_file())
            refs = installed / "inputs/references"
            self.assertEqual(reference_scgen(refs), (refs / SCGEN).resolve())
            with patch("benchmarks._runtime.require_compute"):
                fit = stage(installed / "inputs/biolord", root / "new-biolord", refs)
                calibrated = stage(installed / "inputs/calibration", root / "new-calibration", refs, True)
            for model in (fit, calibrated):
                self.assertTrue((model / "inputs/prepare_pass").is_file())
                self.assertFalse(json.loads((model / "input_staging.json").read_text())["arrays_changed"])
            prepared = json.loads((fit / "inputs/provenance.json").read_text())
            self.assertEqual(prepared["runner_sha256"], sha256(SCRIPTS / "biolord_cellline_go_v1.py"))
            self.assertTrue((fit / "inputs/source_provenance.json").is_file())
            original_bytes = (installed / "inputs/biolord/inputs/k562/training.h5ad").read_bytes()
            self.assertEqual((fit / "inputs/k562/training.h5ad").read_bytes(), original_bytes)
            calibrated_meta = json.loads((calibrated / "inputs/manifest.json").read_text())
            self.assertEqual(calibrated_meta["scgen_root"], str((refs / SCGEN).resolve()))

    def test_config_export_rebases_symlinked_and_resolved_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            actual = base / "actual"
            actual.mkdir()
            alias = base / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            roots = sources(alias)
            state = roots["benchmark"] / "runs/STATE_zeroshot/k562/seed0/config.toml"
            state.write_text(
                state.read_text() + f'resolved = "{roots["benchmark"].resolve()}/data/k562"\n'
            )
            morph = roots["agop"] / "morph_pipeline/MORPH/data/scdata_file_path.csv"
            morph.write_text(
                morph.read_text()
                + f'resolved,{roots["benchmark"].resolve()}/morph_crosscl/data/k562_hvg.h5ad\n'
            )
            specification = plan(roots)
            before = hashes(roots)
            for bundle, relative in (("genetic", "config.toml"), ("morph", "scdata_file_path.csv")):
                record = next(entry for entry in specification["profiles"][bundle]["files"]
                              if entry["path"] == relative)
                target = base / bundle
                target.mkdir()
                _copy(record, roots, target)
                text = (target / relative).read_text()
                self.assertNotIn(str(alias), text)
                self.assertNotIn(str(actual.resolve()), text)
                self.assertIn("{data_dir}/k562" if bundle == "genetic"
                              else "{data_dir}/k562_hvg.h5ad", text)
            self.assertEqual(hashes(roots), before)

    def test_identical_installed_dependencies_are_verified_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, specification = self.fixture(root)
            with patch("benchmarks._runtime.require_compute"):
                output = export(specification, root / "archives", workers=1, compression_level=0)
            target = root / "installed"
            install(output / "inputs.json", target, ["genetic"])
            before = sha256(target / "inputs/genetic/manifest.json")
            install(output / "inputs.json", target, ["biolord", "genetic"])
            self.assertEqual(sha256(target / "inputs/genetic/manifest.json"), before)
            (target / "inputs/genetic/k562/replogle.h5ad").write_text("modified")
            with self.assertRaises(ValueError):
                install(output / "inputs.json", target, ["genetic"])

    def test_changed_source_is_rejected_without_copying_new_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots, specification = self.fixture(root)
            record = specification["profiles"]["genetic"]["files"][0]
            source = roots[record["root"]] / record["source"]
            source.write_bytes(source.read_bytes() + b"changed")
            destination = root / "copy"
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, "changed after planning"):
                _copy(record, roots, destination)
            self.assertFalse((destination / record["path"]).exists())

    def test_bundle_dependency_cycles_and_unknown_names_fail(self):
        catalog = {"bundles": {"genetic": {"requires": ["references"]},
                               "references": {"requires": ["genetic"]}}}
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            selected_bundles(catalog, ["genetic"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            selected_bundles(catalog, ["not-a-bundle"])

    def test_catalog_index_tampering_fails_before_installing_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, specification = self.fixture(root)
            with patch("benchmarks._runtime.require_compute"):
                output = export(specification, root / "archives", workers=2)
            catalog = json.loads((output / "inputs.json").read_text())
            index = output / catalog["bundles"]["processed"]["index"]
            index.write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                install(output / "inputs.json", root / "installed", ["processed"])
            self.assertFalse((root / "installed").exists())

    def test_exports_require_an_allocation(self):
        with patch("benchmarks._runtime.require_compute", side_effect=RuntimeError("allocation required")):
            with self.assertRaisesRegex(RuntimeError, "allocation required"):
                export({}, "/unused")

    def test_large_installations_require_an_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            put_json(root, "inputs.json", {"schema_version": 1, "bundles": {
                "genetic": {"requires": [], "size": 1024**3}}})
            with patch("benchmarks._runtime.require_compute", side_effect=RuntimeError("allocation required")):
                with self.assertRaisesRegex(RuntimeError, "allocation required"):
                    install(root / "inputs.json", root / "data", ["genetic"])

    def test_frozen_unknown_protocol_is_not_rebound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen(root / "bundle")
            put_json(root / "bundle", "inputs/provenance.json", {"protocol": "different", "artifacts": {}})
            files = [record_file(root / "bundle", path) for path in (root / "bundle").rglob("*") if path.is_file()]
            put_json(root / "bundle", "bundle.json", {"data_kind": "frozen_condition_means"})
            files.append(record_file(root / "bundle", root / "bundle/bundle.json"))
            put_json(root / "bundle", "manifest.json", {"schema_version": 1, "files": files})
            with patch("benchmarks._runtime.require_compute"):
                with self.assertRaisesRegex(ValueError, "protocol"):
                    stage(root / "bundle", root / "fit", root / "references")
            self.assertFalse((root / "fit").exists())

    def test_native_sources_compile_and_keep_function_bodies(self):
        manifest = json.loads((SCRIPTS.parent / "source_manifest.json").read_text())
        for path in SCRIPTS.glob("*.py"):
            compile(path.read_text(), str(path), "exec")
        for name in ("common.py", "figure4c_author_v2.py"):
            path = SCRIPTS / name
            self.assertEqual(sha256(path), manifest[name]["portable_sha256"])
            actual = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                      for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
            self.assertEqual(actual, manifest[name]["function_ast_sha256"])

    def test_packaging_help_does_not_import_native_sdks(self):
        for arguments in (["plan", "--help"], ["export", "--help"], ["install", "--help"]):
            result = subprocess.run([sys.executable, "-B", "-m", "data.training_inputs", *arguments],
                                    cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_prepared_native_cli_prints_a_staging_plan(self):
        result = subprocess.run([sys.executable, "-B", "-m", "models.biolord.train",
            "--stage", "prepare", "--prepared_inputs", "/installed/biolord",
            "--reference_dir", "/installed/references", "--out_dir", "/new/results"],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads(result.stdout)
        self.assertFalse(metadata["execute"])
        self.assertFalse(metadata["resampling"])
        self.assertFalse(metadata["training"])


if __name__ == "__main__":
    unittest.main()
