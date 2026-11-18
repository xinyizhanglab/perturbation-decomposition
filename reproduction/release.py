"""Build new releases from read-only sources. Never overwrite a destination."""

import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
from collections import defaultdict

import numpy as np

from . import VERSION
from .catalog import BIOLORD, CALIBRATION, CELLS, CONTEXT, MATCHING, SCGEN
from .io import new_directory, read_json, record_file, sha256, write_json


def snapshot_sources(sources):
    result = {}
    for name, root in sources.items():
        root = Path(root).resolve()
        item = {"path": str(root), "exists": root.is_dir()}
        if (root / ".git").is_dir():
            env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                    text=True, env=env, capture_output=True)
            if commit.returncode:
                item["git"] = "No valid Git repository (possibly an empty .git directory)"
                result[name] = item
                continue
            item["commit"] = commit.stdout.strip()
            item["status"] = subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain=v1"], text=True, env=env)
            tracked = subprocess.check_output(
                ["git", "-C", str(root), "ls-files"], text=True, env=env).splitlines()
            item["tracked_sha256"] = {
                p: sha256(root / p) for p in tracked if (root / p).is_file()}
        result[name] = item
    return result


def _copy(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        raise FileExistsError(target)
    before = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if before != sha256(source) or before != sha256(target):
        raise ValueError(f"Source changed during export: {source}")
    return {"source": str(source.resolve()), "source_sha256": before}


def _metadata(model, target, repeat, protocol):
    return {"model": model, "target": target, "repeat": repeat, "setting": "source_only",
            "protocol": protocol, "repeat_kind": "initialization_seed_not_CV",
            "reference_definition": "mean_true_target_delta_over_complete_eligible_cohort",
            "control_subtraction": ("reconstructed_target_control" if model == "biolord"
                                    else "observed_target_control"),
            "cohorts_globally_harmonized": False}


def export_native(sources, output, provenance):
    """Full Figure 3 predictions. scGen reads 27,430 native fit archives."""
    agop = sources["agop"]
    from models.biolord import from_recovered_archive
    from models.scgen import from_binary_archives
    tasks = [json.loads(line) for line in
             (agop / CONTEXT / "manifests/scgen_3to1.jsonl").read_text().splitlines()]
    if len(tasks) != 5486:
        raise ValueError("Unexpected scGen manifest")
    expected = read_json(output / "final/figure3/results.json")["results"]
    from .metrics import evaluate
    for repeat in range(5):
        for target in CELLS:
            for model in ("biolord", "scgen"):
                if model == "biolord":
                    source = agop / BIOLORD / f"seed{repeat}" / target / "predictions.npz"
                    meta = _metadata(model, target, repeat, "biolord_norman_go_pseudobulk_cellline_categorical_v1")
                    data = from_recovered_archive(source, meta)
                    data.metadata["source_sha256"] = sha256(source)
                else:
                    selected = [t for t in tasks if t["target"] == target]
                    paths = [agop / SCGEN / f"seed{repeat}" / target /
                             task["perturbation"] / "predictions.npz" for task in selected]
                    meta = _metadata(model, target, repeat, "scgen_binary_target_controls_real_cell_labels_native_all_v1")
                    data = from_binary_archives(paths, meta)
                    data.metadata["source_fit_sha256"] = {str(p.relative_to(agop)): sha256(p) for p in paths}
                label = "BioLORD" if model == "biolord" else "scGen"
                actual = evaluate(data)["pref"]
                reference = expected[label][target]["values"][repeat]
                # Source reports store scalar values, not arbitrary rounded means.
                if isinstance(reference, dict):
                    reference = reference["pref"]
                np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-7)
                directory = output / "predictions" / model / target / f"repeat{repeat}"
                data.save(directory)
    provenance["native_prediction_validation"] = "40 target/repeat archives checked against final Figure 3"


def _relocate(path, sources):
    """Resolve archived paths against explicitly supplied source roots."""
    path = str(path)
    for marker, key in (("/agop-perturbation/", "agop"),
                        ("/benchmark_2026/", "benchmark"),
                        ("/perturbation-decomposition/", "release")):
        if marker in path:
            return sources[key] / path.split(marker, 1)[1]
    raise ValueError(f"Unmapped archived input path: {path}")


def export_matching(sources, output, provenance):
    """Export small, complete balanced panels, never inference or retraining."""
    agop = sources["agop"]
    root = agop / MATCHING
    manifest = read_json(root / "assembly_manifest.json")
    panels, features = manifest["panels"], np.asarray(manifest["features"])
    native = agop / CONTEXT / "results/component_alignment_all_models_v1_retry1"
    with np.load(native / "inputs/anova.npz", allow_pickle=False) as data:
        contexts = data["cell_lines"].astype(str) if "cell_lines" in data else np.asarray(CELLS)
        perturbations = data["perturbations"].astype(str)
        truth = data["true_deltas"]
        if not np.array_equal(features, data["feature_names"].astype(str)):
            raise ValueError("Matched truth feature order mismatch")
    if contexts.tolist() != manifest["contexts"]:
        raise ValueError("Context order mismatch")
    index = {p: i for i, p in enumerate(perturbations)}
    for repeat, panel in panels.items():
        directory = output / "matched" / f"repeat{repeat}"
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "truth.npz", true_deltas=truth[:, [index[p] for p in panel]],
                            contexts=contexts, perturbations=np.asarray(panel), feature_names=features)
    for assembly in manifest["assemblies"]:
        repeat = str(assembly["repeat"])
        panel = panels[repeat]
        rows, inputs = [], []
        for expected_context, source in zip(manifest["contexts"], assembly["sources"], strict=True):
            if source["context"] != expected_context:
                raise ValueError("Assembly context order mismatch")
            path = _relocate(source["path"], sources)
            before = sha256(path)
            if before != source["sha256"]:
                raise ValueError(f"Frozen prediction changed: {path}")
            with np.load(path, allow_pickle=False) as data:
                if not np.array_equal(features, data["feature_names"].astype(str)):
                    raise ValueError("Matched prediction gene order mismatch")
                labels = data["test_perts"].astype(str).tolist()
                if len(set(labels)) != len(labels):
                    raise ValueError("Duplicate perturbations")
                rows.append(data["predictions"][[labels.index(p) for p in panel]])
            if sha256(path) != before:
                raise ValueError("Input changed during matched export")
            inputs.append({"sha256": before, "source": str(path), "context": expected_context})
        directory = output / "matched" / assembly["model"] / assembly["setting"]
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"repeat{repeat}.npz", predictions=np.stack(rows),
                            contexts=np.asarray(manifest["contexts"]),
                            perturbations=np.asarray(panel), feature_names=features)
        write_json(directory / f"repeat{repeat}.json", {
            "inputs": inputs, "repeat": int(repeat), "model": assembly["model"],
            "setting": assembly["setting"], "cross_target_fit_assembly": True,
            "scgen_cross_perturbation_fit_assembly": assembly["model"] == "scgen"})
    provenance["matching"] = {"panels": panels, "n_assemblies": len(manifest["assemblies"]),
                              "protocol": "independent_balanced_cross_target_assembly"}


def export_genetic(sources, output):
    # Pickle is only read from the explicitly supplied, trusted local source project.
    for cell in CELLS:
        path = sources["release"] / "data/processed/genetic" / f"{cell}.pkl"
        with path.open("rb") as stream:
            data = pickle.load(stream)
        labels = sorted(data["deltas"])
        responses = np.stack([data["deltas"][p] for p in labels])
        names = np.asarray(data["hvg_names"], dtype=str)
        if names.shape != (responses.shape[1],) or len(set(names)) != len(names):
            raise ValueError("Cannot establish unique ordered gene identities")
        directory = output / "genetic"
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / f"{cell}.npz", perturbations=np.asarray(labels),
                            true_deltas=responses, feature_names=names,
                            observed_control_mean=data["ctrl_mean"])


def enrichment_plot_data(annotations, simple, responses):
    """Historical term filtering, without mutating shared enrichment records.

    Selection preserves the historical shared-row FDR assignment order,
    max overlap > 0.5, and up to seven terms/side.
    """
    universe = {p["gene"] for cell in CELLS for p in responses[cell]["per_pert"]}
    term_genes = defaultdict(set)
    for gene in universe:
        for term in annotations.get(gene, []):
            term_genes[term].add(gene)
    output = {}
    for cell in CELLS:
        rows = [dict(row) for row in simple[cell]]
        eligible_sides = {}
        # Preserve the old script's aliasing behavior for terms significant on
        # both sides; do not silently "fix" historical term selection.
        for side in ("depmap", "source"):
            eligible_sides[side] = [row for row in rows
                                   if row[f"fdr_{side}"] < .1 and row[f"fold_{side}"] > 1]
            for row in eligible_sides[side]:
                row["fdr"] = row[f"fdr_{side}"]
        sides = {}
        for side in ("depmap", "source"):
            kept, sets = [], []
            for row in sorted(eligible_sides[side], key=lambda r: r["fdr"]):
                genes = term_genes.get(row["term"], set())
                if not genes or any(len(genes & other)/min(len(genes), len(other)) > .5 for other in sets):
                    continue
                kept.append(row)
                sets.append(genes)
                if len(kept) == 7:
                    break
            sides[side] = kept
        items = []
        for side in ("source", "depmap"):
            rows = reversed(sides[side]) if side == "source" else sides[side]
            for row in rows:
                items.append({"term": row["term"], "name": row["name"][:45],
                              "fdr": row[f"fdr_{side}"], "direction": side,
                              "score": float(np.log2(max(row[f"fold_{side}"], 1.01))) *
                              (-1 if side == "source" else 1)})
        output[cell] = items
    return output


def build(sources, destination, profile="figures"):
    sources = {k: Path(v).resolve() for k, v in sources.items()}
    for name in ("release", "agop", "benchmark"):
        if name not in sources or not sources[name].is_dir():
            raise ValueError(f"Missing source root: {name}")
    destination = Path(destination).resolve()
    if any(destination.is_relative_to(p) for p in sources.values()):
        raise ValueError("Release destination must be outside ALL original source projects")
    if profile not in ("figures", "analysis", "full"):
        raise ValueError("Unknown profile")
    if profile != "figures" and any(str(p).startswith("/mnt/storage01/home/") for p in sources.values()):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Analysis/full export on this cluster requires SLURM; no jobs are submitted automatically")
    before = snapshot_sources(sources)
    output = new_directory(destination)
    provenance = {"sources_before": before, "profile": profile, "version": VERSION,
                  "no_training": True, "no_original_writes": True, "copied": {}}

    def copy(source, relative):
        provenance["copied"][relative] = _copy(source, output / relative)

    paper = sources["agop"] / "scripts/paper_figures_june"
    folders = ("figure1", "figure2", "figure3", "figure4", "figure_1_supplementary",
               "figure3_supplementary", "figure4_supplementary", "decomp_supplementary",
               "supplemetary_figure_2_2", "supplementary_figure_2_3", "figure_pathway", "figure_tahoe")
    for folder in folders:
        for path in sorted((paper / folder).glob("*")):
            if path.is_file() and path.suffix in (".json", ".svg", ".pdf"):
                copy(path, f"paper/{folder}/{path.name}")
    for source_root, target in ((BIOLORD, "figure3"), (SCGEN, "scgen_audit"),
                                (CALIBRATION, "figure4c"), (MATCHING, "components")):
        report = sources["agop"] / source_root
        if target != "components":
            report /= "report"
        for path in sorted(report.iterdir()):
            if path.is_file() and path.suffix in (".json", ".csv", ".md", ".svg", ".pdf"):
                copy(path, f"final/{target}/{path.name}")
        if target == "components":
            copy(report / "component_correlations.npz", "final/components/component_correlations.npz")
    benchmark = sources["benchmark"]
    from data.figure_inputs import TABLES
    for source, target in TABLES.items():
        copy(benchmark / source, target)
    input_paths = {
        "go_annotations": benchmark / "data/go/go_annotations.pkl",
        "enrichment": benchmark / "scripts/figure_enrichment/simple_enrichment_results.json",
        "responses": benchmark / "scripts/figure_source_vs_depmap/source_vs_depmap_results.json",
    }
    initial = {key: sha256(path) for key, path in input_paths.items()}
    # Trusted local input only; the public-facing exported bundle has JSON, not pickle.
    with input_paths["go_annotations"].open("rb") as stream:
        annotations = pickle.load(stream)
    selected = enrichment_plot_data(annotations, read_json(input_paths["enrichment"]),
                                   read_json(input_paths["responses"]))
    write_json(output / "benchmark/enrichment_plot_data.json", selected)
    if initial != {key: sha256(path) for key, path in input_paths.items()}:
        raise ValueError("Enrichment input changed during export")
    provenance["enrichment_sources"] = {
        key: {"source": str(path), "sha256": initial[key]} for key, path in input_paths.items()}
    # Validate the recovered rather than the invalid original BioLORD outputs.
    if not (sources["agop"] / BIOLORD / "recovery_pass").is_file():
        raise ValueError("Final BioLORD recovery completion marker is absent")
    if profile in ("analysis", "full"):
        export_matching(sources, output, provenance)
        export_genetic(sources, output)
    if profile == "full":
        export_native(sources, output, provenance)
    after = snapshot_sources(sources)
    if before != after:
        raise RuntimeError("Original source snapshot changed during export")
    provenance["sources_after_identical"] = True
    write_json(output / "provenance.json", provenance)
    entries = [record_file(output, p) for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "manifest.json", {
        "schema_version": 1, "version": VERSION, "profile": profile,
        "public_release_url": None, "files": entries,
        "note": "Local candidate; not a published release. Manifest is the integrity trust root."})
    return output
