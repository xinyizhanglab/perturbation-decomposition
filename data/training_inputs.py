"""Package existing model inputs without resampling, normalization, or fitting."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from reproduction.download import download, pack
from reproduction.io import new_directory, path_bindings, read_json, safe_path, sha256, verify, write_json


CELLS = ("k562", "rpe1", "hepg2", "jurkat")
DATASETS = dict(zip(CELLS, ("replogle_k562_essential", "replogle_rpe1_essential",
                           "nadig_hepg2", "nadig_jurkat")))
CONTEXT = "benchmarks/context_generalization"
BIOLORD = f"{CONTEXT}/results/biolord/three_to_one_cellline_go_author_v1"
SCGEN = f"{CONTEXT}/results/scgen/three_to_one_author_aligned_v1_numpycompat"
LEGACY_CALIBRATION = f"{CONTEXT}/results/figure4c_calibration_v1"
CALIBRATION = f"{CONTEXT}/results/figure4c_author_aligned_v2_retry1"
PROJECT = Path(__file__).resolve().parents[1]
INSTALL_LIMIT = 512 * 1024**2
PROFILES = {
    "processed": {"destination": "processed", "models": ["ridge", "mlp"], "requires": []},
    "genetic": {"destination": "inputs/genetic", "models": ["state", "scgen"], "requires": []},
    "morph": {"destination": "inputs/morph", "models": ["morph"], "requires": []},
    **{f"scgpt-{cell}": {"destination": f"inputs/scgpt/{dataset}", "models": ["scgpt"],
                        "requires": ["scgpt-pretrained"]} for cell, dataset in DATASETS.items()},
    "scgpt-pretrained": {"destination": "inputs/scgpt-pretrained", "models": ["scgpt"], "requires": []},
    "biolord": {"destination": "inputs/biolord", "models": ["biolord"], "requires": ["references"]},
    "references": {"destination": "inputs/references", "models": ["biolord", "scgen"], "requires": []},
    "calibration": {"destination": "inputs/calibration", "models": ["biolord"], "requires": ["references"]},
}
MODELS = ("ridge", "mlp", "morph", "state", "scgpt", "biolord", "scgen")
ALLOWED = {".h5ad", ".pkl", ".npz", ".json", ".jsonl", ".csv", ".txt", ".toml", ".gmt"}


def _source(roots, alias, relative):
    return safe_path(roots[alias], relative)


def plan(roots):
    if set(roots) != {"benchmark", "agop", "release"}:
        raise ValueError("Input roots must name benchmark, agop, and release.")
    input_roots = {name: Path(path).absolute() for name, path in roots.items()}
    roots = {name: path.resolve() for name, path in input_roots.items()}
    profiles = {name: {**copy.deepcopy(spec), "files": [], "generated": {}}
                for name, spec in PROFILES.items()}
    missing = []

    def add(bundle, alias, relative, destination=None, bindings=None, expected=None):
        source = _source(roots, alias, relative)
        destination = destination or relative
        safe_path(Path("/bundle"), destination)
        record = {"root": alias, "source": relative, "path": destination,
                  "action": "rebase" if bindings else "copy"}
        if source.is_file():
            stat = source.stat()
            record.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        else:
            record.update(size=None, mtime_ns=None)
            missing.append(f"{bundle}: {alias}/{relative}")
        if bindings:
            record["bindings"] = bindings
        if expected:
            record["expected_sha256"] = expected
        profiles[bundle]["files"].append(record)

    def tree(bundle, alias, relative, destination, expected=None):
        source = _source(roots, alias, relative)
        if not source.is_dir():
            missing.append(f"{bundle}: {alias}/{relative}/")
            return
        for path in sorted(source.rglob("*")):
            if path.is_file() and (path.suffix in ALLOWED or path.name in {"prepare_pass", "pilot_pass"}):
                child = str(path.relative_to(source))
                add(bundle, alias, str(Path(relative) / child),
                    str(Path(destination) / child), expected=(expected or {}).get(child))

    tree("processed", "release", "data/processed", ".")
    for cell in CELLS:
        for filename in ("replogle.h5ad", "split_info.json"):
            add("genetic", "benchmark", f"data/{cell}/{filename}", f"{cell}/{filename}")
        source = f"runs/STATE_zeroshot/{cell}/seed0/config.toml"
        bindings = path_bindings(input_roots["benchmark"] / "data", "{data_dir}")
        add("genetic", "benchmark", source, f"state/source_only/{cell}/config.toml", bindings)
        if cell == "k562":
            add("genetic", "benchmark", source, "config.toml", bindings)
        for seed in range(5):
            add("genetic", "benchmark", f"runs/STATE_crossCL/{cell}/seed{seed}/config.toml",
                f"state/target30/{cell}/seed{seed}/config.toml", bindings)

    morph_bindings = {}
    for cell, dataset in DATASETS.items():
        source = f"morph_crosscl/data/{cell}_hvg.h5ad"
        add("morph", "benchmark", source, f"{cell}_hvg.h5ad")
        morph_bindings.update(path_bindings(
            input_roots["benchmark"] / source, f"{{data_dir}}/{cell}_hvg.h5ad"))
        add("morph", "benchmark", f"morph_crosscl/data/{cell}_hvg_splits.csv", f"{cell}_hvg_splits.csv")
        add("morph", "agop", f"morph_pipeline/outputs/{dataset}/pert_embeddings.pkl",
            f"within_embeddings/{dataset}/pert_embeddings.pkl")
    for source, target in (
        ("morph_crosscl/data/depmap_gene_effect.pkl", "depmap_gene_effect.pkl"),
        ("runs/MORPH_norman/norman_hvg.h5ad", "norman/norman_hvg.h5ad"),
        ("runs/MORPH_norman/norman_depmap.pkl", "norman/norman_depmap.pkl"),
    ):
        add("morph", "benchmark", source, target)
        morph_bindings.update(path_bindings(input_roots["benchmark"] / source, "{data_dir}/" + target))
    add("morph", "benchmark", "runs/MORPH_norman/norman_splits.csv", "norman_hvg_splits.csv")
    for filename in ("scdata_file_path.csv", "perturb_embed_file_path.csv"):
        add("morph", "agop", f"morph_pipeline/MORPH/data/{filename}", filename, morph_bindings)

    base_config = read_json(PROJECT / "configs/training/scgpt.json")
    for cell, dataset in DATASETS.items():
        bundle, source = f"scgpt-{cell}", f"data/gears/{dataset}"
        for filename in ("perturb_processed.h5ad", "train_perts.txt", "test_perts.txt", "data_pyg/cell_graphs.pkl"):
            add(bundle, "benchmark", f"{source}/{filename}", filename)
        tree(bundle, "benchmark", f"{source}/splits", "splits")
        config = copy.deepcopy(base_config)
        config["arguments"].update(dataset=dataset,
            train_perts_file=f"{{data_dir}}/{dataset}/train_perts.txt",
            test_perts_file=f"{{data_dir}}/{dataset}/test_perts.txt")
        profiles[bundle]["generated"]["training.json"] = config
    for filename in ("args.json", "config.json", "vocab.json", "best_model.pt"):
        add("scgpt-pretrained", "agop", f"data/raw/scgpt_pretrained/{filename}", filename)
    add("scgpt-pretrained", "benchmark", "scripts/run_scgpt_benchmark.py", "driver.py")

    provenance_path = _source(roots, "agop", f"{BIOLORD}/inputs/provenance.json")
    expected = read_json(provenance_path).get("artifacts", {}) if provenance_path.is_file() else {}
    tree("biolord", "agop", f"{BIOLORD}/inputs", "inputs", expected)
    frozen = _source(roots, "agop", f"{CALIBRATION}/inputs/manifest.json")
    expected = read_json(frozen).get("artifacts", {}) if frozen.is_file() else {}
    tree("calibration", "agop", f"{CALIBRATION}/inputs", "inputs", expected)

    manifest = f"{CONTEXT}/manifests/scgen_3to1.jsonl"
    add("references", "agop", manifest)
    path = _source(roots, "agop", manifest)
    tasks = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    for seed in range(5):
        for cell in CELLS:
            add("references", "agop",
                f"{CONTEXT}/results/biolord/three_to_one_corrected_v2/seed{seed}/{cell}/predictions.npz")
        for task in tasks:
            suffix = f"seed{seed}/{task['target']}/{task['perturbation']}"
            add("references", "agop", f"{CONTEXT}/results/scgen/three_to_one_5seed/{suffix}/predictions.npz")
            for filename in ("predictions.npz", "metadata.json", "membership.npz"):
                add("references", "agop", f"{SCGEN}/{suffix}/{filename}")
    for relative in (
        f"{SCGEN}/inputs/manifest.jsonl", f"{SCGEN}/report/results.json",
        "data/biolord_cellline_go_v1/gene2go.pkl",
        "scripts/paper_figures_june/figure3_crosscl_biolord_corrected_v5_compact/"
        "fig3_crosscl_transfer_biolord_corrected_v5_compact_results.json",
    ):
        add("references", "agop", relative)
    tree("references", "agop", f"{LEGACY_CALIBRATION}/inputs", f"{LEGACY_CALIBRATION}/inputs")

    for name, spec in profiles.items():
        paths = [entry["path"] for entry in spec["files"]]
        if len(set(paths)) != len(paths):
            raise ValueError(f"Duplicate paths in {name}.")
        spec["source_bytes"] = sum(entry["size"] or 0 for entry in spec["files"])
        spec["file_count"] = len(spec["files"])
    return {"schema_version": 1, "version": "training-inputs-v1",
            "roots": {key: str(value) for key, value in roots.items()},
            "profiles": profiles, "missing": missing,
            "source_bytes": sum(item["source_bytes"] for item in profiles.values()),
            "scgpt_config_sha256": sha256(PROJECT / "configs/training/scgpt.json"),
            "arrays_transformed": False, "training_performed": False, "scores_recalculated": False}


def _copy(record, roots, output):
    source = _source(roots, record["root"], record["source"])
    before = source.stat()
    if (before.st_size, before.st_mtime_ns) != (record["size"], record["mtime_ns"]):
        raise ValueError(f"Source changed after planning: {source}")
    target = safe_path(output, record["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    if record["action"] == "rebase":
        raw = source.read_bytes()
        digest.update(raw)
        text = raw.decode()
        for old, new in sorted(record["bindings"].items(), key=lambda item: len(item[0]), reverse=True):
            text = text.replace(old, new)
        if "/mnt/" in text:
            raise ValueError(f"Unresolved machine path in {record['path']}")
        with target.open("x") as stream:
            stream.write(text)
    elif record["action"] == "copy":
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            while block := incoming.read(1024 * 1024):
                digest.update(block)
                outgoing.write(block)
    else:
        raise ValueError(f"Unknown input action: {record['action']}")
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise ValueError(f"Source changed during export: {source}")
    source_hash, target_hash = digest.hexdigest(), sha256(target)
    if record.get("expected_sha256") and source_hash != record["expected_sha256"]:
        raise ValueError(f"Prepared input no longer matches its recorded checksum: {source}")
    if record["action"] == "copy" and target_hash != source_hash:
        raise ValueError(f"Copy checksum mismatch: {target}")
    return {"path": record["path"], "size": target.stat().st_size, "sha256": target_hash,
            "source_root": record["root"], "source_path": record["source"],
            "source_sha256": source_hash, "operation": record["action"]}


def _profile(name, spec, roots, output, scratch, level):
    with tempfile.TemporaryDirectory(prefix=f"inputs-{name}-", dir=scratch) as directory:
        root = Path(directory)
        records = [_copy(entry, roots, root) for entry in spec["files"]]
        for relative, value in spec.get("generated", {}).items():
            target = safe_path(root, relative)
            write_json(target, value)
            records.append({"path": relative, "size": target.stat().st_size,
                            "sha256": sha256(target), "operation": "configuration_template"})
        metadata = {"schema_version": 1, "bundle": name, "version": f"inputs-{name}-v1",
                    "arrays_transformed": False,
                    "data_kind": {"biolord": "frozen_condition_means",
                                  "calibration": "frozen_calibration"}.get(name, "prepared_inputs")}
        if name == "references":
            metadata["native_scgen_root"] = SCGEN
        write_json(root / "bundle.json", metadata)
        write_json(root / "provenance.json", {
            "sources_before": {alias: {"path": str(path)} for alias, path in roots.items()},
            "operation": "existing_input_export", "resampling": False, "normalization": False})
        for relative in ("bundle.json", "provenance.json"):
            target = root / relative
            records.append({"path": relative, "size": target.stat().st_size, "sha256": sha256(target)})
        write_json(root / "manifest.json", {"schema_version": 1, "version": f"inputs-{name}-v1",
            "profile": "training-inputs", "bundle": name, "files": records})
        index = pack(root, output / f"inputs-{name}-v1.tar.gz", compresslevel=level)
        print(f"PACKED {name}", flush=True)
        return {**{key: spec[key] for key in ("destination", "models", "requires")},
                "index": index.name, "index_sha256": sha256(index), **read_json(index)}


def export(specification, destination, workers=4, compression_level=6):
    from benchmarks._runtime import require_compute
    require_compute("Model training-input export")
    if specification["missing"]:
        raise ValueError("Required input files are missing; regenerate the plan after supplying them.")
    if not 1 <= workers <= 8 or not 0 <= compression_level <= 9:
        raise ValueError("Use 1..8 workers and a gzip level in 0..9.")
    if sha256(PROJECT / "configs/training/scgpt.json") != specification["scgpt_config_sha256"]:
        raise ValueError("The scGPT configuration changed after planning.")
    roots = {key: Path(value).resolve() for key, value in specification["roots"].items()}
    destination = Path(destination).resolve()
    if any(destination.is_relative_to(path) for path in roots.values()):
        raise ValueError("Export must be outside every read-only source repository.")
    scratch = Path(os.environ.get("TMPDIR", tempfile.gettempdir())).resolve()
    sizes = sorted((item["source_bytes"] for item in specification["profiles"].values()), reverse=True)
    scratch_required = sum(sizes[:workers]) + 1024**3
    if shutil.disk_usage(scratch).free < scratch_required:
        raise ValueError(f"Insufficient TMPDIR space: need at least {scratch_required} bytes.")
    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    output_required = int(specification["source_bytes"] * 1.02) + 1024**3
    if shutil.disk_usage(parent).free < output_required:
        raise ValueError(f"Insufficient output space: reserve {output_required} bytes.")
    output = new_directory(destination)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {name: pool.submit(_profile, name, spec, roots, output, scratch, compression_level)
                   for name, spec in specification["profiles"].items()}
        bundles = {name: future.result() for name, future in futures.items()}
    write_json(output / "inputs.json", {"schema_version": 1, "version": specification["version"],
        "bundles": bundles, "arrays_transformed": False, "training_performed": False,
        "scores_recalculated": False})
    return output


def selected_bundles(catalog, names):
    pending, selected, active = list(names), [], set()

    def visit(name):
        if name in active:
            raise ValueError("Cyclic input-bundle dependencies.")
        if name in selected:
            return
        if name not in catalog["bundles"] or name not in PROFILES:
            raise ValueError(f"Unknown input bundle: {name}")
        active.add(name)
        for required in catalog["bundles"][name]["requires"]:
            visit(required)
        active.remove(name)
        selected.append(name)
    for name in pending:
        visit(name)
    return selected


def for_model(model, target="all"):
    if model == "scgpt":
        return [f"scgpt-{cell}" for cell in (CELLS if target == "all" else (target,))]
    return {"ridge": ["processed"], "mlp": ["processed"], "state": ["genetic"],
            "morph": ["morph"], "biolord": ["biolord"], "scgen": ["genetic", "references"]}[model]


def install(catalog_path, destination, names):
    catalog_path, destination = Path(catalog_path).resolve(), Path(destination).resolve()
    catalog = read_json(catalog_path)
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported input catalog.")
    names = selected_bundles(catalog, names)
    if sum(catalog["bundles"][name]["size"] for name in names) > INSTALL_LIMIT:
        from benchmarks._runtime import require_compute
        require_compute("Large model-input installation")
    prepared = []
    for name in names:
        item = catalog["bundles"][name]
        if item["destination"] != PROFILES[name]["destination"]:
            raise ValueError(f"Unexpected destination for {name}.")
        target = safe_path(destination, item["destination"])
        existing = target.exists()
        if existing:
            manifest = target / "manifest.json"
            if not manifest.is_file() or sha256(manifest) != item["manifest_sha256"]:
                raise FileExistsError(f"Different or incomplete inputs already exist: {target}")
            verify(target)
        index = safe_path(catalog_path.parent, item["index"])
        if sha256(index) != item["index_sha256"]:
            raise ValueError(f"Input index checksum mismatch: {name}")
        data = read_json(index)
        for key in ("sha256", "size", "manifest_sha256", "version", "archive"):
            if data.get(key) != item.get(key):
                raise ValueError(f"Input catalog/index disagree: {name} {key}")
        archive = safe_path(catalog_path.parent, item["archive"])
        prepared.append((index, target, archive if archive.is_file() else None, existing))
    return [target if existing else download(index, target, archive)
            for index, target, archive, existing in prepared]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--roots", type=Path, required=True)
    planning.add_argument("--out_dir", type=Path, required=True)
    exporting = commands.add_parser("export")
    exporting.add_argument("--plan", type=Path, required=True)
    exporting.add_argument("--out_dir", type=Path, required=True)
    exporting.add_argument("--workers", type=int, default=4)
    exporting.add_argument("--compression_level", type=int, default=6)
    exporting.add_argument("--execute", action="store_true")
    installing = commands.add_parser("install")
    installing.add_argument("--catalog", type=Path, required=True)
    installing.add_argument("--out_dir", type=Path, default=Path("data"))
    choice = installing.add_mutually_exclusive_group(required=True)
    choice.add_argument("--bundles", help="Comma-separated bundle names")
    choice.add_argument("--model", choices=MODELS)
    installing.add_argument("--target", choices=(*CELLS, "all"), default="all")
    listing = commands.add_parser("list")
    listing.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            specification = plan(read_json(args.roots))
            for path in specification["roots"].values():
                if args.out_dir.resolve().is_relative_to(Path(path)):
                    raise ValueError("Plan output must be outside the read-only source repositories.")
            output = new_directory(args.out_dir)
            write_json(output / "plan.json", specification)
            print(json.dumps({"plan": str(output / "plan.json"),
                "source_bytes": specification["source_bytes"], "missing": specification["missing"],
                "bundles": {name: {"files": item["file_count"], "source_bytes": item["source_bytes"]}
                            for name, item in specification["profiles"].items()}}, indent=2))
        elif args.command == "export":
            specification = read_json(args.plan)
            if not args.execute:
                print(json.dumps({"execute": False, "source_bytes": specification["source_bytes"],
                                  "bundles": list(specification["profiles"])}, indent=2))
            else:
                print(export(specification, args.out_dir, args.workers, args.compression_level))
        elif args.command == "install":
            names = for_model(args.model, args.target) if args.model else args.bundles.split(",")
            print("\n".join(map(str, install(args.catalog, args.out_dir, names))))
        else:
            catalog = read_json(args.catalog)
            for name, item in catalog["bundles"].items():
                print(f"{name:22} {item['size'] / 1e9:8.3f} GB  {item['destination']}")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
