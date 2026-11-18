"""Prepare current paper-figure downloads from existing saved outputs."""

import argparse
import json
from pathlib import Path
import shutil

from reproduction.io import (
    check_output_location, new_directory, read_json, record_file, safe_path,
    sha256, verify, write_json,
)


CATALOG = Path(__file__).resolve().parents[1] / "reproduction/current_figures.json"


def plan(source, research):
    source, research = Path(source).resolve(), Path(research).resolve()
    manifest = verify(source)
    if manifest.get("profile") != "figures" or manifest.get("example"):
        raise ValueError("Current paper figures require a real saved figure bundle.")
    catalog = read_json(CATALOG)
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported current-figure catalog schema.")
    declared = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    for relative, checksum in catalog["required_inputs"].items():
        if declared.get(relative) != checksum:
            raise ValueError(f"Required frozen plotting input differs or is missing: {relative}")
    targets = []
    for relative, entry in catalog["files"].items():
        path = safe_path(research, relative)
        safe_path(source, entry["path"])
        if entry["path"] in declared or entry["path"] == "manifest.json":
            raise ValueError(f"Refusing to replace an existing artifact: {entry['path']}")
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Missing or changed saved figure input: {relative}")
        targets.append(entry["path"])
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate current-figure output paths.")
    return manifest, catalog


def prepare(source, research, destination):
    source, research, destination = (Path(path).resolve() for path in (source, research, destination))
    manifest, catalog = plan(source, research)
    check_output_location(source, destination)
    if destination.is_relative_to(research):
        raise ValueError("Output must be outside the read-only research project.")
    if destination.exists():
        raise FileExistsError(destination)
    if sum(entry["size"] for entry in manifest["files"]) > 32 * 1024 ** 2:
        from benchmarks._runtime import require_compute
        require_compute("Large figure-data packaging")
    output = new_directory(destination)
    for entry in manifest["files"]:
        relative = entry["path"]
        if relative == "provenance.json":
            continue
        path = safe_path(source, relative)
        target = safe_path(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(path) != entry["sha256"] or sha256(target) != entry["sha256"]:
            raise ValueError(f"Source changed during figure export: {relative}")
    for relative, entry in catalog["files"].items():
        path = safe_path(research, relative)
        target = safe_path(output, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(path) != entry["sha256"] or sha256(target) != entry["sha256"]:
            raise ValueError(f"Saved figure input changed during export: {relative}")
    provenance = read_json(source / "provenance.json") if (source / "provenance.json").is_file() else {}
    provenance.setdefault("sources_before", {})["current_figures"] = {"path": str(research)}
    provenance["current_figures"] = {
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "catalog_sha256": sha256(CATALOG), "files": catalog["files"],
        "analysis_recomputed": False, "metrics_recomputed": False, "models_trained": False,
        "existing_figure_assets_changed": False,
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "manifest.json", {
        **{key: value for key, value in manifest.items() if key not in ("version", "files")},
        "version": catalog["version"],
        "files": [record_file(output, path) for path in sorted(output.rglob("*")) if path.is_file()],
    })
    verify(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True, type=Path)
    parser.add_argument("--research_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            print(prepare(args.source_dir, args.research_dir, args.out_dir))
        else:
            _, catalog = plan(args.source_dir, args.research_dir)
            print(json.dumps({"execute": False, "version": catalog["version"],
                              "out_dir": str(args.out_dir), "new_files": len(catalog["files"]),
                              "analysis_recomputed": False}, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
