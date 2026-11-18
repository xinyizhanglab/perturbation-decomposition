"""Extend a figure bundle with existing saved numerical tables."""

from pathlib import Path
import shutil

from reproduction.io import (
    check_output_location, new_directory, read_json, record_file, safe_path,
    sha256, verify, write_json,
)


TABLES = {
    "scripts/figure_pathway_crosscl/crosscl_pathway_results.json":
        "benchmark/pathway/crosscl_pathway_results.json",
    "scripts/figure_source_vs_depmap/source_vs_depmap_results.json":
        "benchmark/source_vs_depmap/source_vs_depmap_results.json",
}


def prepare(source, benchmark, destination):
    source, benchmark, destination = (Path(p).resolve() for p in (source, benchmark, destination))
    manifest = verify(source)
    if manifest.get("example"):
        raise ValueError("A synthetic example cannot be extended into a manuscript figure bundle.")
    check_output_location(source, destination)
    if destination.is_relative_to(benchmark):
        raise ValueError("Figure bundle destination must be outside the benchmark source project.")
    declared = {entry["path"] for entry in manifest["files"]}
    inputs = {}
    for relative, output in TABLES.items():
        path = safe_path(benchmark, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved figure table: {path}")
        read_json(path)
        if output in declared:
            raise ValueError(f"The figure bundle already contains {output}. Select the earlier bundle.")
        inputs[relative] = sha256(path)
    output = new_directory(destination)
    for entry in manifest["files"]:
        relative = entry["path"]
        path = safe_path(source, relative)
        target = safe_path(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(path) != entry["sha256"] or sha256(target) != entry["sha256"]:
            raise ValueError(f"Source changed during figure-bundle export: {relative}")
    provenance_path = output / "provenance.json"
    provenance = read_json(provenance_path) if provenance_path.exists() else {}
    for relative, target in TABLES.items():
        path = safe_path(benchmark, relative)
        dest = safe_path(output, target)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        if sha256(path) != inputs[relative] or sha256(dest) != inputs[relative]:
            raise ValueError(f"Saved figure table changed during copying: {relative}")
    provenance.setdefault("sources_before", {})["figure_input_benchmark"] = {"path": str(benchmark)}
    provenance["figure_input_export"] = {
        "source_bundle_manifest_sha256": sha256(source / "manifest.json"),
        "tables": {target: {"source": relative, "sha256": inputs[relative]}
                   for relative, target in TABLES.items()},
        "analysis_recomputed": False, "existing_figure_assets_changed": False,
    }
    # Only the newly created bundle's provenance is extended.
    with provenance_path.open("w") as stream:
        import json
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    write_json(output / "manifest.json", {
        **{key: value for key, value in manifest.items() if key not in ("files", "version")},
        "version": manifest["version"] + "-original-inputs-v1",
        "files": [record_file(output, p) for p in sorted(output.rglob("*")) if p.is_file()],
    })
    verify(output)
    return output
