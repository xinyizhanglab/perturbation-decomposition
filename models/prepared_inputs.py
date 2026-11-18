"""Stage verified frozen BioLORD inputs without repeating preprocessing."""

from pathlib import Path
import shutil

from reproduction.io import check_output_location, new_directory, read_json, safe_path, sha256, verify, write_json


SCRIPTS = Path(__file__).resolve().parent / "_context_generalization/scripts"
EXPERIMENTS = {
    "biolord": "biolord/three_to_one_cellline_go_author_v1",
    "calibration": "figure4c_author_aligned_v2",
}
PROTOCOLS = {"biolord": "biolord_norman_go_pseudobulk_cellline_categorical_v1",
             "calibration": "fig4c_go_cellline_biolord_binary_scgen_baseline_only_v2"}


def _recorded_source(name, expected):
    contract = read_json(SCRIPTS.parent / "source_manifest.json")
    if name not in contract or expected not in {
            contract[name]["original_sha256"], contract[name]["portable_sha256"],
            sha256(safe_path(SCRIPTS, name))}:
        raise ValueError(f"Unrecognized recorded preparation source: {name}")


def reference_scgen(root):
    root = Path(root)
    marker = root / "bundle.json"
    if not marker.is_file():
        return None
    metadata = read_json(marker)
    relative = metadata.get("native_scgen_root")
    return safe_path(root, relative) if relative else None


def stage(bundle, output, references, calibration=False):
    from benchmarks._runtime import require_compute
    require_compute("Frozen training-input staging")
    bundle, output, references = Path(bundle).resolve(), Path(output).resolve(), Path(references).resolve()
    if output.is_relative_to(bundle) or output.is_relative_to(references):
        raise ValueError("Prepared outputs must be outside every read-only input root.")
    check_output_location(bundle, output)
    check_output_location(references, output)
    if output.exists():
        raise FileExistsError(output)
    manifest = verify(bundle)
    metadata = read_json(bundle / "bundle.json")
    expected_kind = "frozen_calibration" if calibration else "frozen_condition_means"
    if metadata.get("data_kind") != expected_kind:
        raise ValueError(f"Expected {expected_kind}, not {metadata.get('data_kind')}.")
    source = bundle / "inputs"
    name = "manifest.json" if calibration else "provenance.json"
    original = read_json(source / name)
    if original.get("protocol") != PROTOCOLS["calibration" if calibration else "biolord"]:
        raise ValueError("Frozen input protocol does not match the selected experiment.")
    declared = {entry["path"]: entry for entry in manifest["files"]}
    if f"inputs/source_{name}" in declared:
        raise ValueError("Frozen inputs contain a reserved staging metadata filename.")
    for relative, expected in original["artifacts"].items():
        path = safe_path(source, relative)
        if f"inputs/{relative}" not in declared or sha256(path) != expected:
            raise ValueError(f"Frozen input differs from its recorded preparation: {relative}")
    if not (source / "prepare_pass").is_file() or (source / "prepare_pass").stat().st_size:
        raise ValueError("The frozen inputs lack their preparation completion marker.")
    bridged = dict(original)
    if calibration:
        for filename, expected in original["source_sha256"].items():
            _recorded_source(filename, expected)
        bridged["original_source_sha256"] = dict(original["source_sha256"])
        bridged["source_sha256"] = {filename: sha256(safe_path(SCRIPTS, filename))
                                    for filename in original["source_sha256"]}
        bridged["legacy_root"] = str(references / "benchmarks/context_generalization/results/figure4c_calibration_v1")
        scgen = reference_scgen(references)
        if scgen:
            bridged["scgen_root"] = str(scgen)
    else:
        _recorded_source("biolord_cellline_go_v1.py", original["runner_sha256"])
        bridged["original_runner_sha256"] = original["runner_sha256"]
        bridged["runner_sha256"] = sha256(SCRIPTS / "biolord_cellline_go_v1.py")
    bridged["portable_staging"] = {
        "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "original_preparation_sha256": sha256(source / name),
        "arrays_changed": False, "sampling_changed": False,
        "source_binding": "Portable I/O adapters; original preparation evidence retained.",
    }
    root = new_directory(output)
    target = root / "inputs"
    target.mkdir()
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.relative_to(source).as_posix() in {name, "prepare_pass"}:
            continue
        relative = path.relative_to(source)
        if f"inputs/{relative.as_posix()}" not in declared:
            raise ValueError(f"Undeclared input file: {relative}")
        destination = safe_path(target, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if sha256(destination) != declared[f"inputs/{relative.as_posix()}"]["sha256"]:
            raise ValueError(f"Frozen staging changed {relative}")
    shutil.copyfile(source / name, target / ("source_" + name))
    if sha256(target / ("source_" + name)) != declared[f"inputs/{name}"]["sha256"]:
        raise ValueError("Original preparation metadata changed while staging.")
    write_json(target / name, bridged)
    (target / "prepare_pass").touch(exist_ok=False)
    write_json(root / "input_staging.json", bridged["portable_staging"])
    return root
