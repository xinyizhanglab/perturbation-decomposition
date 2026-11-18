"""Strict, non-overwriting artifact I/O."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe artifact path: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Artifact escapes root: {relative}")
    return result


def path_bindings(path, replacement):
    """Map lexical and resolved paths to the same portable configuration value."""
    path = Path(path)
    return {str(path.absolute()): str(replacement), str(path.resolve()): str(replacement)}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(json_safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_output_location(root, destination):
    """Refuse writes into the bundle or any recorded original source project."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    protected = [root]
    provenance = root / "provenance.json"
    if provenance.is_file():
        records = read_json(provenance).get("sources_before", {})
        protected.extend(Path(entry["path"]).resolve() for entry in records.values()
                         if isinstance(entry, dict) and entry.get("path"))
    if any(destination.is_relative_to(path) for path in protected):
        raise ValueError("Output must be outside the artifact bundle and all original source projects")
    return destination


def record_file(root, path, **metadata):
    return {"path": str(Path(path).relative_to(root)),
            "size": Path(path).stat().st_size, "sha256": sha256(path), **metadata}


def verify(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported release schema")
    paths = [entry["path"] for entry in manifest["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate artifact paths")
    for entry in manifest["files"]:
        path = safe_path(root, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing or wrong-sized artifact: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {entry['path']}")
    return manifest
