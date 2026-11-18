"""Checksummed release packing/download; no public endpoint is assumed."""

from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

from .io import check_output_location, new_directory, read_json, safe_path, sha256, verify, write_json


def pack(root, archive, compresslevel=9):
    if not 0 <= compresslevel <= 9:
        raise ValueError("gzip compression level must be in 0..9")
    root, archive = Path(root), Path(archive)
    declared = read_json(root / "manifest.json")
    if sum(entry["size"] for entry in declared["files"]) > 32*1024**2:
        from benchmarks._runtime import require_compute
        require_compute("Large artifact packaging")
    manifest = verify(root)
    check_output_location(root, archive)
    if archive.exists() or archive.with_suffix(archive.suffix + ".json").exists():
        raise FileExistsError("Archive or release index already exists")
    if archive.resolve().is_relative_to(root.resolve()):
        raise ValueError("Archive must be outside its input bundle")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x:gz", compresslevel=compresslevel) as stream:
        for relative in ["manifest.json", *(e["path"] for e in manifest["files"])]:
            stream.add(safe_path(root, relative), arcname=relative, recursive=False)
    index = archive.with_suffix(archive.suffix + ".json")
    write_json(index, {"schema_version": 1, "version": manifest["version"],
                      "archive": archive.name, "url": None, "sha256": sha256(archive),
                      "size": archive.stat().st_size,
                      "manifest_sha256": sha256(root / "manifest.json")})
    return index


def extract(archive, destination, expected_sha256, expected_size=None, manifest_sha256=None):
    archive = Path(archive)
    if sha256(archive) != expected_sha256:
        raise ValueError("Archive checksum mismatch")
    if expected_size is not None and archive.stat().st_size != expected_size:
        raise ValueError("Archive size mismatch")
    destination = new_directory(destination)
    # Validate ALL members first; reject links, devices, traversal and duplicates.
    with tarfile.open(archive, "r:*") as stream:
        members = stream.getmembers()
        seen = set()
        for member in members:
            safe_path(destination, member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError("Archives must contain unique regular files only")
            seen.add(member.name)
        if "manifest.json" not in seen:
            raise ValueError("Missing archive manifest")
        manifest_bytes = stream.extractfile("manifest.json").read()
        if manifest_sha256:
            import hashlib
            if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
                raise ValueError("Manifest trust-root mismatch")
        import json
        manifest = json.loads(manifest_bytes)
        declared = {e["path"]: e for e in manifest["files"]}
        if seen != {"manifest.json", *declared}:
            raise ValueError("Archive contents differ from its manifest")
        for member in members:
            if member.name != "manifest.json" and member.size != declared[member.name]["size"]:
                raise ValueError("Declared uncompressed size mismatch")
            target = safe_path(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink)
    verify(destination)
    return destination


def download(index, destination, archive=None):
    index = read_json(index)  # A reviewed, local, version-pinned trust root.
    if index.get("schema_version") != 1:
        raise ValueError("Unsupported release index")
    if archive is not None:
        return extract(archive, destination, index["sha256"], index["size"],
                       index["manifest_sha256"])
    url = index.get("url")
    if not url:
        raise ValueError("This index has no download URL. Supply --archive with the matching .tar.gz file.")
    if urlparse(url).scheme != "https":
        raise ValueError("Public release downloads require HTTPS")
    # A failed download leaves the temporary archive, not a partially trusted bundle.
    with tempfile.NamedTemporaryFile(prefix="perturbation-release-", suffix=".tar.gz", delete=False) as sink:
        path = Path(sink.name)
        with urlopen(url, timeout=60) as source:
            if urlparse(source.geturl()).scheme != "https":
                raise ValueError("Refusing a non-HTTPS redirect")
            received = 0
            while block := source.read(1024 * 1024):
                received += len(block)
                if received > index["size"]:
                    raise ValueError("Download exceeds declared archive size")
                sink.write(block)
    return extract(path, destination, index["sha256"], index["size"], index["manifest_sha256"])
