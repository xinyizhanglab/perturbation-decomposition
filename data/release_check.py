"""Check the pinned public release assets without downloading model archives."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


REGISTRY = Path(__file__).resolve().parents[1] / "reproduction/optional_release.json"
METADATA_LIMIT = 2 * 1024 ** 2


def registry():
    result = json.loads(REGISTRY.read_text())
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported optional-release schema.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", result["repository"]):
        raise ValueError("Invalid release repository.")
    if not isinstance(result["tag"], str) or not result["tag"] or len(result["tag"]) > 256:
        raise ValueError("Invalid pinned release tag.")
    if not result.get("assets"):
        raise ValueError("The pinned release has no expected assets.")
    for name, entry in result["assets"].items():
        if Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("Unsafe release asset filename.")
        if type(entry["size"]) is not int or entry["size"] <= 0:
            raise ValueError(f"Invalid expected asset size: {name}")
        checksum = entry.get("sha256")
        if checksum is not None and not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"Invalid expected checksum: {name}")
        if entry["kind"] not in ("archive", "archive_part", "index", "catalog"):
            raise ValueError(f"Invalid asset kind: {name}")
    return result


def release_url(spec):
    return f"https://github.com/{spec['repository']}/releases/tag/{quote(spec['tag'], safe='')}"


def _read(url):
    if urlparse(url).scheme != "https":
        raise ValueError("Release checks require HTTPS.")
    request = Request(url, headers={
        "User-Agent": "perturbation-decomposition-release-check",
        "Accept": "application/vnd.github+json",
    })
    with urlopen(request, timeout=30) as response:
        if urlparse(response.geturl()).scheme != "https":
            raise ValueError("Refusing a non-HTTPS release redirect.")
        content = response.read(METADATA_LIMIT + 1)
    if len(content) > METADATA_LIMIT:
        raise ValueError("Release metadata exceeds the size limit.")
    return content


def check(spec):
    api = f"https://api.github.com/repos/{spec['repository']}"
    try:
        release = json.loads(_read(f"{api}/releases/tags/{quote(spec['tag'], safe='')}"))
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(
                f"No anonymously accessible release at {release_url(spec)}. "
                "The release may still be a draft, private, or not published under this tag. "
                "No uploaded assets were verified."
            ) from exc
        raise
    if release.get("tag_name") != spec["tag"] or release.get("draft") is not False:
        raise ValueError("The response is not the pinned published release.")
    identifier = release.get("id")
    if type(identifier) is not int or identifier <= 0:
        raise ValueError("Invalid release identifier.")
    observed = {}
    for page in range(1, 12):
        rows = json.loads(_read(f"{api}/releases/{identifier}/assets?per_page=100&page={page}"))
        if not isinstance(rows, list):
            raise ValueError("Invalid release asset listing.")
        for row in rows:
            name = row["name"]
            if name in observed:
                raise ValueError(f"Duplicate release asset: {name}")
            observed[name] = row
        if len(rows) < 100:
            break
    else:
        raise ValueError("Release asset pagination exceeded its limit.")
    missing, mismatched, verified_json, verified_digests = [], [], [], []
    for name, expected in spec["assets"].items():
        actual = observed.get(name)
        if actual is None:
            missing.append(name)
            continue
        if actual.get("size") != expected["size"] or actual.get("state") != "uploaded":
            mismatched.append({"asset": name, "reason": "size or upload state differs"})
            continue
        checksum = expected.get("sha256")
        reported = actual.get("digest")
        if checksum and reported:
            if reported != "sha256:" + checksum:
                mismatched.append({"asset": name, "reason": "reported SHA256 differs"})
                continue
            verified_digests.append(name)
        if expected["kind"] in ("index", "catalog"):
            url = (f"https://github.com/{spec['repository']}/releases/download/"
                   f"{quote(spec['tag'], safe='')}/{quote(name, safe='')}")
            content = _read(url)
            if len(content) != expected["size"] or hashlib.sha256(content).hexdigest() != checksum:
                mismatched.append({"asset": name, "reason": "JSON content checksum differs"})
                continue
            verified_json.append(name)
    return {
        "repository": spec["repository"], "tag": spec["tag"], "release": release_url(spec),
        "complete": not missing and not mismatched,
        "expected_assets": len(spec["assets"]), "listed_assets": len(observed),
        "missing": sorted(missing), "mismatched": mismatched,
        "json_checksums_verified": sorted(verified_json),
        "reported_archive_or_index_digests_matched": sorted(verified_digests),
        "additional_assets": sorted(set(observed) - set(spec["assets"])),
        "large_archive_bytes_downloaded": 0,
        "archive_contents_independently_verified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="List expected assets without network access.")
    args = parser.parse_args(argv)
    try:
        spec = registry()
        if args.plan:
            result = {"network_access": False, "release": release_url(spec),
                      "expected_assets": len(spec["assets"]), "assets": spec["assets"]}
        else:
            result = check(spec)
        print(json.dumps(result, indent=2))
        return 0 if args.plan or result["complete"] else 1
    except (OSError, ValueError, KeyError, TypeError, URLError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
