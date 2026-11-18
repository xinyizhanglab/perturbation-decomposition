"""Rendering evidence and appearance checks for preserved figure assets."""

import argparse
import base64
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import re
from unittest.mock import patch
import xml.etree.ElementTree as ET

from reproduction.catalog import FINAL_REFERENCE_PREFIXES as FINAL
from reproduction.catalog import REFERENCE_PREFIXES, figure_ids
from reproduction.io import read_json, safe_path, sha256, verify


PROFILE_PATH = Path(__file__).with_name("_original") / "rendering_profiles.json"
# The source routine's filename differs from the archived asset's filename.
OUTPUT_ALIASES = {"fig1_dsup": {"fig1_identity_variance.svg": "fig1dsup_identity_variance.svg"}}
NUMBER = re.compile(r"(?<![A-Za-z#])[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
FONT_PREFIXES = {
    "DejaVuSans": "DejaVu Sans", "DejaVuSans-Bold": "DejaVu Sans",
    "DejaVuSans-Oblique": "DejaVu Sans", "DejaVuSans-BoldOblique": "DejaVu Sans",
    "ArialMT": "Arial", "Arial-BoldMT": "Arial", "Arial-ItalicMT": "Arial",
    "Arial-BoldItalicMT": "Arial",
}


def reference_entries(identifier, manifest):
    prefixes = ((FINAL[identifier],) if identifier in FINAL else
                tuple("paper/" + p for p in REFERENCE_PREFIXES.get(identifier, ())))
    return [entry for entry in manifest["files"]
            if entry["path"].endswith((".svg", ".pdf"))
            and any(entry["path"].startswith(prefix) for prefix in prefixes)]


def _number(value):
    def normalized(match):
        number = round(float(match.group()), 4)
        return format(0.0 if number == 0 else number, ".4f").rstrip("0").rstrip(".")
    return "".join(part if index % 2 else NUMBER.sub(normalized, part)
                   for index, part in enumerate(re.split(r"(#[0-9a-fA-F]{3,8}\b)", value)))


def drawing_document(path):
    """Canonical SVG drawing, ignoring timestamps and generated element IDs.

    Coordinates are compared at 0.0001 SVG unit precision. Embedded PNGs are
    compared by decoded pixels, not compression bytes. No plotted data or
    original asset is modified.
    """
    from PIL import Image
    root = ET.parse(path).getroot()

    def drawing_nodes(element):
        if element.tag.rsplit("}", 1)[-1] == "metadata":
            return
        yield element
        for child in element:
            yield from drawing_nodes(child)

    ids = {element.attrib["id"]: f"element_{index}"
           for index, element in enumerate(drawing_nodes(root)) if "id" in element.attrib}

    def visit(element):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "metadata":
            return None
        attributes = {}
        for key, value in sorted(element.attrib.items()):
            key = key.rsplit("}", 1)[-1]
            if key == "id":
                attributes[key] = ids[value]
                continue
            if key == "href" and value.startswith("data:image/png;base64,"):
                image = Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGBA")
                attributes[key] = {"pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                                   "size": list(image.size)}
                continue
            if value.startswith("#") and value[1:] in ids:
                value = "#" + ids[value[1:]]
            value = re.sub(r"url\(#([^)]+)\)",
                           lambda match: "url(#" + ids.get(match[1], match[1]) + ")", value)
            if key == "style":
                value = ";".join(sorted(item.strip() for item in value.split(";") if item.strip()))
            attributes[key] = _number(value)
        children = [child for child in (visit(item) for item in element) if child is not None]
        text = element.text.strip() if element.text and element.text.strip() else None
        return [tag, attributes, text, children]
    return visit(root)


def drawing_sha256(path):
    return hashlib.sha256(json.dumps(drawing_document(path), sort_keys=True).encode()).hexdigest()


def svg_evidence(path):
    path = Path(path)
    text = path.read_text()
    root = ET.fromstring(text)
    glyph_fonts = sorted(set(re.findall(r'id="([A-Za-z][A-Za-z0-9_-]*?)-[0-9a-f]+"', text)))
    css_families = sorted(set(
        family.strip().strip("'\"")
        for value in re.findall(r"font-family:\s*([^;\"<>]+)", text)
        for family in value.split(",") if family.strip()
    ))
    families = set(css_families)
    unknown = []
    for font in glyph_fonts:
        if font in FONT_PREFIXES:
            families.add(FONT_PREFIXES[font])
        else:
            unknown.append(font)
    tags = Counter(element.tag.rsplit("}", 1)[-1] for element in root.iter())
    versions = sorted(set(re.findall(r"Matplotlib v([\d.]+)", text)))
    styles = sorted(set(element.attrib["style"] for element in root.iter()
                        if "style" in element.attrib))
    return {
        "sha256": sha256(path), "drawing_sha256": drawing_sha256(path),
        "width": root.get("width"), "height": root.get("height"), "viewBox": root.get("viewBox"),
        "font_families": sorted(families), "glyph_fonts": glyph_fonts,
        "unrecognized_glyph_fonts": unknown, "matplotlib_versions": versions,
        "svg_fonttype": "none" if tags["text"] else "path",
        "elements": dict(sorted(tags.items())),
        "styles_sha256": hashlib.sha256(json.dumps(styles).encode()).hexdigest(),
    }


def pdf_fonttype(path):
    content = Path(path).read_bytes()
    type3 = bool(re.search(rb"/Subtype\s*/Type3\b", content))
    truetype = bool(re.search(rb"/Subtype\s*/(?:CIDFontType2|TrueType)\b", content))
    return 3 if type3 and not truetype else 42 if truetype and not type3 else None


def pdf_evidence(path):
    content = Path(path).read_bytes()
    boxes = sorted(set(" ".join(_number(match.decode()).split()) for match in
                       re.findall(rb"/MediaBox\s*\[([^\]]+)\]", content)))
    return {"sha256": sha256(path), "pdf_fonttype": pdf_fonttype(path), "media_boxes": boxes}


def build_profiles(root, recipes):
    root = Path(root)
    manifest = verify(root)
    entries = {entry["path"]: entry for entry in manifest["files"]}
    figures = {}
    for identifier in figure_ids():
        references = reference_entries(identifier, manifest)
        assets = {}
        for entry in references:
            path = safe_path(root, entry["path"])
            assets[entry["path"]] = svg_evidence(path) if path.suffix == ".svg" else pdf_evidence(path)
        svgs = {path: record for path, record in assets.items() if path.endswith(".svg")}
        families = {family for record in svgs.values() for family in record["font_families"]}
        fonttypes = {record["svg_fonttype"] for record in svgs.values()}
        pdftypes = {record["pdf_fonttype"] for path, record in assets.items() if path.endswith(".pdf")}
        unknown_fonts = any(record["unrecognized_glyph_fonts"] for record in svgs.values())
        overrides = {}
        if len(families) == 1 and len(fonttypes) == 1 and not unknown_fonts:
            family = next(iter(families))
            overrides = {"font.family": [family], "font.sans-serif": [family],
                         "svg.fonttype": next(iter(fonttypes))}
            if len(pdftypes) == 1 and None not in pdftypes:
                overrides["pdf.fonttype"] = next(iter(pdftypes))
        recipe = recipes.get(identifier)
        if recipe and recipe.get("font_check_family") and svgs and not unknown_fonts:
            overrides = {}
            if len(fonttypes) == 1:
                overrides["svg.fonttype"] = next(iter(fonttypes))
            if len(pdftypes) == 1 and None not in pdftypes:
                overrides["pdf.fonttype"] = next(iter(pdftypes))
        figures[identifier] = {
            "status": ("profiled" if recipe and overrides else
                       "reference_only" if assets else "no_saved_reference"),
            "rc_overrides": overrides, "assets": assets,
            "output_aliases": OUTPUT_ALIASES.get(identifier, {}),
            "input": recipe["input"] if recipe else None,
            "input_sha256": entries[recipe["input"]]["sha256"]
                if recipe and recipe["input"] in entries else None,
            "compatibility": "unverified" if recipe and overrides else "not_applicable",
        }
        if recipe:
            for key in ("font_check_family", "isolate_font_families", "packaged_input"):
                if key in recipe:
                    figures[identifier][key] = recipe[key]
            if recipe.get("packaged_input"):
                from figures.original import SOURCES
                figures[identifier]["packaged_input_sha256"] = sha256(
                    safe_path(SOURCES, recipe["packaged_input"]))
    return {
        "schema_version": 1,
        "normalization": {"version": 1, "coordinate_decimals": 4,
                          "ignored": ["SVG export metadata", "generated element IDs", "PNG compression"]},
        "figures": figures,
    }


def profile(identifier, root=None, require_verified=True):
    registry = read_json(PROFILE_PATH)
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported figure rendering profile schema")
    entry = registry["figures"][identifier]
    if require_verified and entry["compatibility"] != "matched":
        raise ValueError(f"{identifier}: the original routine is not rendering-compatible with its saved "
                         "figure profile. Use --mode reference for the unchanged figure.")
    if entry.get("packaged_input"):
        from figures.original import SOURCES
        if sha256(safe_path(SOURCES, entry["packaged_input"])) != entry["packaged_input_sha256"]:
            raise ValueError(f"{identifier}: packaged saved plotting input changed.")
    if root is not None:
        manifest = verify(root)
        files = {record["path"]: record["sha256"] for record in manifest["files"]}
        for path, expected in entry["assets"].items():
            if files.get(path) != expected["sha256"]:
                raise ValueError(f"{identifier}: saved rendering reference changed or is missing: {path}")
        if entry["input"] and files.get(entry["input"]) != entry["input_sha256"]:
            raise ValueError(f"{identifier}: saved plotting input differs from the rendering profile")
    return entry


@contextmanager
def saved_font_families(families):
    """Restore an observed PDF font without rewriting a saved SVG fallback list."""
    if not families:
        yield
        return
    from matplotlib import font_manager
    manager = font_manager.fontManager
    fonts = [font for font in manager.ttflist if font.name in families]
    if set(families) - {font.name for font in fonts}:
        raise ValueError("A recorded font family is unavailable; no replacement font was selected.")
    manager._findfont_cached.cache_clear()
    try:
        with patch.object(manager, "ttflist", fonts):
            yield
    finally:
        manager._findfont_cached.cache_clear()


@contextmanager
def saved_settings(overrides):
    """Enforce artifact-observed settings even when source code requests a fallback font."""
    import matplotlib
    original_setitem = type(matplotlib.rcParams).__setitem__

    def setitem(instance, key, value):
        return original_setitem(instance, key, copy.deepcopy(overrides.get(key, value)))

    with matplotlib.rc_context():
        with patch.object(type(matplotlib.rcParams), "__setitem__", setitem):
            matplotlib.rcParams.update(overrides)
            yield


def compare_outputs(directory, entry):
    directory = Path(directory)
    expected = {Path(path).name: (path, record) for path, record in entry["assets"].items()
                if path.endswith(".svg")}
    actual = {path.name: path for path in directory.glob("*.svg")}
    aliases = entry["output_aliases"]
    mapped = {aliases.get(name, name): path for name, path in actual.items()}
    if set(mapped) != set(expected):
        raise ValueError("Original routine's output panel set differs from its saved figure assets")
    records = []
    for name, path in sorted(mapped.items()):
        relative, reference = expected[name]
        observed = svg_evidence(path)
        match = observed["drawing_sha256"] == reference["drawing_sha256"]
        records.append({"output": path.name, "reference": relative, "matched": match,
                        "format": "svg",
                        "expected_viewBox": reference["viewBox"], "actual_viewBox": observed["viewBox"],
                        "expected_drawing_sha256": reference["drawing_sha256"],
                        "actual_drawing_sha256": observed["drawing_sha256"]})
    for relative, reference in entry["assets"].items():
        if not relative.endswith(".pdf"):
            continue
        reference_name = Path(relative).name
        source_svg = next((source for source, target in aliases.items()
                           if Path(target).with_suffix(".pdf").name == reference_name), None)
        output_name = Path(source_svg).with_suffix(".pdf").name if source_svg else reference_name
        path = directory / output_name
        if not path.is_file():
            raise ValueError(f"Original routine did not export its saved PDF panel: {output_name}")
        observed = pdf_evidence(path)
        match = (observed["pdf_fonttype"] == reference["pdf_fonttype"]
                 and observed["media_boxes"] == reference["media_boxes"])
        records.append({"output": path.name, "reference": relative, "matched": match,
                        "format": "pdf", "expected_fonttype": reference["pdf_fonttype"],
                        "actual_fonttype": observed["pdf_fonttype"],
                        "expected_media_boxes": reference["media_boxes"],
                        "actual_media_boxes": observed["media_boxes"]})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=figure_ids())
    parser.add_argument("--results_dir", type=Path, help="Check the saved asset/input hashes in this bundle")
    args = parser.parse_args()
    try:
        if args.figure:
            print(json.dumps(profile(args.figure, args.results_dir, require_verified=False), indent=2))
        else:
            figures = read_json(PROFILE_PATH)["figures"]
            for identifier, entry in sorted(figures.items()):
                families = entry["rc_overrides"].get("font.family", [])
                print(f"{identifier:36} {','.join(families) or 'unavailable':16} "
                      f"{entry['status']:18} {entry['compatibility']}")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
