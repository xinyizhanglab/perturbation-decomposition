"""Export unchanged figure assets or run original plotting routines."""

import argparse
import os
from pathlib import Path

from reproduction.catalog import FINAL_REFERENCE_PREFIXES, REFERENCE_PREFIXES, RENDERABLE
from reproduction.figures import coverage, render
from reproduction.io import verify


DEFAULT_RESULTS = Path(os.environ.get("PERTURBATION_RESULTS", "results/precomputed"))


def available(root, mode):
    manifest = verify(root)
    paths = {entry["path"] for entry in manifest["files"]}
    rows = coverage()
    if mode == "original":
        from figures.original import available as original_available
        return original_available(paths)
    if mode in ("rerender", "summary"):
        return [row["id"] for row in rows if row["rerender"] and RENDERABLE[row["id"]][0] in paths]
    final = FINAL_REFERENCE_PREFIXES
    ids = []
    for row in rows:
        prefixes = (final[row["id"]],) if row["id"] in final else tuple(
            "paper/" + p for p in REFERENCE_PREFIXES.get(row["id"], ()))
        if any(path.endswith((".pdf", ".svg")) and path.startswith(prefix)
               for path in paths for prefix in prefixes):
            ids.append(row["id"])
    return ids


def main(argv=None, default_figure=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", "--root", type=Path, default=DEFAULT_RESULTS,
                        help="Installed figure bundle (default: results/precomputed)")
    parser.add_argument("--out_dir", "--out", type=Path,
                        default=Path("results/figures") / (default_figure or "all"))
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--figure", action="append", help="Figure entrypoint name")
    choice.add_argument("--all", action="store_true", help="Render all supported figures in this bundle")
    choice.add_argument("--list", action="store_true", help="List supported figures and modes")
    parser.add_argument("--mode", choices=("reference", "original", "summary", "rerender"),
                        help="Default: unchanged reference assets. Original runs preserved plotting code; "
                             "summary is an explicitly simplified diagnostic layout.")
    parser.add_argument("--font_dir", type=Path,
                        help="Directory containing the original licensed fonts for original mode")
    args = parser.parse_args(argv)
    try:
        if args.mode is None:
            manifest_path = args.results_dir / "manifest.json"
            from reproduction.io import read_json
            example = manifest_path.is_file() and read_json(manifest_path).get("example")
            args.mode = "summary" if example else "reference"
        if args.list:
            declared = available(args.results_dir, args.mode) if (args.results_dir / "manifest.json").is_file() else None
            print(f"{'Figure':36} {'Original':10} {'Reference':11} {'Summary':9} "
                  f"{'Drawing profile':18} In selected mode")
            for row in coverage():
                in_bundle = "—" if declared is None else ("yes" if row["id"] in declared else "no")
                print(f"{row['id']:36} {str(row['original']):10} {str(row['reference']):11} "
                      f"{str(row['summary']):9} {row['rendering_compatibility']:18} {in_bundle}")
            return
        if not (args.results_dir / "manifest.json").is_file():
            parser.error("No figure bundle found. Install one with python -m data.artifacts install; see README.md.")
        selected = args.figure or ([default_figure] if default_figure and not args.all else None)
        ids = available(args.results_dir, args.mode)
        if selected and set(selected) - set(ids):
            parser.error(f"Unavailable in {args.mode} mode: {sorted(set(selected) - set(ids))}. "
                         "Use --list with --mode reference, original, or summary to inspect support.")
        selected = selected or ids
        if not selected:
            parser.error(f"The bundle has no figures supported in {args.mode} mode.")
        output = render(args.results_dir, args.out_dir, selected, args.mode, args.font_dir)
        print(f"{len(selected)} figure entrypoint(s) written to {output}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


def figure_main(identifier):
    main(default_figure=identifier)


if __name__ == "__main__":
    main()
