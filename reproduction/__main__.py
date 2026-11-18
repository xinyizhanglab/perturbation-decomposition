"""Single, discoverable CLI; defaults never train, submit jobs, or overwrite."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build a NEW local artifact candidate from read-only sources")
    for source in ("agop", "benchmark", "release"):
        build.add_argument(f"--{source}-root", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--profile", choices=("figures", "analysis", "full"), default="figures")
    check = commands.add_parser("verify")
    check.add_argument("--root", required=True, type=Path)
    commands.add_parser("list", help="List original plotting, reference export, and summary coverage")
    plots = commands.add_parser("figures")
    plots.add_argument("--root", required=True, type=Path)
    plots.add_argument("--out", required=True, type=Path)
    selection = plots.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="All supported figures IN THE SELECTED MODE")
    selection.add_argument("--figure", action="append")
    plots.add_argument("--mode", choices=("reference", "original", "summary", "rerender"),
                       default="reference")
    plots.add_argument("--font_dir", type=Path)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--root", required=True, type=Path)
    analysis.add_argument("--out", required=True, type=Path)
    analysis.add_argument("--analysis", choices=("matching", "genetic", "component-scores"), default="matching")
    archive = commands.add_parser("pack")
    archive.add_argument("--root", required=True, type=Path)
    archive.add_argument("--out", required=True, type=Path)
    fetch = commands.add_parser("download")
    fetch.add_argument("--index", type=Path, required=True, help="Reviewed local release-index JSON")
    fetch.add_argument("--archive", type=Path, help="Local archive instead of public download")
    fetch.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            from .release import build
            print(build({key: getattr(args, key + "_root") for key in
                         ("agop", "benchmark", "release")}, args.out, args.profile))
        elif args.command == "verify":
            from .io import verify
            result = verify(args.root)
            print(f"VERIFIED {result['version']}: {len(result['files'])} files")
        elif args.command == "list":
            from .figures import coverage
            print(json.dumps(coverage(), indent=2))
        elif args.command == "figures":
            from .figures import render
            print(render(args.root, args.out, args.figure, args.mode, args.font_dir))
        elif args.command == "analyze":
            from .analyze import component_scores, genetic, matching
            function = {"matching": matching, "genetic": genetic,
                        "component-scores": component_scores}[args.analysis]
            print(function(args.root, args.out))
        elif args.command == "pack":
            from .download import pack
            print(pack(args.root, args.out))
        elif args.command == "download":
            from .download import download
            print(download(args.index, args.out, args.archive))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
