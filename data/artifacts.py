"""Install, verify, and package versioned data and prediction bundles."""

import argparse
import json
from pathlib import Path
import shutil

from reproduction.download import download, pack
from reproduction.io import new_directory, record_file, sha256, verify, write_json


DATASETS = {
    "genetic": ["genetic/k562.pkl", "genetic/rpe1.pkl", "genetic/hepg2.pkl",
                "genetic/jurkat.pkl", "genetic/depmap_embeddings.pkl"],
    "norman": ["norman/norman.pkl", "norman/depmap_embeddings.pkl", "norman/split_info.json"],
    "tahoe": ["tahoe/pseudobulk_deltas.pkl", "tahoe/morgan_fingerprints.pkl",
              "tahoe/cc_features.pkl", "tahoe/drug_moa.json", "tahoe/smiles.json"],
}


def required_files(dataset):
    names = DATASETS if dataset == "all" else (dataset,)
    return [path for name in names for path in DATASETS[name]]


def validate_data(root, dataset):
    root = Path(root)
    missing = [p for p in required_files(dataset) if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing processed inputs:\n  " + "\n  ".join(str(root / p) for p in missing) +
            "\nInstall the processed-data bundle; see README.md.")
    return required_files(dataset)


def create_example(destination):
    root = new_directory(destination)
    path = root / "paper/figure1/fig1dsup_identity_variance_results.json"
    write_json(path, {cell: {"pct": value} for cell, value in
                      zip(("k562", "rpe1", "hepg2", "jurkat"), (10, 20, 30, 40))})
    write_json(root / "manifest.json", {
        "schema_version": 1, "version": "example-v1", "profile": "figures",
        "example": True, "description": "Synthetic example, not manuscript results",
        "files": [record_file(root, path)]})
    return root


def prepare_processed(source, destination, dataset="all"):
    from benchmarks._runtime import require_compute
    source, destination = Path(source).resolve(), Path(destination).resolve()
    files = validate_data(source, dataset)
    if destination.is_relative_to(source):
        raise ValueError("The bundle destination must be outside the source data directory")
    require_compute("Processed-data packaging")
    root = new_directory(destination)
    records = {}
    for relative in files + (["hallmark.gmt"] if (source / "hallmark.gmt").is_file() else []):
        path = source / relative
        digest = sha256(path)
        output = root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
        if sha256(path) != digest or sha256(output) != digest:
            raise ValueError(f"Input changed while packaging: {relative}")
        records[relative] = digest
    write_json(root / "provenance.json", {"sources_before": {"processed": {"path": str(source)}},
                                         "source_sha256": records})
    write_json(root / "manifest.json", {
        "schema_version": 1, "version": "processed-v1", "profile": "processed",
        "datasets": list(DATASETS) if dataset == "all" else [dataset],
        "files": [record_file(root, p) for p in sorted(root.rglob("*")) if p.is_file()]})
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("example", help="Create a small synthetic plotting example")
    example.add_argument("--out_dir", required=True, type=Path)
    install = commands.add_parser("install", help="Install a checksummed local or HTTPS bundle")
    install.add_argument("--index", required=True, type=Path)
    install.add_argument("--archive", type=Path, help="Local archive; otherwise use the index's HTTPS URL")
    install.add_argument("--out_dir", required=True, type=Path)
    check = commands.add_parser("verify", help="Verify all declared file sizes and hashes")
    check.add_argument("--root", required=True, type=Path)
    validation = commands.add_parser("validate", help="Check required processed-data files")
    validation.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    validation.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    prepare = commands.add_parser("prepare", help="Package existing processed inputs")
    prepare.add_argument("--source_dir", required=True, type=Path)
    prepare.add_argument("--out_dir", required=True, type=Path)
    prepare.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    figures = commands.add_parser("figures", help="Copy a figure bundle and add existing saved plotting tables")
    figures.add_argument("--source_dir", required=True, type=Path, help="Existing verified figure bundle")
    figures.add_argument("--benchmark_dir", required=True, type=Path, help="Read-only benchmark source project")
    figures.add_argument("--out_dir", required=True, type=Path, help="New standalone bundle destination")
    archive = commands.add_parser("pack", help="Create an archive and matching integrity index")
    archive.add_argument("--root", required=True, type=Path)
    archive.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "example":
            print(create_example(args.out_dir))
        elif args.command == "install":
            print(download(args.index, args.out_dir, args.archive))
        elif args.command == "verify":
            manifest = verify(args.root)
            print(json.dumps({"version": manifest["version"], "profile": manifest["profile"],
                              "verified_files": len(manifest["files"])}, indent=2))
        elif args.command == "validate":
            paths = validate_data(args.data_dir, args.dataset)
            print(f"Found {len(paths)} required processed input files")
        elif args.command == "prepare":
            print(prepare_processed(args.source_dir, args.out_dir, args.dataset))
        elif args.command == "figures":
            from data.figure_inputs import prepare as prepare_figures
            print(prepare_figures(args.source_dir, args.benchmark_dir, args.out_dir))
        elif args.command == "pack":
            print(pack(args.root, args.out))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
