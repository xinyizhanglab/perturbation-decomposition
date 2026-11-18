"""Check installation and dataset configuration without loading single-cell data."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

import yaml


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def validate_config(path):
    config = yaml.load(Path(path).read_text(), Loader=UniqueLoader)
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), dict):
        raise ValueError("Configuration must contain a datasets mapping")
    for key, entry in config["datasets"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"Dataset {key!r} needs an indented mapping with a name")
        if not isinstance(entry.get("n_hvgs"), int) or entry["n_hvgs"] < 1:
            raise ValueError(f"Dataset {key!r} needs a positive n_hvgs")
    if not isinstance(config.get("models"), dict):
        raise ValueError("Configuration must contain model reference defaults")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--require", choices=("core", "analysis"), default="core")
    args = parser.parse_args(argv)
    groups = {"core": ("numpy", "matplotlib", "PyYAML"),
              "analysis": ("scipy", "scikit-learn", "pandas", "anndata")}
    packages = {}
    required_missing = []
    for group, names in groups.items():
        for name in names:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
                if group == "core" or args.require == "analysis":
                    required_missing.append(name)
    try:
        config = validate_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    print(json.dumps({"python": sys.version.split()[0], "packages": packages,
                      "datasets": list(config["datasets"]),
                      "configuration": str(args.config)}, indent=2))
    if required_missing:
        command = "uv sync --locked" + (" --extra analysis" if args.require == "analysis" else "")
        parser.exit(2, f"Missing packages: {', '.join(required_missing)}. Run {command}.\n")


if __name__ == "__main__":
    main()
