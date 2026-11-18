"""Author-aligned binary scGen import; no native model dependency."""

from pathlib import Path

from reproduction.io import read_json
from reproduction.predictions import aggregate_binary


def from_binary_archives(paths, metadata):
    """Validate native completion metadata before complete-cohort aggregation."""
    paths = list(paths)
    if metadata["model"] != "scgen" or metadata["setting"] != "source_only":
        raise ValueError("Only the native binary scGen baseline is supported")
    for path in paths:
        original = read_json(Path(path).parent / "metadata.json")
        if (original["target"] != metadata["target"] or original["seed"] != metadata["repeat"]
                or original["protocol"] != metadata["protocol"]
                or original.get("old_truth_exact_match") is not True):
            raise ValueError("scGen fit metadata differs from the requested experiment")
    return aggregate_binary(paths, metadata)
