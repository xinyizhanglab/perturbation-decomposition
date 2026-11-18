"""Final BioLORD prediction import; native training uses the model-specific environment."""

from pathlib import Path

from reproduction.io import read_json
from reproduction.predictions import native_archive


def from_recovered_archive(path, metadata):
    """Reject the known-invalid original export, then validate full gene axes."""
    path = Path(path)
    original = read_json(path.parent / "metrics.json")
    recovery = original.get("prediction_recovery", {})
    if recovery.get("retrained") is not False or recovery.get("weights_unchanged") is not True:
        raise ValueError("Only the completed, unchanged-weight BioLORD recovery is authoritative")
    if (metadata["model"] != "biolord" or original["target"] != metadata["target"]
            or original["seed"] != metadata["repeat"] or original["protocol"] != metadata["protocol"]):
        raise ValueError("BioLORD fit metadata differs from the requested experiment")
    return native_archive(path, metadata)
