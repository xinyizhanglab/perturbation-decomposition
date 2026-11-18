"""Load saved numerical inputs for supplementary plots."""

import gzip
import json
from pathlib import Path

import numpy as np

from reproduction.io import read_json


SUPPORTED = (
    "fig1_supp_hallmark", "fig1_supp_noise",
    "fig2_supp_deciles", "fig2_supp_scatter",
    "fig3_supp1_slopes", "fig3_supp1_calibration", "fig3_supp1",
)
MAX_SNAPSHOT_BYTES = 32 * 1024 ** 2


def load_plot_data(path):
    path = Path(path)
    if path.suffix != ".gz":
        return read_json(path)
    with gzip.open(path, "rb") as stream:
        content = stream.read(MAX_SNAPSHOT_BYTES + 1)
    if len(content) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Saved plotting input exceeds its size limit.")
    return json.loads(content)


def restore(value):
    if isinstance(value, list):
        return [restore(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind == "nonfinite":
        if value["value"] not in ("nan", "inf", "-inf"):
            raise ValueError("Invalid saved nonfinite value.")
        return float(value["value"])
    if kind in ("array", "scalar"):
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Object arrays are not supported in plotting inputs.")
        if kind == "scalar":
            return np.asarray(restore(value["value"]), dtype=dtype)[()]
        result = np.asarray(restore(value["values"]), dtype=dtype)
        if list(result.shape) != value["shape"]:
            raise ValueError("Saved plotting array shape mismatch.")
        return result
    return {key: restore(item) for key, item in value.items()}


def prepare(identifier, data):
    if identifier not in SUPPORTED:
        raise ValueError(f"Unknown saved supplementary plot: {identifier}")
    if identifier == "fig3_supp1":
        return {}
    values = restore(data)
    if identifier in ("fig2_supp_deciles", "fig2_supp_scatter"):
        return {"curves": values["captured"]}
    if identifier == "fig3_supp1_calibration":
        return {"calibration": values["captured"]}
    return values
