"""Explicit read-only input roots and separate writable experiment outputs."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]


def required(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing {name}; use python -m models.biolord.train or python -m models.scgen.train")
    result = Path(value).expanduser().resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {result}")
    return result


REFERENCE_ROOT = required("PD_REFERENCE_ROOT")
DATA_ROOT = required("PD_RAW_DATA")
BENCHMARK_ROOT = DATA_ROOT.parent
GO_FILE = Path(os.environ.get("PD_GO_FILE", REFERENCE_ROOT / "data/biolord_cellline_go_v1/gene2go.pkl"))
RESULT_ROOT = Path(os.environ.get("PD_RESULT_ROOT", PROJECT / "results/native_experiments")).resolve()
for protected in (REFERENCE_ROOT, DATA_ROOT, BENCHMARK_ROOT):
    if RESULT_ROOT.is_relative_to(protected) or PROJECT.is_relative_to(protected):
        raise ValueError("Experiment output/code workspace must be outside read-only input projects")
