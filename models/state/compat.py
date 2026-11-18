"""Restore the recorded STATE helper without editing the installed upstream SDK."""

import importlib
from importlib.metadata import version
from pathlib import Path

from reproduction.io import read_json, sha256


def install():
    directory = Path(__file__).with_name("_recorded")
    evidence = read_json(directory / "source.json")
    if version("arc-state") != evidence["arc_state"]:
        raise RuntimeError("Use the locked arc-state environment; no alternative model is substituted.")
    module = importlib.import_module("state.tx.models.utils")
    actual = sha256(module.__file__)
    if actual not in {evidence["stock_utils_sha256"], evidence["recorded_utils_sha256"]}:
        raise RuntimeError("Unrecognized STATE helper revision; refusing to alter its behavior.")
    if sha256(directory / "gpt2.py") != evidence["snapshot_sha256"]:
        raise RuntimeError("Recorded STATE compatibility source changed.")
    if actual != evidence["recorded_utils_sha256"]:
        from models.state._recorded.gpt2 import GPT2BidirectionalModel
        module.GPT2BidirectionalModel = GPT2BidirectionalModel
    return evidence
