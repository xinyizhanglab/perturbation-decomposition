"""Paired native-model +30% target calibration; no target-test cells in fitting."""

from portable_paths import (ROOT as PORTABLE_ROOT, PROJECT as PORTABLE_PROJECT, DATA_ROOT, REFERENCE_ROOT, BENCHMARK_ROOT, GO_FILE as PORTABLE_GO_FILE, RESULT_ROOT)

import ast
import hashlib
import json
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd

from common import CELL_LINES, CONTROL_LABEL


ROOT = PORTABLE_ROOT
PROJECT = PORTABLE_PROJECT
BENCH = BENCHMARK_ROOT
OUTPUT = REFERENCE_ROOT / "benchmarks/context_generalization/results/figure4c_calibration_v1"
PROTOCOL = "paired_target_perturbation_calibration_state_memberships_v1"
SETTINGS = ("source_only", "target30")


def read_state_split(path):
    """Read literal lists from the original single-target STATE fewshot section."""
    text = path.read_text()
    sections = re.findall(r'^\[fewshot\.([^\n]+)\]\s*\n(.*?)(?=^\[|\Z)', text,
                          flags=re.MULTILINE | re.DOTALL)
    if len(sections) != 1:
        raise ValueError(f"Expected one target fewshot section: {path}")
    _, body = sections[0]
    result = {}
    for key in ("test", "val"):
        match = re.search(rf"^{key}\s*=\s*(\[[^\n]*\])\s*$", body, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing {key} list in {path}")
        values = ast.literal_eval(match.group(1))
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Invalid {key} membership: {path}")
        result[key] = set(values)
    if result["test"] & result["val"]:
        raise ValueError("Original STATE test/validation memberships overlap")
    return result


def make_split(available, shared, target, original):
    sources = [cell for cell in CELL_LINES if cell != target]
    cohort = available[target] & set.union(*(available[cell] for cell in sources))
    calibration = cohort - original["test"] - original["val"]
    test = cohort & original["test"] & set(shared)
    if not calibration or not test or calibration & test:
        raise ValueError("Empty or leaking target calibration/test membership")
    return {"sources": sources, "cohort": sorted(cohort), "calibration": sorted(calibration),
            "test": sorted(test), "validation_excluded": sorted(cohort & original["val"]),
            "n_original_test": len(original["test"]),
            "n_original_validation": len(original["val"]),
            "calibration_fraction_of_native_cohort": len(calibration) / len(cohort)}


def build_training(samples, target, split, setting):
    if setting not in SETTINGS:
        raise ValueError(setting)
    parts = []
    calibration = set(split["calibration"]) if setting == "target30" else set()
    cohort = set(split["cohort"])
    names = samples[target]["feature_names"].astype(str).tolist()
    for cell in CELL_LINES:
        sample = samples[cell]
        if sample["feature_names"].astype(str).tolist() != names:
            raise ValueError("Source/target feature order differs")
        labels = sample["labels"].astype(str)
        allowed = {CONTROL_LABEL, *calibration} if cell == target else {CONTROL_LABEL, *cohort}
        mask = np.isin(labels, sorted(allowed))
        selected = labels[mask]
        rows = sample["row_indices"][mask]
        role = (np.where(selected == CONTROL_LABEL, "target_control", "target_calibration")
                if cell == target else np.full(len(rows), "source"))
        obs = pd.DataFrame({"perturbation": selected, "cell_state": cell,
                            "role": role, "split": "train", "original_row_index": rows},
                           index=[f"{cell}:{row}" for row in rows])
        parts.append(ad.AnnData(X=sample["X"][mask].copy(), obs=obs,
                               var=pd.DataFrame(index=names)))
    training = ad.concat(parts, join="inner")
    training.obs["perturbation"] = training.obs["perturbation"].astype("category")
    training.obs["cell_state"] = training.obs["cell_state"].astype("category")
    target_mask = training.obs["cell_state"].astype(str).to_numpy() == target
    target_labels = set(training.obs.loc[target_mask, "perturbation"].astype(str))
    if target_labels != {CONTROL_LABEL, *calibration}:
        raise ValueError("Target training membership differs from authorized calibration")
    forbidden = set(split["test"]) | set(split["validation_excluded"])
    if target_labels & forbidden:
        raise ValueError("Target held-out perturbations entered model AnnData")
    for perturbation in split["test"]:
        if not ((training.obs["perturbation"].astype(str) == perturbation) & ~target_mask).any():
            raise ValueError(f"Unobserved source perturbation: {perturbation}")
    controls = np.flatnonzero(training.obs["role"].to_numpy() == "target_control")
    if len(controls) < 2 or training.obs_names.has_duplicates:
        raise ValueError("Missing target controls or duplicate registered cell IDs")
    audit = {"n_training_cells": training.n_obs,
             "n_target_controls": len(controls),
             "n_target_calibration_cells": int(np.sum(training.obs["role"] == "target_calibration")),
             "n_target_calibration_perturbations": len(calibration),
             "target_test_perturbations_in_fitting": 0,
             "target_validation_perturbations_in_fitting": 0,
             "training_membership_sha256": hashlib.sha256("\n".join(training.obs_names).encode()).hexdigest(),
             "test_membership_sha256": hashlib.sha256("\n".join(split["test"]).encode()).hexdigest()}
    return training, controls, audit


def row_correlations(left, right):
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Aligned response matrices required")
    left = np.asarray(left, dtype=float) - np.asarray(left, dtype=float).mean(axis=1, keepdims=True)
    right = np.asarray(right, dtype=float) - np.asarray(right, dtype=float).mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(np.sum(left * right, axis=1), denominator,
                     out=np.zeros(len(left)), where=denominator > 1e-12)


def evaluate(predictions, perturbations, target, anova):
    index = {gene: position for position, gene in enumerate(anova["perturbations"].astype(str))}
    rows = [index[gene] for gene in perturbations]
    cell = list(CELL_LINES).index(target)
    truth = anova["true_deltas"][cell, rows]
    beta = anova["beta"][rows]
    gamma = anova["gamma"][cell, rows]
    if predictions.shape != truth.shape or not np.isfinite(predictions).all():
        raise ValueError("Invalid native calibration predictions")
    reference = truth.mean(axis=0)
    return {"pref": float(row_correlations(predictions - reference, truth - reference).mean()),
            "beta": float(row_correlations(predictions, beta).mean()),
            "gamma": float(row_correlations(predictions, gamma).mean())}


def load_inputs(root):
    manifest = json.loads((root / "inputs/manifest.json").read_text())
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Unexpected calibration input protocol")
    samples = {}
    for cell in CELL_LINES:
        with np.load(root / f"inputs/{cell}.npz", allow_pickle=False) as data:
            samples[cell] = {key: data[key] for key in data.files}
    with np.load(root / "inputs/anova.npz", allow_pickle=False) as data:
        anova = {key: data[key] for key in data.files}
    return manifest, samples, anova
