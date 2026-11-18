"""New Figure 4c inputs and BioLORD fits; preserve every earlier artifact.

Reuse Figure 4c's sampled cells, STATE calibration memberships, full-pseudobulk
ANOVA truth and scoring. Use the corrected GO + cell-line BioLORD formulation.
Author-aligned binary scGen predictions are rescored, never retrained here.
"""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch

from biolord_cellline_go_v1 import (
    AUTHOR_COMMIT, GO_KEY, MODULE, PLAN, PROJECT, ROOT, TRAIN, build_model,
    make_pseudobulk, registered_control_dataset, sha256, state_sample_digest, write_json,
)
from calibration_v1 import OUTPUT as LEGACY, SETTINGS, evaluate, load_inputs
from common import CELL_LINES, CONTROL_LABEL
from recover_biolord_cellline_go_export_v2 import expression_mean
from scgen_author_aligned_v1 import OUTPUT as SCGEN, read_tasks


OUTPUT = ROOT / "results/figure4c_author_aligned_v2"
GO_INPUTS = ROOT / "results/biolord/three_to_one_cellline_go_author_v1/inputs"
CHECKPOINTS = PROJECT / "checkpoints/biolord_fig4c_author_aligned_v2"
PROTOCOL = "fig4c_go_cellline_biolord_binary_scgen_baseline_only_v2"
STEM = "fig4c_calibration_author_aligned_v2"
NA_REASON = "Not applicable: binary-per-perturbation scGen cannot ingest other target perturbations."
SOURCE_FILES = (
    "figure4c_author_v2.py", "report_figure4c_author_v2.py", "calibration_v1.py", "common.py",
    "biolord_cellline_go_v1.py", "recover_biolord_cellline_go_export_v2.py",
    "run_biolord_corrected_v2.py", "scgen_author_aligned_v1.py", "report_scgen_author_aligned_v1.py",
)


def task_spec(task):
    if not 0 <= task < 40:
        raise ValueError("BioLORD array task must be in 0..39")
    return CELL_LINES[task // 10], (task % 10) // 2, SETTINGS[task % 2]


def load_descriptors(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(zip(data["perturbations"].astype(str), data["descriptors"].copy()))


def build_pair(samples, target, split, descriptors):
    """Identical source/control means and inner validation in the paired fits."""
    cohort, calibration = set(split["cohort"]), set(split["calibration"])
    forbidden = set(split["test"]) | set(split["validation_excluded"])
    if calibration & forbidden or not calibration <= cohort or not set(split["test"]) <= cohort:
        raise ValueError("Invalid or leaking calibration/test membership")
    if any(CONTROL_LABEL in split[key] for key in ("cohort", "calibration", "test", "validation_excluded")):
        raise ValueError("Control cannot be a perturbation split member")
    names = samples[target]["feature_names"].astype(str).tolist()
    sources, membership = {}, {}
    for cell in CELL_LINES:
        sample = samples[cell]
        if sample["feature_names"].astype(str).tolist() != names:
            raise ValueError("Cross-cell-line feature order mismatch")
        labels = sample["labels"].astype(str)
        allowed = {CONTROL_LABEL, *calibration} if cell == target else {CONTROL_LABEL, *cohort}
        groups, rows = {}, {}
        for perturbation in sorted(allowed & set(labels)):
            selected = np.flatnonzero(labels == perturbation)
            groups[perturbation] = sample["X"][selected]
            rows[perturbation] = sample["row_indices"][selected].tolist()
        membership[cell] = rows
        if cell != target:
            sources[cell] = groups
        else:
            target_groups = groups
    if set(target_groups) != {CONTROL_LABEL, *calibration}:
        raise ValueError("Missing authorized target calibration/control cells")
    baseline = make_pseudobulk(sources, target_groups[CONTROL_LABEL], names, target, descriptors)
    new_labels = sorted(calibration)
    obs = pd.DataFrame({"cell_line": target, "perturbation": new_labels,
                        "role": "target_calibration", "split": "train",
                        "n_cells": [len(target_groups[p]) for p in new_labels]},
                       index=[f"{target}:{p}" for p in new_labels])
    addition = ad.AnnData(np.stack([target_groups[p].mean(0) for p in new_labels]).astype(np.float32),
                         obs=obs, var=baseline.var.copy())
    addition.obsm[GO_KEY] = np.stack([descriptors[p] for p in new_labels]).astype(np.float32)
    calibrated = ad.concat([baseline, addition], join="inner", merge="same")
    calibrated.obs["cell_line"] = calibrated.obs["cell_line"].astype("category")
    pair = dict(zip(SETTINGS, (baseline, calibrated)))
    for setting, data in pair.items():
        assert_training(data, target, split, setting, descriptors)
    if not np.array_equal(baseline.X, calibrated[baseline.obs_names].X):
        raise ValueError("Paired source/control means changed")
    if baseline.obs["split"].tolist() != calibrated.obs.loc[baseline.obs_names, "split"].tolist():
        raise ValueError("Paired source-only validation changed")
    return pair, membership


def assert_training(data, target, split, setting, descriptors):
    if setting not in SETTINGS or data.obs_names.has_duplicates:
        raise ValueError("Invalid setting or duplicate condition means")
    target_mask = data.obs["cell_line"].astype(str).to_numpy() == target
    target_perts = data.obs.loc[target_mask, "perturbation"].astype(str)
    allowed = {CONTROL_LABEL, *(split["calibration"] if setting == "target30" else [])}
    if set(target_perts) != allowed or len(target_perts) != len(allowed):
        raise ValueError("Target fitting includes forbidden/missing condition means")
    if set(target_perts) & (set(split["test"]) | set(split["validation_excluded"])):
        raise ValueError("Held-out target perturbations entered fitting")
    if not (data.obs.loc[target_mask, "split"] == "train").all():
        raise ValueError("Target controls/calibration unexpectedly in inner validation")
    validation = data.obs["split"].astype(str) == "validation"
    if not validation.any() or not (data.obs.loc[validation, "role"] == "source").all():
        raise ValueError("Inner validation must contain source condition means only")
    source_train = ~target_mask & (data.obs["split"].astype(str).to_numpy() == "train")
    if not set(split["test"]) <= set(data.obs.loc[source_train, "perturbation"].astype(str)):
        raise ValueError("Held-out perturbation lacks optimizer-training source evidence")
    expected_go = np.stack([descriptors[p] for p in data.obs["perturbation"].astype(str)])
    if not np.array_equal(data.obsm[GO_KEY], expected_go):
        raise ValueError("GO attributes changed")
    if not np.isfinite(data.X).all() or not np.isfinite(expected_go).all():
        raise ValueError("Nonfinite model input")


def verify_inputs(root, task=None, full=False):
    if not (root / "inputs/prepare_pass").is_file():
        raise ValueError("Preparation not complete")
    manifest = json.loads((root / "inputs/manifest.json").read_text())
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Wrong Figure 4c protocol")
    for name, checksum in manifest["source_sha256"].items():
        if sha256(Path(__file__).parent / name) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    mandatory = {"legacy_manifest.json", "original_figure4c.json", "anova.npz", "go_features.npz", "scgen_manifest.jsonl"}
    if task is not None:
        target, seed, setting = task_spec(task)
        mandatory |= {f"{target}/seed{seed}/{setting}.h5ad", f"{target}/seed{seed}/membership.json"}
    for name, checksum in manifest["artifacts"].items():
        if (full or name in mandatory) and sha256(root / "inputs" / name) != checksum:
            raise ValueError(f"Frozen input changed: {name}")
    return manifest


def prepare(root=OUTPUT):
    if root.exists():
        raise FileExistsError(f"Refusing overwrite: {root}")
    manifest, samples, anova = load_inputs(LEGACY)
    # Preserve the original all-cell ANOVA and exact STATE memberships.
    if sha256(LEGACY / "inputs/original_figure4c.json") != manifest["baseline_sha256"]:
        raise ValueError("Legacy Figure 4c baseline checksum mismatch")
    go_provenance = json.loads((GO_INPUTS / "provenance.json").read_text())
    if sha256(GO_INPUTS / "go_features.npz") != go_provenance["artifacts"]["go_features.npz"]:
        raise ValueError("Original GO features changed")
    descriptors = load_descriptors(GO_INPUTS / "go_features.npz")
    tasks = read_tasks(SCGEN / "inputs/manifest.jsonl")
    task_lookup = {(task["target"], task["perturbation"]): index for index, task in enumerate(tasks)}
    for target in CELL_LINES:
        for seed in range(5):
            split = manifest["splits"][target][str(seed)]
            missing = set(split["cohort"]) - set(descriptors)
            if missing:
                raise ValueError(f"Missing GO descriptors, never silently drop: {sorted(missing)}")
            if any((target, p) not in task_lookup for p in split["test"]):
                raise ValueError("Figure 4c test perturbation lacks a completed binary scGen task")
            for p in split["test"]:
                if not (SCGEN / f"seed{seed}/{target}/{p}/metadata.json").is_file():
                    raise ValueError(f"Missing completed scGen fit {target}/{seed}/{p}")
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    frozen = {"legacy_manifest.json": LEGACY / "inputs/manifest.json",
              "original_figure4c.json": LEGACY / "inputs/original_figure4c.json",
              "anova.npz": LEGACY / "inputs/anova.npz",
              "go_features.npz": GO_INPUTS / "go_features.npz",
              "scgen_manifest.jsonl": SCGEN / "inputs/manifest.jsonl"}
    for name, source in frozen.items():
        shutil.copyfile(source, inputs / name)
    artifacts = {name: sha256(inputs / name) for name in frozen}
    for target in CELL_LINES:
        for seed in range(5):
            split = manifest["splits"][target][str(seed)]
            pair, membership = build_pair(samples, target, split, descriptors)
            directory = inputs / target / f"seed{seed}"
            directory.mkdir(parents=True)
            for setting, data in pair.items():
                data.write_h5ad(directory / f"{setting}.h5ad")
                name = f"{target}/seed{seed}/{setting}.h5ad"
                artifacts[name] = sha256(inputs / name)
            write_json(directory / "membership.json", membership)
            name = f"{target}/seed{seed}/membership.json"
            artifacts[name] = sha256(inputs / name)
            print(f"PREPARED {target} seed{seed}: {len(split['test'])} test; "
                  f"{len(split['calibration'])} calibration", flush=True)
    write_json(inputs / "manifest.json", {
        "protocol": PROTOCOL, "legacy_root": str(LEGACY), "scgen_root": str(SCGEN),
        "baseline_sha256": manifest["baseline_sha256"], "splits": manifest["splits"],
        "sampling": {"biolord": "unchanged Figure 4c samples, <=64 cells/group, seed42",
                     "scgen": "unchanged Figure 3 binary fits, <=128 cells/group"},
        "author_commit": AUTHOR_COMMIT, "module": MODULE, "plan": PLAN, "training": TRAIN,
        "go_coverage": go_provenance["go_coverage"], "scgen_target30": NA_REASON,
        "artifacts": artifacts,
        "source_sha256": {name: sha256(Path(__file__).parent / name) for name in SOURCE_FILES},
    })
    (inputs / "prepare_pass").touch(exist_ok=False)
    verify_inputs(root, full=True)
    print(f"PREPARE_PASS {root}", flush=True)


def fit(root, task, log_dir, checkpoint_root=CHECKPOINTS):
    if not torch.cuda.is_available():
        raise RuntimeError("BioLORD training requires a SLURM GPU allocation")
    manifest = verify_inputs(root, task=task)
    target, seed, setting = task_spec(task)
    split = manifest["splits"][target][str(seed)]
    output = root / f"biolord/{target}/seed{seed}/{setting}"
    checkpoint = checkpoint_root / target / f"seed{seed}/{setting}"
    if output.exists() or checkpoint.exists():
        raise FileExistsError(f"Refusing overwrite: {output} or {checkpoint}")
    data = ad.read_h5ad(root / f"inputs/{target}/seed{seed}/{setting}.h5ad")
    descriptors = load_descriptors(root / "inputs/go_features.npz")
    assert_training(data, target, split, setting, descriptors)
    model = build_model(data, seed)
    initial = state_sample_digest(model)
    started = time.monotonic()
    model.train(accelerator="gpu", device=1, plan_kwargs=dict(PLAN), logger=False,
                default_root_dir=str(log_dir), num_workers=0, **TRAIN)
    trained = state_sample_digest(model)
    if trained == initial:
        raise ValueError("Model parameters did not change")
    controls = np.flatnonzero((data.obs["cell_line"].astype(str).to_numpy() == target)
                              & (data.obs["perturbation"].astype(str).to_numpy() == CONTROL_LABEL))
    if len(controls) != 1:
        raise ValueError("Expected one target-control condition mean")
    native_controls = registered_control_dataset(model, controls)
    baseline = expression_mean(model, native_controls, descriptors[CONTROL_LABEL])
    expression = np.stack([expression_mean(model, native_controls, descriptors[p]) for p in split["test"]])
    predictions = (expression - baseline).astype(np.float32)
    if state_sample_digest(model) != trained:
        raise ValueError("Inference changed trained weights")
    with np.load(root / "inputs/anova.npz", allow_pickle=False) as file:
        anova = {key: file[key] for key in file.files}
    metrics = evaluate(predictions, split["test"], target, anova)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint), overwrite=False, save_anndata=False)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", predictions=predictions,
                        predicted_expression_means=expression, reconstructed_control_mean=baseline,
                        test_perts=np.asarray(split["test"]), feature_names=data.var_names.to_numpy().astype(str))
    np.savez_compressed(output / "membership.npz", condition_ids=data.obs_names.to_numpy().astype(str),
                        optimizer_indices=np.asarray(model.train_indices),
                        validation_indices=np.asarray(model.valid_indices), test_indices=np.asarray(model.test_indices))
    expected_train = np.flatnonzero(data.obs["split"].to_numpy() == "train")
    expected_val = np.flatnonzero(data.obs["split"].to_numpy() == "validation")
    if not np.array_equal(np.sort(model.train_indices), expected_train) or not np.array_equal(np.sort(model.valid_indices), expected_val):
        raise ValueError("Native optimizer/validation membership differs from audited inputs")
    write_json(output / "metrics.json", {
        "protocol": PROTOCOL, "model": "biolord", "target": target, "seed": seed, "setting": setting,
        "task": task, "seconds": time.monotonic() - started, "epochs_completed": int(model.training_plan.current_epoch),
        "n_test_perturbations": len(split["test"]), "n_target_calibration_means": (
            len(split["calibration"]) if setting == "target30" else 0),
        "initial_state_sample_sha256": initial, "trained_state_sample_sha256": trained,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint / "model.pt"),
        "predictions_sha256": sha256(output / "predictions.npz"),
        "membership_sha256": sha256(output / "membership.npz"), "biolord_version": version("biolord"),
        "module": {**MODULE, "seed": seed}, "plan": PLAN, "training": TRAIN,
        "control_subtraction": "reconstructed_target_control_mean", "target_test_in_fitting": 0,
        "target_validation_in_fitting": 0, **metrics,
    })
    print(f"FIT_PASS {target} seed{seed} {setting}: {metrics}; {time.monotonic()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "report"))
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--task", type=int)
    parser.add_argument("--training-log-dir", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Preparation, training, and full report must run through SLURM")
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "train":
        if args.task is None or args.training_log_dir is None:
            parser.error("train requires --task and --training-log-dir")
        fit(args.root, args.task, args.training_log_dir)
    else:
        from report_figure4c_author_v2 import report
        report(args.root)


if __name__ == "__main__":
    main()
