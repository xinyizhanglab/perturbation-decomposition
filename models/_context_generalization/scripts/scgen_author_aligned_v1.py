"""Binary scGen transfer following the authors' held-out-stimulation tutorial.

Benchmark-specific choices retained from Figure 3: X_hvg features, <=128
cells/condition, original perturbation manifest and sampled-pseudobulk pref.
"""

from portable_paths import (ROOT as PORTABLE_ROOT, PROJECT as PORTABLE_PROJECT, DATA_ROOT, REFERENCE_ROOT, BENCHMARK_ROOT, GO_FILE as PORTABLE_GO_FILE, RESULT_ROOT)

import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch

from common import CELL_LINES, CONTROL_LABEL, dataset_path, hvg_names
from run_biolord_corrected_v2 import state_sample_digest


ROOT = PORTABLE_ROOT
PROJECT = PORTABLE_PROJECT
OUTPUT = RESULT_ROOT / "scgen/three_to_one_author_aligned_v1_numpycompat"
MANIFEST = REFERENCE_ROOT / "benchmarks/context_generalization/manifests/scgen_3to1.jsonl"
OLD_RESULTS = REFERENCE_ROOT / "benchmarks/context_generalization/results/scgen/three_to_one_5seed"
PROTOCOL = "scgen_binary_target_controls_real_cell_labels_native_all_v1"
DATA = DATA_ROOT
N_SHARDS = 128


@contextmanager
def legacy_numpy_infinity():
    """Restore only the old infinity spelling during native Lightning training.

    Lightning 1.7.7 initializes its EarlyStopping score with np.Inf. NumPy 2
    removed that spelling; np.inf is the identical positive-infinity value.
    Restore the original module state even if fitting raises an exception.
    """
    missing = object()
    previous = np.__dict__.get("Inf", missing)
    np.Inf = np.inf
    try:
        yield
    finally:
        if previous is missing:
            del np.Inf
        else:
            np.Inf = previous


def read_tasks(path=MANIFEST):
    tasks = [json.loads(line) for line in path.read_text().splitlines()]
    if len(tasks) != 5486:
        raise ValueError(f"Expected original 5486-task Figure 3 manifest: {path}")
    if len({(task['target'], task['perturbation']) for task in tasks}) != len(tasks):
        raise ValueError("Duplicate target/perturbation tasks")
    return tasks


def sample_indices(labels, perturbation, seed, maximum=128):
    """Exactly preserve common.sample_groups' sorted-condition sampling order."""
    rng = np.random.RandomState(seed)
    result = {}
    for condition in sorted({CONTROL_LABEL, perturbation}):
        rows = np.flatnonzero(labels == condition)
        if len(rows) > maximum:
            rows = np.sort(rng.choice(rows, size=maximum, replace=False))
        if len(rows):
            result[condition] = rows
    if set(result) != {CONTROL_LABEL, perturbation}:
        raise ValueError("Missing control or perturbation")
    return result


def load_cache(data_root=DATA):
    """Worker-only: load each existing expression matrix once, not per fit."""
    cache = {}
    for cell in CELL_LINES:
        path = dataset_path(data_root, cell)
        data = ad.read_h5ad(path, backed="r")
        try:
            matrix = np.asarray(data.obsm["X_hvg"], dtype=np.float32)
            cache[cell] = {"X": matrix, "labels": data.obs["gene"].astype(str).to_numpy(),
                           "features": hvg_names(data, matrix.shape[1])}
        finally:
            data.file.close()
        print(f"CACHE {cell}: {matrix.shape}", flush=True)
    return cache


def build_task(cache, task, task_index):
    """Target perturbed expression is excluded from the complete model AnnData."""
    target, perturbation = task["target"], task["perturbation"]
    if target in task["sources"] or not task["sources"]:
        raise ValueError("Invalid source/target assignment")
    features = cache[target]["features"]
    parts = []
    for source_index, source in enumerate(task["sources"]):
        sample = cache[source]
        if sample["features"] != features:
            raise ValueError("Source/target feature-order mismatch")
        selected = sample_indices(sample["labels"], perturbation, 42 + task_index + source_index)
        for condition in (CONTROL_LABEL, perturbation):
            rows = selected[condition]
            obs = pd.DataFrame({"condition": "control" if condition == CONTROL_LABEL else "stimulated",
                                "cell_state": source, "perturbation": condition,
                                "role": "source", "original_row_index": rows},
                               index=[f"{source}:{row}" for row in rows])
            parts.append(ad.AnnData(sample["X"][rows].copy(), obs=obs,
                                    var=pd.DataFrame(index=features)))
    selected = sample_indices(cache[target]["labels"], perturbation, 42 + task_index + 100)
    controls = selected[CONTROL_LABEL]
    target_obs = pd.DataFrame({"condition": "control", "cell_state": target,
                               "perturbation": CONTROL_LABEL, "role": "target_control",
                               "original_row_index": controls},
                              index=[f"{target}:{row}" for row in controls])
    parts.append(ad.AnnData(cache[target]["X"][controls].copy(), obs=target_obs,
                            var=pd.DataFrame(index=features)))
    training = ad.concat(parts, join="inner")
    for key in ("condition", "cell_state"):
        training.obs[key] = training.obs[key].astype("category")
    target_mask = training.obs["cell_state"].astype(str) == target
    if not (training.obs.loc[target_mask, "condition"] == "control").all():
        raise ValueError("Target stimulated cells leaked into model AnnData")
    if set(training.obs["cell_state"].astype(str)) != {target, *task["sources"]}:
        raise ValueError("Real cell-line labels were lost")
    if training.obs_names.has_duplicates:
        raise ValueError("Duplicate training-cell IDs")
    if len(controls) < 2 or training.n_obs < 20:
        raise ValueError("Insufficient cells for training and validation")
    return training, controls, selected[perturbation]


def fit_predict(training, target, seed, log_dir, max_epochs=100, checkpoint_dir=None):
    """Use the tutorial's model, batch size and early-stopping arguments."""
    import scgen
    import scvi

    if not torch.cuda.is_available():
        raise RuntimeError("Submit native scGen training through SLURM with a GPU")
    scvi.settings.seed = seed
    scgen.SCGEN.setup_anndata(training, batch_key="condition", labels_key="cell_state")
    model = scgen.SCGEN(training)
    initial = state_sample_digest(model)
    started = time.monotonic()
    with legacy_numpy_infinity():
        model.train(max_epochs=max_epochs, batch_size=32, use_gpu=True,
                    train_size=0.9, early_stopping=True, early_stopping_patience=25,
                    enable_checkpointing=False, logger=False, num_sanity_val_steps=0,
                    enable_progress_bar=False, simple_progress_bar=False,
                    default_root_dir=str(log_dir))
    model.module.eval()
    # Deliberately retain the tutorial's default restrict_arithmetic_to='all'.
    # Target controls participate in this arithmetic; no target stimulated cells do.
    predicted, latent_shift = model.predict(ctrl_key="control", stim_key="stimulated",
                                            celltype_to_predict=target)
    target_controls = training[training.obs["role"] == "target_control"]
    if not predicted.obs_names.equals(target_controls.obs_names):
        raise ValueError("Native prediction did not use the expected target controls")
    control_mean = target_controls.X.mean(0)
    predicted_mean = np.asarray(predicted.X).mean(0)
    delta = np.asarray(predicted_mean - control_mean, dtype=np.float32)
    if not np.isfinite(delta).all() or delta.shape != (training.n_vars,):
        raise ValueError("Invalid predicted response")
    train_indices = np.asarray(model.train_indices)
    val_indices = np.asarray(model.validation_indices)
    if set(train_indices) & set(val_indices) or set(train_indices) | set(val_indices) != set(range(training.n_obs)):
        raise ValueError("Unexpected native optimizer/validation membership")
    trained = state_sample_digest(model)
    if initial == trained:
        raise ValueError("Model parameters did not update")
    stopping = [type(callback).__name__ for callback in model.trainer.callbacks
                if "EarlyStopping" in type(callback).__name__]
    if len(stopping) != 1 or not len(val_indices):
        raise ValueError("Native early stopping/validation was not enabled")
    metadata = {"effective_training_seed": seed, "batch_size": 32, "max_epochs": max_epochs,
                "early_stopping": True, "early_stopping_patience": 25,
                "early_stopping_callback": stopping[0],
                "numpy_compatibility": "temporary_np_Inf_alias_during_native_train",
                "epochs_completed": int(model.trainer.current_epoch),
                "seconds": time.monotonic() - started,
                "n_model_cells": training.n_obs, "n_optimizer_cells": len(train_indices),
                "n_validation_cells": len(val_indices), "target_perturbed_cells_in_fitting": 0,
                "target_controls_in_model_anndata": True, "arithmetic": "native_default_all",
                "control_subtraction": "observed_target_control_mean",
                "initial_state_sample_sha256": initial, "trained_state_sample_sha256": trained,
                "training_membership_sha256": hashlib.sha256("\n".join(training.obs_names).encode()).hexdigest(),
                "scgen_version": version("scgen"), "scvi_version": version("scvi-tools")}
    if checkpoint_dir is not None:
        model.save(str(checkpoint_dir), overwrite=False, save_anndata=True)
        metadata["checkpoint"] = str(checkpoint_dir)
        metadata["checkpoint_model_sha256"] = hashlib.sha256(
            (checkpoint_dir / "model.pt").read_bytes()).hexdigest()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return delta, predicted_mean, control_mean, latent_shift, train_indices, val_indices, metadata


def run_one(cache, task, task_index, seed, root, log_dir, max_epochs=100):
    directory = root / f"seed{seed}" / task["target"] / task["perturbation"]
    if directory.exists():
        meta = directory / "metadata.json"
        if not meta.is_file():
            raise FileExistsError(f"Incomplete output preserved; refusing overwrite: {directory}")
        saved = json.loads(meta.read_text())
        selected_tasks = os.environ.get("PD_SCGEN_CHECKPOINT_TASKS", "")
        requested_checkpoint = (os.environ.get("PD_SAVE_SCGEN_CHECKPOINTS") == "1" and
                                (not selected_tasks or task_index in
                                 {int(value) for value in selected_tasks.split(",")}))
        if requested_checkpoint and not (directory / "checkpoint/model.pt").is_file():
            raise ValueError("Completed fits have no recoverable weights; use a new training output root for checkpoint export")
        if (saved["protocol"], saved["task_index"], saved["seed"], saved["max_epochs"]) != (PROTOCOL, task_index, seed, max_epochs):
            raise ValueError(f"Existing output does not match requested fit: {directory}")
        with np.load(directory / "predictions.npz", allow_pickle=False) as arrays:
            if not np.isfinite(arrays["predictions"]).all():
                raise ValueError("Existing fit contains invalid predictions")
        print(f"SKIP_VALIDATED {seed}/{task_index}", flush=True)
        return
    training, control_rows, perturbed_rows = build_task(cache, task, task_index)
    # Truth is evaluation-only; exactly match the original Figure 3 samples.
    true_delta = (cache[task["target"]]["X"][perturbed_rows].mean(0)
                  - cache[task["target"]]["X"][control_rows].mean(0))
    old_path = OLD_RESULTS / f"seed{seed}" / task["target"] / task["perturbation"] / "predictions.npz"
    with np.load(old_path, allow_pickle=False) as old:
        if not np.array_equal(true_delta, old["true_deltas"][0]):
            raise ValueError(f"Target evaluation truth differs from old Figure 3: {old_path}")
    directory.mkdir(parents=True, exist_ok=False)
    print(f"START seed{seed} task{task_index} {task['target']}/{task['perturbation']} "
          f"n={training.n_obs}", flush=True)
    selected = os.environ.get("PD_SCGEN_CHECKPOINT_TASKS", "")
    retain = (os.environ.get("PD_SAVE_SCGEN_CHECKPOINTS") == "1" and
              (not selected or task_index in {int(value) for value in selected.split(",")}))
    checkpoint_dir = directory / "checkpoint" if retain else None
    delta, predicted_mean, control_mean, shift, optimizer, validation, details = fit_predict(
        training, task["target"], seed + task_index, log_dir, max_epochs, checkpoint_dir)
    if retain:
        with (checkpoint_dir / "protocol.json").open("x") as handle:
            json.dump({"protocol": PROTOCOL, "target": task["target"],
                       "perturbation": task["perturbation"], "seed": seed,
                       "task_index": task_index, "model_sha256": details["checkpoint_model_sha256"]}, handle)
    np.savez_compressed(directory / "predictions.npz", test_perts=np.asarray([task["perturbation"]]),
                        predictions=delta[None], true_deltas=true_delta[None],
                        predicted_expression_mean=predicted_mean, observed_control_mean=control_mean,
                        latent_shift=shift, feature_names=training.var_names.to_numpy().astype(str))
    np.savez_compressed(directory / "membership.npz",
                        cell_line=training.obs["cell_state"].astype(str).to_numpy().astype(str),
                        perturbation=training.obs["perturbation"].astype(str).to_numpy().astype(str),
                        original_row_index=training.obs["original_row_index"].to_numpy(),
                        role=training.obs["role"].to_numpy().astype(str),
                        optimizer_indices=optimizer, validation_indices=validation,
                        evaluation_perturbed_row_indices=perturbed_rows)
    metadata = {"protocol": PROTOCOL, "task_index": task_index, "seed": seed,
                "sampling_seed": 42, "max_cells_per_group": 128,
                "target": task["target"], "sources": task["sources"],
                "perturbation": task["perturbation"], "old_truth_exact_match": True, **details}
    with (directory / "metadata.json").open("x") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"DONE seed{seed} task{task_index} {task['target']}/{task['perturbation']} "
          f"{details['seconds']:.1f}s epochs={details['epochs_completed']}", flush=True)


def complete_shard(root, shard, n_shards):
    status = root / ".completion"
    status.mkdir(exist_ok=True)
    (status / f"shard{shard}.done").touch(exist_ok=True)
    if not all((status / f"shard{index}.done").is_file() for index in range(n_shards)):
        print(f"SHARD_PASS {shard}/{n_shards}", flush=True)
        return
    try:
        (status / "report_claim").mkdir(exist_ok=False)
    except FileExistsError:
        return
    from report_scgen_author_aligned_v1 import report
    report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--n-shards", type=int, default=N_SHARDS)
    parser.add_argument("--workers", type=int, default=N_SHARDS)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--training-log-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads((args.root / "inputs/provenance.json").read_text())
    if snapshot["protocol"] != PROTOCOL or snapshot["n_shards"] != args.n_shards:
        raise ValueError("Prepared protocol/shard count mismatch")
    if not 0 < args.workers <= args.n_shards or not 0 <= args.shard < args.workers:
        raise ValueError("Invalid worker/shard assignment")
    tasks = read_tasks(args.root / "inputs/manifest.jsonl")
    cache = load_cache()
    # Distribute target/perturbation tasks, keeping all five repeats on one worker.
    owned_shards = list(range(args.shard, args.n_shards, args.workers))
    for shard in owned_shards:
        for task_index in range(shard, len(tasks), args.n_shards):
            for seed in range(5):
                run_one(cache, tasks[task_index], task_index, seed, args.root, args.training_log_dir)
    for shard in owned_shards:
        complete_shard(args.root, shard, args.n_shards)


if __name__ == "__main__":
    main()
