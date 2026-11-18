#!/usr/bin/env python3
"""Native BioLORD transfer with fitted target-control codes and explicit seeds.

Target perturbed cells never enter model fitting. Target controls do enter
fitting, so this is control-adapted response transfer, not strict zero-shot
generalization to a completely unseen cell line.
"""

from portable_paths import (ROOT as PORTABLE_ROOT, PROJECT as PORTABLE_PROJECT, DATA_ROOT, REFERENCE_ROOT, BENCHMARK_ROOT, GO_FILE as PORTABLE_GO_FILE, RESULT_ROOT)

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from common import CELL_LINES, CONTROL_LABEL, dataset_path, pref, sample_groups, std_r, valid_perturbations


VERSION = "biolord_corrected_v2"
ROOT = PORTABLE_ROOT


def make_training_data(source_groups, target_controls, names, target, sampling_seed=42):
    parts = []
    generator = np.random.RandomState(sampling_seed)
    for source, groups in source_groups.items():
        if source == target:
            raise ValueError("Target cell line cannot be a source")
        for perturbation, matrix in sorted(groups.items()):
            split = np.full(len(matrix), "train", dtype=object)
            if len(matrix) > 1:
                count = max(1, int(len(matrix) * 0.1))
                split[generator.choice(len(matrix), size=count, replace=False)] = "validation"
            obs = pd.DataFrame({
                "perturbation": perturbation, "cell_line": source,
                "role": "source", "split": split,
            }, index=[f"{source}:{perturbation}:{index}" for index in range(len(matrix))])
            parts.append(ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=names)))
    obs = pd.DataFrame({
        "perturbation": CONTROL_LABEL, "cell_line": target,
        "role": "target_control", "split": "train",
    }, index=[f"{target}:target_control:{index}" for index in range(len(target_controls))])
    parts.append(ad.AnnData(X=target_controls, obs=obs, var=pd.DataFrame(index=names)))
    data = ad.concat(parts, join="inner")
    data.obs["perturbation"] = data.obs["perturbation"].astype("category")
    target_indices = np.flatnonzero(data.obs["role"].to_numpy() == "target_control")
    if len(target_indices) < 2:
        raise ValueError("At least two target controls are required")
    target_rows = data.obs["cell_line"] == target
    if not (data.obs.loc[target_rows, "perturbation"] == CONTROL_LABEL).all():
        raise ValueError("Target perturbed cells leaked into fitting")
    if not (data.obs.iloc[target_indices]["split"] == "train").all():
        raise ValueError("Every target control must have a fitted latent code")
    if not (data.obs["split"] == "validation").any():
        raise ValueError("Source-only validation split is empty")
    return data, target_indices


def build_model(data, seed, n_latent=128, module_options=None):
    import biolord
    import scvi

    scvi.settings.seed = seed
    biolord.Biolord.setup_anndata(data, categorical_attributes_keys=["perturbation"])
    options = {**(module_options or {}), "seed": seed}
    model = biolord.Biolord(
        data, n_latent=n_latent, module_params=options,
        split_key="split", train_split="train", valid_split="validation", test_split="test",
    )
    if torch.initial_seed() != seed or scvi.settings.seed != seed:
        raise RuntimeError("BioLORD reset the requested seed during initialization")
    return model


def state_sample_digest(model):
    digest = hashlib.sha256()
    for name, value in model.module.state_dict().items():
        digest.update(name.encode())
        sample = value.detach().reshape(-1)[:64].cpu().contiguous().numpy()
        digest.update(sample.tobytes())
    return digest.hexdigest()


def registered_control_dataset(model, target_indices):
    from scvi import REGISTRY_KEYS

    if not set(target_indices).issubset(set(model.train_indices)):
        raise ValueError("Prediction controls were not assigned to the training split")
    data = model.get_dataset(model.adata, indices=target_indices)
    actual = data[REGISTRY_KEYS.INDICES_KEY].detach().cpu().numpy().ravel()
    if not np.array_equal(actual, target_indices):
        raise ValueError("Prediction controls do not retain their registered global indices")
    if not (model.adata.obs.iloc[actual]["role"] == "target_control").all():
        raise ValueError("Prediction attempted to use source-cell latent codes")
    return {key: value.to(model.device) for key, value in data.items()}


@torch.no_grad()
def condition_mean(model, control_dataset, perturbation):
    model.module.eval()
    category = model.categorical_attributes_map["perturbation"][perturbation]
    dataset = dict(control_dataset)
    dataset["perturbation"] = torch.full_like(control_dataset["perturbation"], category)
    means, _ = model.module.get_expression(dataset)
    if not torch.isfinite(means).all():
        raise ValueError(f"Nonfinite prediction for {perturbation}")
    return means.detach().cpu().numpy().mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=CELL_LINES, required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=RESULT_ROOT / "biolord/three_to_one_corrected_v2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-cells-per-group", type=int, default=64)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--n-latent", type=int, default=128)
    parser.add_argument("--training-log-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.out_root / f"seed{args.seed}" / args.target
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this training through SLURM with an allocated GPU")
    sources = [source for source in CELL_LINES if source != args.target]
    available = {cell: valid_perturbations(dataset_path(args.data_root, cell), args.min_cells)
                 for cell in CELL_LINES}
    perturbations = sorted(available[args.target] & set.union(*(available[source] for source in sources)))
    if not perturbations:
        raise ValueError("No source-observed target perturbations")
    source_groups = {}
    names = None
    for index, source in enumerate(sources):
        groups, current_names = sample_groups(
            dataset_path(args.data_root, source), [CONTROL_LABEL, *(set(perturbations) & available[source])],
            args.max_cells_per_group, args.sampling_seed + index,
        )
        if CONTROL_LABEL not in groups:
            raise ValueError(f"Missing controls in {source}")
        if names is not None and current_names != names:
            raise ValueError("Source feature order mismatch")
        names = current_names
        source_groups[source] = groups
    target_groups, target_names = sample_groups(
        dataset_path(args.data_root, args.target), [CONTROL_LABEL, *perturbations],
        args.max_cells_per_group, args.sampling_seed + 100,
    )
    if names != target_names or CONTROL_LABEL not in target_groups:
        raise ValueError("Target feature order mismatch or missing target controls")
    data, target_indices = make_training_data(
        source_groups, target_groups[CONTROL_LABEL], names, args.target, args.sampling_seed,
    )
    model = build_model(data, args.seed, args.n_latent)
    initial_digest = state_sample_digest(model)
    torch_seed_after_init = torch.initial_seed()
    controls = registered_control_dataset(model, target_indices)
    del controls
    model.train(
        max_epochs=args.epochs, accelerator="gpu", device=1, batch_size=args.batch_size,
        early_stopping=True, enable_checkpointing=False, logger=False,
        default_root_dir=str(args.training_log_dir),
    )
    model.module.eval()
    controls = registered_control_dataset(model, target_indices)
    baseline = condition_mean(model, controls, CONTROL_LABEL)
    predictions = np.stack([condition_mean(model, controls, perturbation) - baseline
                            for perturbation in perturbations]).astype(np.float32)
    true_control = target_groups[CONTROL_LABEL].mean(axis=0)
    truth = np.stack([target_groups[perturbation].mean(axis=0) - true_control
                      for perturbation in perturbations]).astype(np.float32)
    reference = truth.mean(axis=0)
    if not np.isfinite(predictions).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite evaluation arrays")
    metrics = {
        "model": "biolord", "implementation": VERSION,
        "setting": "three_source_cell_lines_with_target_control_adaptation",
        "target": args.target, "sources": sources,
        "target_controls_in_fitting": True, "target_perturbed_cells_in_fitting": 0,
        "n_target_controls": int(len(target_indices)), "n_fit_cells": int(data.n_obs),
        "n_training_cells": int(len(model.train_indices)), "n_validation_cells": int(len(model.valid_indices)),
        "n_perturbations": len(perturbations), "seed": args.seed, "module_seed_argument": args.seed,
        "torch_seed_after_model_init": torch_seed_after_init, "sampling_seed": args.sampling_seed,
        "initial_state_sample_sha256": initial_digest,
        "trained_state_sample_sha256": state_sample_digest(model),
        "pref": pref(truth, predictions, reference), "std_r": std_r(truth, predictions),
        "epochs_requested": args.epochs, "max_cells_per_group": args.max_cells_per_group,
        "features": "existing aligned 2000-gene X_hvg; categorical perturbation only",
        "biolord_version": importlib.metadata.version("biolord"),
    }
    if model.training_plan is not None:
        metrics["trainer_epoch_at_finish"] = int(model.training_plan.current_epoch)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", test_perts=np.asarray(perturbations),
                        predictions=predictions, true_deltas=truth, train_mean=reference,
                        target_control_indices=target_indices, feature_names=np.asarray(names))
    with (output / "metrics.json").open("x") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
