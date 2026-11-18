"""BioLORD's genetic GO-neighbor formulation adapted to cross-cell-line transfer.

New artifacts only. Preparation and training must run on SLURM compute nodes.
The Norman author configuration is fixed before evaluation; no target perturbed
expression enters model fitting, validation, or feature construction.
"""

from portable_paths import (ROOT as PORTABLE_ROOT, PROJECT as PORTABLE_PROJECT, DATA_ROOT, REFERENCE_ROOT, BENCHMARK_ROOT, GO_FILE as PORTABLE_GO_FILE, RESULT_ROOT)

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from common import CELL_LINES, CONTROL_LABEL, dataset_path, hvg_names, pref, std_r
from run_biolord_corrected_v2 import registered_control_dataset, state_sample_digest


ROOT = PORTABLE_ROOT
PROJECT = PORTABLE_PROJECT
OUTPUT = RESULT_ROOT / "biolord/three_to_one_cellline_go_author_v1"
OLD = REFERENCE_ROOT / "benchmarks/context_generalization/results/biolord/three_to_one_corrected_v2"
FIGURE = REFERENCE_ROOT / "benchmarks/context_generalization/results/scgen/three_to_one_author_aligned_v1_numpycompat/report/results.json"
GO_FILE = PORTABLE_GO_FILE
DATA = DATA_ROOT
CHECKPOINTS = PROJECT / "checkpoints/biolord_cellline_go_author_v1"
PROTOCOL = "biolord_norman_go_pseudobulk_cellline_categorical_v1"
GO_KEY = "perturbation_neighbors1"
AUTHOR_COMMIT = "16bfefccc0caa013b11d222f9b02aaf535807f85"
GO_SHA256 = "f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771"

# scripts/biolord/norman/norman_optimal_config.py at AUTHOR_COMMIT.
# unknown_attribute_penalty is missing there: use the authors' value from
# utils/parameters_norman.py. It has no effect with unknown_attributes=False.
# The optimal config's latent_lr=.01 is used explicitly; the helper parameter
# file instead says .1. This also sets the newly added cell-line embedding LR.
MODULE = {
    "decoder_width": 32, "decoder_depth": 2, "decoder_activation": True,
    "attribute_nn_width": 64, "attribute_nn_depth": 2, "attribute_nn_activation": True,
    "use_batch_norm": False, "use_layer_norm": False,
    "unknown_attribute_noise_param": .2, "unknown_attributes": False,
    "n_latent_attribute_ordered": 32, "n_latent_attribute_categorical": 16,
    "reconstruction_penalty": 10000., "unknown_attribute_penalty": 10000.,
    "attribute_dropout_rate": 0., "gene_likelihood": "normal",
}
PLAN = {
    "n_epochs_warmup": 0, "latent_lr": .01, "latent_wd": .00001,
    "decoder_lr": .01, "decoder_wd": .01,
    "attribute_nn_lr": .001, "attribute_nn_wd": 4e-8,
    "step_size_lr": 45, "cosine_scheduler": True, "scheduler_final_lr": 1e-5,
}
TRAIN = {"max_epochs": 200, "batch_size": 32, "early_stopping": True,
         "early_stopping_patience": 200, "check_val_every_n_epoch": 5,
         "enable_checkpointing": False}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


class GOUnpickler(pickle.Unpickler):
    """The published resource is a dict of strings to sets; forbid other globals."""

    def find_class(self, module, name):
        if module in ("builtins", "__builtin__") and name in ("set", "frozenset"):
            return {"set": set, "frozenset": frozenset}[name]
        raise ValueError(f"Unexpected pickle global: {module}.{name}")


def load_annotations(path):
    if sha256(path) != GO_SHA256:
        raise ValueError("GO annotation resource differs from the pinned public download")
    with Path(path).open("rb") as handle:
        annotations = GOUnpickler(handle).load()
    if not isinstance(annotations, dict) or any(
        not isinstance(gene, str) or not isinstance(terms, (set, frozenset))
        or any(not isinstance(term, str) for term in terms)
        for gene, terms in annotations.items()
    ):
        raise ValueError("Unexpected gene-to-GO format")
    return annotations


def go_features(annotations, perturbations, gene_universe, k=20):
    """Exact GEARS Jaccard >.1, top k+1 incl. self, then author sum>1 filter.

    Sparse products only accelerate set intersection counting. The descriptor
    is a vector of gene-neighbor similarity scores, NOT a GO-term one-hot.
    Missing annotations and control have zero vectors, with coverage reported.
    Ties use sorted source-gene order. No expression values are used here.
    """
    genes = sorted(set(gene_universe) | (set(perturbations) - {CONTROL_LABEL}))
    labels = sorted(set(perturbations) | {CONTROL_LABEL})
    terms = sorted({term for gene in genes for term in annotations.get(gene, set())})
    term_index = {term: index for index, term in enumerate(terms)}
    gene_index = {gene: index for index, gene in enumerate(genes)}
    row, column = [], []
    for index, gene in enumerate(genes):
        for term in sorted(annotations.get(gene, set())):
            row.append(index)
            column.append(term_index[term])
    incidence = sparse.csr_matrix((np.ones(len(row), dtype=np.int32), (row, column)),
                                 shape=(len(genes), len(terms)))
    sizes = np.asarray(incidence.sum(1)).ravel()
    neighbors = np.zeros((len(labels), len(genes)), dtype=np.float64)
    for index, gene in enumerate(labels):
        if gene == CONTROL_LABEL:
            continue
        query = gene_index[gene]
        overlap = (incidence[query] @ incidence.T).toarray().ravel()
        union = sizes + sizes[query] - overlap
        scores = np.divide(overlap, union, out=np.zeros(len(genes)), where=union > 0)
        # Matches nlargest(k+1) on the >.1 graph, with deterministic tie order.
        candidates = np.flatnonzero(scores > .1)
        selected = candidates[np.argsort(-scores[candidates], kind="stable")[:k + 1]]
        neighbors[index, selected] = scores[selected]
    keep = neighbors.sum(0) > 1  # weighted SUM, not number of nonzero entries
    features = neighbors[:, keep].astype(np.float32)
    if not features.shape[1]:
        raise ValueError("The authors' weighted sum>1 filter removed all GO features")
    coverage = {
        "missing_annotation": [g for g in labels if g != CONTROL_LABEL and not annotations.get(g)],
        "zero_after_author_filter": [g for g, values in zip(labels, features)
                                     if g != CONTROL_LABEL and not np.any(values)],
        "n_features": int(features.shape[1]), "n_gene_universe": len(genes),
        "k": k, "score_threshold": .1, "column_filter": "weighted_column_sum > 1",
        "control_descriptor": "zero", "missing_descriptor_policy": "zero; never drop evaluation genes",
    }
    return labels, features, np.asarray(genes)[keep].tolist(), coverage


def sample_with_membership(path, requested, cap, seed, min_cells=10):
    """Exactly reproduce corrected-v2 sampling and record original row indices."""
    data = ad.read_h5ad(path, backed="r")
    try:
        labels = data.obs["gene"].astype(str).to_numpy()
        matrix = data.obsm["X_hvg"]
        generator = np.random.RandomState(seed)
        groups, membership = {}, {}
        for perturbation in sorted(set(requested)):
            indices = np.flatnonzero(labels == perturbation)
            if not len(indices) or (perturbation != CONTROL_LABEL and len(indices) < min_cells):
                continue
            if len(indices) > cap:
                indices = np.sort(generator.choice(indices, size=cap, replace=False))
            groups[perturbation] = np.asarray(matrix[indices], dtype=np.float32)
            membership[perturbation] = indices.tolist()
        names = hvg_names(data, matrix.shape[1])
        return groups, names, membership
    finally:
        data.file.close()


def make_pseudobulk(source_groups, target_controls, names, target, descriptors,
                    sampling_seed=42):
    """Condition means as in the genetic model; target perturbations not accepted."""
    means, rows = [], []
    for source in sorted(source_groups):
        if source == target:
            raise ValueError("Target cannot be a source")
        if CONTROL_LABEL not in source_groups[source]:
            raise ValueError("Missing source control")
        for perturbation, cells in sorted(source_groups[source].items()):
            means.append(cells.mean(0))
            rows.append({"cell_line": source, "perturbation": perturbation,
                         "role": "source", "split": "train", "n_cells": len(cells)})
    if len(target_controls) < 2:
        raise ValueError("At least two target controls required")
    means.append(target_controls.mean(0))
    rows.append({"cell_line": target, "perturbation": CONTROL_LABEL,
                 "role": "target_control", "split": "train", "n_cells": len(target_controls)})
    obs = pd.DataFrame(rows)
    obs.index = [f"{r['cell_line']}:{r['perturbation']}" for r in rows]
    # Fixed source-only validation of cell-line/perturbation combinations.
    # Keep controls and >=1 source per perturbation in optimizer training.
    generator = np.random.RandomState(sampling_seed)
    eligible = np.flatnonzero((obs.role == "source") & (obs.perturbation != CONTROL_LABEL))
    remaining = obs.loc[obs.role == "source", "perturbation"].value_counts().to_dict()
    goal = max(1, int(.1 * len(eligible)))
    validation = []
    for index in generator.permutation(eligible):
        perturbation = obs.iloc[index].perturbation
        if remaining[perturbation] > 1:
            validation.append(index)
            remaining[perturbation] -= 1
        if len(validation) == goal:
            break
    if not validation:
        raise ValueError("No source-only validation possible while retaining source evidence")
    obs.iloc[validation, obs.columns.get_loc("split")] = "validation"
    data = ad.AnnData(np.stack(means).astype(np.float32), obs=obs,
                      var=pd.DataFrame(index=names))
    data.obs["cell_line"] = data.obs["cell_line"].astype("category")
    data.obsm[GO_KEY] = np.stack([descriptors[p] for p in obs.perturbation]).astype(np.float32)
    if not np.isfinite(data.X).all() or not np.isfinite(data.obsm[GO_KEY]).all():
        raise ValueError("Nonfinite training values")
    return data


def build_model(data, seed):
    import biolord
    import scvi

    scvi.settings.seed = seed
    biolord.Biolord.setup_anndata(data, ordered_attributes_keys=[GO_KEY],
                                categorical_attributes_keys=["cell_line"])
    model = biolord.Biolord(data, n_latent=32, module_params={**MODULE, "seed": seed},
                            train_classifiers=False, split_key="split", train_split="train",
                            valid_split="validation", test_split="test")
    if torch.initial_seed() != seed or model.module.latent_codes.embed:
        raise ValueError("Wrong seed or unknown attributes unexpectedly enabled")
    return model


@torch.no_grad()
def expression_mean(model, control_dataset, descriptor):
    """Authors' native ordered-attribute replacement; preserve target cell line."""
    model.module.eval()
    tensors = dict(control_dataset)
    attribute = torch.as_tensor(descriptor, dtype=tensors[GO_KEY].dtype, device=model.device)
    if attribute.ndim != 1 or attribute.shape[0] != tensors[GO_KEY].shape[1]:
        raise ValueError("GO descriptor has wrong dimensions")
    tensors[GO_KEY] = attribute.unsqueeze(0).expand(tensors[GO_KEY].shape[0], -1)
    expression, _ = model.module.get_expression(tensors)
    if not torch.isfinite(expression).all():
        raise ValueError("Nonfinite native prediction")
    values = expression.detach().cpu().numpy()
    if values.ndim == 1 and tensors[GO_KEY].shape[0] == 1:
        values = values[None, :]
    if values.shape != (tensors[GO_KEY].shape[0], model.adata.n_vars):
        raise ValueError("Native expression lost its observation/gene axes")
    return values.mean(axis=0)


def assert_matches_old(perturbations, truth, features, old_path):
    with np.load(old_path, allow_pickle=False) as old:
        for key, value in (("test_perts", perturbations), ("true_deltas", truth),
                           ("feature_names", features)):
            if not np.array_equal(old[key], value):
                raise ValueError(f"Original Figure 3 {key} changed")


def prepare(root=OUTPUT, data_root=DATA, go_file=GO_FILE, figure=FIGURE):
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    annotations = load_annotations(go_file)
    original_figure = figure.read_bytes()
    cohorts, old_hashes = {}, {}
    for target in CELL_LINES:
        path = OLD / "seed0" / target / "predictions.npz"
        with np.load(path, allow_pickle=False) as old:
            cohorts[target] = old["test_perts"].astype(str).tolist()
        old_hashes[target] = sha256(path)
    perturbations = sorted(set().union(*map(set, cohorts.values())))
    reference = ad.read_h5ad(dataset_path(data_root, CELL_LINES[0]), backed="r")
    try:
        gene_universe = reference.var_names.astype(str).tolist()
    finally:
        reference.file.close()
    labels, features, neighbors, coverage = go_features(annotations, perturbations, gene_universe)
    descriptors = dict(zip(labels, features))
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    with (inputs / "original_figure3.json").open("xb") as handle:
        handle.write(original_figure)
    np.savez_compressed(inputs / "go_features.npz", perturbations=np.asarray(labels),
                        descriptors=features, neighbor_genes=np.asarray(neighbors))
    write_json(inputs / "go_coverage.json", coverage)
    artifacts = {"original_figure3.json": sha256(inputs / "original_figure3.json"),
                 "go_features.npz": sha256(inputs / "go_features.npz"),
                 "go_coverage.json": sha256(inputs / "go_coverage.json")}
    for target in CELL_LINES:
        selected = cohorts[target]
        sources = [cell for cell in CELL_LINES if cell != target]
        source_groups, memberships = {}, {}
        names = None
        for index, source in enumerate(sources):
            # Absent requested genes are ignored, exactly as in corrected-v2.
            groups, current_names, membership = sample_with_membership(
                dataset_path(data_root, source), [CONTROL_LABEL, *selected], 64, 42 + index)
            if names is not None and names != current_names:
                raise ValueError("Source feature order differs")
            names = current_names
            source_groups[source] = groups
            memberships[source] = membership
        target_groups, target_names, target_membership = sample_with_membership(
            dataset_path(data_root, target), [CONTROL_LABEL, *selected], 64, 142)
        if names != target_names:
            raise ValueError("Target feature order differs")
        truth = np.stack([target_groups[p].mean(0) - target_groups[CONTROL_LABEL].mean(0)
                          for p in selected]).astype(np.float32)
        assert_matches_old(np.asarray(selected), truth, np.asarray(names),
                           OLD / "seed0" / target / "predictions.npz")
        data = make_pseudobulk(source_groups, target_groups[CONTROL_LABEL], names, target, descriptors)
        directory = inputs / target
        directory.mkdir()
        data.write_h5ad(directory / "training.h5ad")
        # Target test membership is audit-only and never passed to build_model.
        write_json(directory / "membership.json", {"source": memberships,
                   "target_control_rows": target_membership[CONTROL_LABEL],
                   "target_evaluation_rows": {p: target_membership[p] for p in selected}})
        np.savez_compressed(directory / "evaluation.npz", test_perts=np.asarray(selected),
                            true_deltas=truth, feature_names=np.asarray(names),
                            observed_control_mean=target_groups[CONTROL_LABEL].mean(0))
        for name in ("training.h5ad", "membership.json", "evaluation.npz"):
            artifacts[f"{target}/{name}"] = sha256(directory / name)
        print(f"PREPARED {target}: {data.n_obs} condition means; {len(selected)} evaluation genes", flush=True)
    write_json(inputs / "provenance.json", {
        "protocol": PROTOCOL, "author_repository": "nitzanlab/biolord_reproducibility",
        "author_commit": AUTHOR_COMMIT, "author_config": "scripts/biolord/norman/norman_optimal_config.py",
        "module": MODULE, "plan": PLAN, "training": TRAIN,
        "n_latent": 32, "go_source": str(go_file), "go_sha256": sha256(go_file),
        "go_source_dataverse_file_id": 6153417, "go_coverage": coverage,
        "go_universe": "shared measured genes plus benchmark perturbation identities; no expression",
        "original_figure3": str(figure), "old_seed0_predictions_sha256": old_hashes,
        "sampling_seed": 42, "max_cells_per_group": 64, "n_seeds": 5,
        "control_subtraction": "reconstructed_target_control_mean",
        "author_adaptations": ["categorical cell_line added to single-cell-line genetic model",
             "fixed source-only combination validation; all controls and >=1 source per perturbation train",
             "existing sampled 64-cell means, aligned X_hvg, cohorts, target truth and pref retained",
             "five initialization seeds on fixed data, not CV",
             "optimal config latent_lr=.01; missing unknown penalty taken from parameters_norman (inactive)"],
        "artifacts": artifacts, "runner_sha256": sha256(Path(__file__)),
    })
    (inputs / "prepare_pass").touch(exist_ok=False)
    print(f"PREPARE_PASS {root}; GO coverage: {coverage}", flush=True)


def verify_inputs(root):
    inputs = root / "inputs"
    if not (inputs / "prepare_pass").is_file():
        raise ValueError("Preparation has not completed")
    provenance = json.loads((inputs / "provenance.json").read_text())
    if provenance["protocol"] != PROTOCOL or provenance["runner_sha256"] != sha256(Path(__file__)):
        raise ValueError("Protocol or runner changed after preparation")
    if provenance["module"] != MODULE or provenance["plan"] != PLAN or provenance["training"] != TRAIN:
        raise ValueError("Author configuration changed")
    for relative, digest in provenance["artifacts"].items():
        if sha256(inputs / relative) != digest:
            raise ValueError(f"Prepared input changed: {relative}")
    return provenance


def train(root, task, training_log_dir, checkpoint_root=CHECKPOINTS):
    if not 0 <= task < 20:
        raise ValueError("Task must be in 0..19")
    target, seed = CELL_LINES[task // 5], task % 5
    output = root / f"seed{seed}" / target
    checkpoint = checkpoint_root / f"seed{seed}" / target
    if output.exists() or checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("Submit training to SLURM with a GPU")
    verify_inputs(root)
    data = ad.read_h5ad(root / "inputs" / target / "training.h5ad")
    target_mask = data.obs.cell_line.astype(str).to_numpy() == target
    if target_mask.sum() != 1 or not np.all(data.obs.perturbation[target_mask] == CONTROL_LABEL):
        raise ValueError("Target perturbations leaked into model data")
    model = build_model(data, seed)
    initial = state_sample_digest(model)
    start = time.monotonic()
    model.train(accelerator="gpu", device=1, plan_kwargs=dict(PLAN), logger=False,
                default_root_dir=str(training_log_dir), num_workers=0, **TRAIN)
    elapsed = time.monotonic() - start
    final = state_sample_digest(model)
    if final == initial:
        raise ValueError("Model parameters did not change")
    controls = registered_control_dataset(model, np.flatnonzero(target_mask))
    with np.load(root / "inputs/go_features.npz", allow_pickle=False) as file:
        descriptors = dict(zip(file["perturbations"].astype(str), file["descriptors"]))
    with np.load(root / "inputs" / target / "evaluation.npz", allow_pickle=False) as file:
        evaluation = {key: file[key].copy() for key in file.files}
    baseline = expression_mean(model, controls, descriptors[CONTROL_LABEL])
    expression = np.stack([expression_mean(model, controls, descriptors[p])
                           for p in evaluation["test_perts"].astype(str)])
    predictions = (expression - baseline).astype(np.float32)
    truth = evaluation["true_deltas"]
    reference = truth.mean(0)
    assert_matches_old(evaluation["test_perts"], truth, evaluation["feature_names"],
                       OLD / f"seed{seed}" / target / "predictions.npz")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint), overwrite=False, save_anndata=False)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", **evaluation, predictions=predictions,
                        train_mean=reference, predicted_expression_means=expression,
                        reconstructed_control_mean=baseline)
    np.savez_compressed(output / "membership.npz", cell_line=data.obs.cell_line.to_numpy(dtype=str),
                        perturbation=data.obs.perturbation.to_numpy(dtype=str),
                        role=data.obs.role.to_numpy(dtype=str),
                        train_indices=np.asarray(model.train_indices),
                        validation_indices=np.asarray(model.valid_indices),
                        test_indices=np.asarray(model.test_indices))
    metadata = {
        "protocol": PROTOCOL, "target": target, "seed": seed, "sampling_seed": 42,
        "module": {**MODULE, "seed": seed}, "plan": PLAN, "training": TRAIN,
        "categorical_attributes": ["cell_line"], "ordered_attributes": [GO_KEY],
        "unknown_attributes": False, "target_perturbed_cells_in_fitting": 0,
        "n_condition_means": data.n_obs, "n_training_means": len(model.train_indices),
        "n_validation_means": len(model.valid_indices), "n_perturbations": len(predictions),
        "old_truth_exact_match": True, "control_subtraction": "reconstructed_target_control_mean",
        "initial_state_sample_sha256": initial, "trained_state_sample_sha256": final,
        "epochs_completed": int(model.training_plan.current_epoch), "seconds": elapsed,
        "biolord_version": version("biolord"), "checkpoint": str(checkpoint),
        "pref": pref(truth, predictions, reference), "std_r": std_r(truth, predictions),
    }
    # Metadata is the final completion marker; incomplete output is never reused.
    write_json(output / "metrics.json", metadata)
    print(f"TRAIN_PASS {target} seed{seed}: pref={metadata['pref']:.5f}; {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "report"))
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--go-file", type=Path, default=GO_FILE)
    parser.add_argument("--task", type=int)
    parser.add_argument("--training-log-dir", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Preparation, training and full reporting must run on SLURM compute nodes")
    if args.action == "prepare":
        prepare(args.root, args.data_root, args.go_file)
    elif args.action == "train":
        if args.task is None or args.training_log_dir is None:
            parser.error("train requires --task and --training-log-dir")
        train(args.root, args.task, args.training_log_dir)
    else:
        from report_biolord_cellline_go_v1 import report
        report(args.root)


if __name__ == "__main__":
    main()
