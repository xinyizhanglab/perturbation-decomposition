"""Re-export full gene vectors from frozen BioLORD checkpoints; never retrain.

The v1 training code and all failed exports remain immutable. The only
prediction change is restoring the singleton batch axis removed by native
BioLORD inference before averaging across observations (never across genes).
"""

from portable_paths import (ROOT as PORTABLE_ROOT, PROJECT as PORTABLE_PROJECT, DATA_ROOT, REFERENCE_ROOT, BENCHMARK_ROOT, GO_FILE as PORTABLE_GO_FILE, RESULT_ROOT)

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time

import anndata as ad
import numpy as np
import torch

from biolord_cellline_go_v1 import (
    CELL_LINES, CONTROL_LABEL, GO_KEY, OUTPUT, assert_matches_old, registered_control_dataset,
    sha256, state_sample_digest, verify_inputs, write_json,
)
from common import pref, std_r
from report_biolord_cellline_go_v1 import report, validate_fit


RECOVERED = OUTPUT.with_name("three_to_one_cellline_go_author_v1_exportfixed_v2")
RECOVERY_PROTOCOL = "singleton_batch_axis_restored_before_observation_mean_v2"


def observation_mean(expression, n_observations, n_genes):
    """Validate native output axes and return exactly one vector over genes."""
    if expression.ndim == 1 and n_observations == 1 and expression.shape[0] == n_genes:
        expression = expression.unsqueeze(0)
    if tuple(expression.shape) != (n_observations, n_genes):
        raise ValueError(f"Native expression shape {tuple(expression.shape)} is not "
                         f"({n_observations}, {n_genes})")
    if not torch.isfinite(expression).all():
        raise ValueError("Nonfinite native expression")
    result = expression.detach().cpu().numpy().mean(axis=0)
    if result.shape != (n_genes,) or not np.isfinite(result).all():
        raise ValueError("Observation mean lost the gene axis")
    return result


@torch.no_grad()
def expression_mean(model, control_dataset, descriptor):
    model.module.eval()
    tensors = dict(control_dataset)
    original = tensors[GO_KEY]
    attribute = torch.as_tensor(descriptor, dtype=original.dtype, device=model.device)
    if attribute.ndim != 1 or attribute.shape[0] != original.shape[1]:
        raise ValueError("Wrong GO descriptor dimensions")
    n_observations = original.shape[0]
    tensors[GO_KEY] = attribute.unsqueeze(0).expand(n_observations, -1)
    expression, _ = model.module.get_expression(tensors)
    return observation_mean(expression, n_observations, model.adata.n_vars)


def load_checkpoint(metadata, data):
    import biolord

    model = biolord.Biolord.load(metadata["checkpoint"], adata=data,
                                accelerator="cpu", device=1)
    if state_sample_digest(model) != metadata["trained_state_sample_sha256"]:
        raise ValueError("Loaded weights differ from the saved training fingerprint")
    if model.module.latent_codes.embed:
        raise ValueError("Unknown latent codes unexpectedly enabled")
    if set(model.categorical_attributes_map) != {"cell_line"} or set(model.ordered_attributes_map) != {GO_KEY}:
        raise ValueError("Checkpoint attributes differ from the trained model")
    return model


def recover_one(parent, root, target, seed, expected_hashes):
    from biolord_cellline_go_v1 import OLD

    original = parent / f"seed{seed}" / target
    metadata = json.loads((original / "metrics.json").read_text())
    checkpoint_file = Path(metadata["checkpoint"]) / "model.pt"
    for path, digest in expected_hashes.items():
        if sha256(path) != digest:
            raise ValueError(f"Original artifact changed before recovery: {path}")
    output = root / f"seed{seed}" / target
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    data = ad.read_h5ad(root / "inputs" / target / "training.h5ad")
    target_indices = np.flatnonzero(data.obs.cell_line.astype(str).to_numpy() == target)
    if len(target_indices) != 1 or data.obs.iloc[target_indices[0]].perturbation != CONTROL_LABEL:
        raise ValueError("Target perturbations entered the prepared model data")
    start = time.monotonic()
    model = load_checkpoint(metadata, data)
    initial_state = state_sample_digest(model)
    controls = registered_control_dataset(model, target_indices)
    with np.load(root / "inputs/go_features.npz", allow_pickle=False) as file:
        descriptors = dict(zip(file["perturbations"].astype(str), file["descriptors"]))
    with np.load(root / "inputs" / target / "evaluation.npz", allow_pickle=False) as file:
        evaluation = {key: file[key].copy() for key in file.files}
    baseline = expression_mean(model, controls, descriptors[CONTROL_LABEL])
    expression = np.stack([expression_mean(model, controls, descriptors[p])
                           for p in evaluation["test_perts"].astype(str)])
    predictions = (expression - baseline).astype(np.float32)
    truth = evaluation["true_deltas"]
    if predictions.shape != truth.shape or predictions.ndim != 2 or not np.isfinite(predictions).all():
        raise ValueError("Recovered predictions do not match the full gene-by-perturbation evaluation")
    if state_sample_digest(model) != initial_state:
        raise ValueError("Model weights changed during prediction")
    assert_matches_old(evaluation["test_perts"], truth, evaluation["feature_names"],
                       OLD / f"seed{seed}" / target / "predictions.npz")
    for path, digest in expected_hashes.items():
        if sha256(path) != digest:
            raise ValueError(f"Original artifact changed during recovery: {path}")
    reference = truth.mean(0)
    score, standard = pref(truth, predictions, reference), std_r(truth, predictions)
    if not np.isfinite(score) or not np.isfinite(standard):
        raise ValueError("Nonfinite recovered metrics")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions.npz", **evaluation, predictions=predictions,
                        train_mean=reference, predicted_expression_means=expression,
                        reconstructed_control_mean=baseline)
    shutil.copyfile(original / "membership.npz", output / "membership.npz")
    metadata.update({
        "pref": score, "std_r": standard,
        "prediction_recovery": {"protocol": RECOVERY_PROTOCOL, "retrained": False,
            "weights_unchanged": True, "device": "cpu", "seconds": time.monotonic() - start,
            "prediction_shape": list(predictions.shape), "checkpoint_sha256": sha256(checkpoint_file),
            "original_export": str(original), "invalid_original_metrics_discarded": True,
            "recovery_script_sha256": sha256(Path(__file__))},
    })
    write_json(output / "metrics.json", metadata)
    validate_fit(root, target, seed)
    print(f"RECOVERED {target} seed{seed}: {predictions.shape}, pref={score:.5f}, "
          f"{metadata['prediction_recovery']['seconds']:.1f}s", flush=True)


def recover(parent=OUTPUT, root=RECOVERED):
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    parent, root = parent.resolve(), root.resolve()
    verify_inputs(parent)
    inventory = {}
    for target in CELL_LINES:
        for seed in range(5):
            directory = parent / f"seed{seed}" / target
            metadata = json.loads((directory / "metrics.json").read_text())
            paths = [directory / name for name in ("metrics.json", "membership.npz", "predictions.npz")]
            paths.append(Path(metadata["checkpoint"]) / "model.pt")
            inventory[f"{target}/seed{seed}"] = {str(path): sha256(path) for path in paths}
    root.mkdir(parents=True, exist_ok=False)
    # Reuse the original immutable, hash-checked inputs; do not rerun preparation.
    (root / "inputs").symlink_to(parent / "inputs", target_is_directory=True)
    provenance = {"protocol": RECOVERY_PROTOCOL, "parent_results": str(parent),
                  "retrained": False, "created_at": datetime.now().astimezone().isoformat(),
                  "recovery_script_sha256": sha256(Path(__file__)),
                  "original_artifacts_sha256": inventory,
                  "frozen_training_script_sha256": sha256(Path(__file__).with_name("biolord_cellline_go_v1.py")),
                  "report_script_sha256": sha256(Path(__file__).with_name("report_biolord_cellline_go_v1.py"))}
    write_json(root / "recovery_provenance.json", provenance)
    # Tiny matrices are faster with one BLAS thread than with per-query threading.
    torch.set_num_threads(1)
    for target in CELL_LINES:
        for seed in range(5):
            recover_one(parent, root, target, seed, inventory[f"{target}/seed{seed}"])
    verify_inputs(parent)
    report(root)
    write_json(root / "report/recovery_provenance.json", provenance)
    with (root / "report/EXPORT_RECOVERY.md").open("x") as handle:
        handle.write("# Prediction export recovery\n\n"
                     "All 20 models were loaded from their original saved checkpoints. "
                     "No training, optimizer updates, data changes, or scoring changes were made. "
                     "The singleton observation axis is restored before averaging observations, "
                     "preserving all 2,000 gene values per perturbation. Original invalid exports "
                     "and their scores remain untouched and must not be used. Checkpoint and "
                     "original-export hashes are recorded in recovery_provenance.json.\n")
    (root / "recovery_pass").touch(exist_ok=False)
    print(f"RECOVERY_REPORT_PASS {root / 'report'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=OUTPUT)
    parser.add_argument("--root", type=Path, default=RECOVERED)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Full recovery and reporting must run on a SLURM compute node")
    recover(args.parent, args.root)


if __name__ == "__main__":
    main()
