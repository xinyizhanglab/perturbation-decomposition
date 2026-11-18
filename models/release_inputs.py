"""Export portable inference companions from declared read-only source artifacts."""

from pathlib import Path
import pickle

import numpy as np

from data.controls import export as export_controls
from data.distribution import ALIGNMENT, _copy, _source
from data.onehot import ordered_labels
from models.checkpoints import convert_matched
from reproduction.io import safe_path, sha256, write_json


def _asset(record, role):
    matches = [item["path"] for item in record["packaged_assets"] if item["role"] == role]
    if len(matches) != 1:
        raise ValueError(f"{record['id']}: expected one {role} companion")
    return matches[0]


def prepare_model_inputs(records, roots, output):
    from models.artifacts import CELLS
    import torch
    controls, features = {}, {}
    for target, dataset in CELLS.items():
        processed = _source(roots, "release", f"data/processed/genetic/{target}.pkl")
        with processed.open("rb") as stream:
            names = np.asarray(pickle.load(stream)["hvg_names"], dtype=str)
        for kind, relative, options in (
                ("genetic", f"data/{target}/replogle.h5ad", {}),
                ("scgpt", f"data/gears/{dataset}/perturb_processed.h5ad",
                 {"matrix_key": "X", "label_key": "condition", "control_label": "ctrl",
                  "gene_name_key": "gene_name"})):
            directory = output / "controls" / kind / target
            export_controls(_source(roots, "benchmark", relative), directory, **options)
            path = directory / "controls.npz"
            with np.load(path, allow_pickle=False) as archive:
                axis = archive["feature_names"].astype(str)
            if kind == "genetic" and not np.array_equal(axis, names):
                raise ValueError("Raw controls and the existing processed gene axes differ.")
            controls[kind, target] = str(path.relative_to(output))
            features[kind, target] = axis
    descriptor_source = _source(roots, "agop", f"{ALIGNMENT}/inputs/embeddings.npz")
    _copy(descriptor_source, output / "companions/embeddings.npz")
    descriptor_axis = "sha256:" + sha256(descriptor_source)
    with np.load(descriptor_source, allow_pickle=False) as archive:
        descriptors = dict(zip(archive["perturbations"].astype(str), archive["embeddings"]))
    morph_source = _source(roots, "agop", "morph_pipeline/MORPH/morph/model.py")
    _copy(morph_source, output / "upstream/morph/model.py")
    license_path = _source(roots, "agop", "morph_pipeline/MORPH/LICENSE")
    if license_path.is_file():
        _copy(license_path, output / "upstream/morph/LICENSE")
    for record in records:
        if not record["weights_available"]:
            record["inference_unavailable_reason"] = record.get("checkpoint_status", "No historical weights")
            continue
        model, target = record["model"], record["target"]
        with np.load(safe_path(output, record["saved"]), allow_pickle=False) as archive:
            cached = {name: archive[name].copy() for name in archive.files}
        kind = "scgpt" if model == "scgpt" else "genetic"
        names = cached.get("feature_names", features[kind, target]).astype(str)
        if cached["predictions"].shape[1] != len(names):
            raise ValueError(f"{record['id']}: saved responses differ from the verified gene axis.")
        if model not in {"biolord"} and not np.array_equal(names, features[kind, target]):
            raise ValueError(f"{record['id']}: cached and control gene ordering differ.")
        arrays = {"test_perts": cached["test_perts"].astype(str), "feature_names": names}
        for key in ("true_deltas", "train_mean", "reference"):
            if key in cached:
                arrays[key] = cached[key]
        recipe = {"model": model, "checkpoint": _asset(record, "weights"),
                  "inputs": str(Path("inputs") / record["id"] / "inputs.npz"),
                  "seed": record["repeat"], "target": target}
        if model in {"ridge", "mlp", "morph"}:
            labels = arrays["test_perts"]
            if set(labels) - set(descriptors):
                raise ValueError(f"{record['id']}: descriptors are missing for held-out perturbations.")
            arrays["descriptors"] = np.asarray([descriptors[name] for name in labels], dtype=np.float32)
            arrays["features"] = arrays["descriptors"]
            arrays["descriptor_axis_id"] = np.asarray(descriptor_axis)
        if model in {"ridge", "mlp"}:
            axes = output / "axes" / record["id"] / "axes.npz"
            axes.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(axes, feature_names=names, descriptor_axis_id=descriptor_axis)
            portable = output / "portable" / record["id"]
            convert_matched(safe_path(output, recipe["checkpoint"]).parent, model, axes, portable, {
                "target": target, "experiment": record["experiment"], "setting": record["setting"],
                "direction": record["direction"], "repeat": record["repeat"]})
            recipe["checkpoint"] = str(portable.relative_to(output))
        elif model == "morph":
            recipe["architecture"] = record["architecture"]
            if record["architecture"] == "benchmark_mean_conditioned":
                arrays["conditioning_mean"] = cached["train_mean"]
            else:
                recipe["source_dir"] = "upstream/morph"
                recipe["controls"] = controls[kind, target]
        elif model == "state":
            recipe["controls"] = controls[kind, target]
            recipe["perturbation_map"] = _asset(record, "perturbation_map")
            mapping = torch.load(safe_path(output, recipe["perturbation_map"]),
                                 map_location="cpu", weights_only=False)
            arrays["perturbation_labels"] = np.asarray(ordered_labels(mapping), dtype=str)
        elif model == "scgpt":
            recipe["controls"] = controls[kind, target]
            recipe["vocab"] = _asset(record, "vocabulary")
            recipe["model_config"] = _asset(record, "config")
            from models.token_policy import recorded_policy
            policy_path = Path("inputs") / record["id"] / "token_policy.json"
            write_json(safe_path(output, policy_path),
                       recorded_policy(safe_path(output, recipe["vocab"]), names.tolist()))
            recipe["token_policy"] = str(policy_path)
        elif model == "biolord":
            recipe["training_data"] = _asset(record, "training_data")
            recipe["go_features"] = _asset(record, "go_features")
        path = safe_path(output, recipe["inputs"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        record["inference"] = recipe
