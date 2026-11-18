"""Portable PCA/Ridge/MLP checkpoints with labeled axes and no pickle loading."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reproduction.io import new_directory, read_json, sha256, write_json


def response_axis_id(feature_names, source_contexts):
    payload = json.dumps({"genes": list(map(str, feature_names)), "sources": list(source_contexts),
                          "builder": "mean_available_source_response"}, separators=(",", ":"))
    return "response-axis-sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _names(values, length, label):
    values = np.asarray(values, dtype=str)
    if values.shape != (length,) or len(set(values)) != length:
        raise ValueError(f"{label} must have one unique identifier per axis entry")
    return values


def save_fitted(predict_fn, directory, feature_names, descriptor_axis_id, metadata=None):
    if not hasattr(predict_fn, "checkpoint_arrays"):
        raise ValueError("Predictor does not expose fitted checkpoint state")
    arrays = {k: np.asarray(v) for k, v in predict_fn.checkpoint_arrays.items()}
    info = {**predict_fn.checkpoint_metadata, **(metadata or {})}
    return save_arrays(arrays, info, directory, feature_names, descriptor_axis_id)


def save_arrays(arrays, metadata, directory, feature_names, descriptor_axis_id):
    if not descriptor_axis_id:
        raise ValueError("The descriptor-axis identity is required")
    g = arrays["ridge_coef"].shape[0] if metadata["kind"] == "ridge" else arrays[f"layer{metadata['n_layers']-1}_bias"].shape[0]
    features = _names(feature_names, g, "feature_names")
    arrays = {**arrays, "feature_names": features}
    if not all(np.isfinite(v).all() for k, v in arrays.items() if k != "feature_names"):
        raise ValueError("Checkpoint parameters must be finite")
    pca = arrays["pca_components"]
    if pca.ndim != 2 or arrays["pca_mean"].shape != (pca.shape[1],):
        raise ValueError("Invalid PCA axes")
    directory = new_directory(directory)
    np.savez_compressed(directory / "weights.npz", **arrays)
    write_json(directory / "model.json", {
        **metadata, "schema_version": 1, "descriptor_axis_id": str(descriptor_axis_id),
        "n_input_features": pca.shape[1], "n_output_genes": len(features),
        "projection": "X_components_minus_mean_components",
        "weights_sha256": sha256(directory / "weights.npz")})
    return directory


def save_ensemble(predictors, directory, feature_names, descriptor_axis_id, metadata=None):
    if not predictors:
        raise ValueError("An ensemble needs at least one fitted model")
    directory = new_directory(directory)
    members = []
    for index, fn in enumerate(predictors):
        name = f"member{index}"
        save_fitted(fn, directory / name, feature_names, descriptor_axis_id, metadata)
        members.append({"path": name, "model_sha256": sha256(directory / name / "model.json")})
    write_json(directory / "model.json", {"schema_version": 1, "kind": "ensemble",
        "members": members, "descriptor_axis_id": descriptor_axis_id,
        "feature_names": list(map(str, feature_names)), **(metadata or {})})
    return directory


def save_predictions_with_axes(path, test_perts, predictions, true_deltas, train_mean, feature_names=None):
    arrays = {"test_perts": np.asarray(test_perts, dtype=str), "predictions": predictions,
              "true_deltas": true_deltas, "train_mean": train_mean}
    if feature_names is not None:
        arrays["feature_names"] = _names(feature_names, predictions.shape[1], "feature_names")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def save_estimator(pca, estimator, directory, feature_names, descriptor_axis_id, metadata=None):
    """Export already fitted dataset-specific estimators without refitting."""
    arrays = {"pca_components": pca.components_.copy(), "pca_mean": pca.mean_.copy()}
    if hasattr(estimator, "coef_"):
        arrays.update(ridge_coef=estimator.coef_.copy(),
                      ridge_intercept=np.asarray(estimator.intercept_).copy())
        info = {"kind": "ridge"}
    else:
        import torch.nn as nn
        layers = [layer for layer in estimator.net if isinstance(layer, nn.Linear)]
        if not layers:
            raise ValueError("Unsupported fitted neural estimator")
        for index, layer in enumerate(layers):
            arrays[f"layer{index}_weight"] = layer.weight.detach().cpu().numpy().copy()
            arrays[f"layer{index}_bias"] = layer.bias.detach().cpu().numpy().copy()
        info = {"kind": "mlp", "n_layers": len(layers)}
    return save_arrays(arrays, {**info, **(metadata or {})}, directory,
                       feature_names, descriptor_axis_id)


class Predictor:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata = read_json(self.directory / "model.json")
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint version")
        self.kind = self.metadata["kind"]
        if self.kind == "ensemble":
            from reproduction.io import safe_path
            members = self.metadata["members"]
            if not members:
                raise ValueError("Empty ensemble")
            self.members = []
            for entry in members:
                member = safe_path(self.directory, entry["path"])
                if sha256(member / "model.json") != entry["model_sha256"]:
                    raise ValueError("Ensemble member metadata changed")
                self.members.append(Predictor(member))
            self.feature_names = np.asarray(self.metadata["feature_names"])
            for member in self.members:
                if (not np.array_equal(member.feature_names, self.feature_names) or
                        member.metadata["descriptor_axis_id"] != self.metadata["descriptor_axis_id"]):
                    raise ValueError("Ensemble axes differ")
        elif self.kind in ("ridge", "mlp"):
            if sha256(self.directory / "weights.npz") != self.metadata["weights_sha256"]:
                raise ValueError("Checkpoint checksum mismatch")
            with np.load(self.directory / "weights.npz", allow_pickle=False) as saved:
                self.arrays = {k: saved[k].copy() for k in saved.files}
            self.feature_names = self.arrays["feature_names"].astype(str)
        else:
            raise ValueError("Unsupported predictor kind")

    def __call__(self, values, descriptor_axis_id=None):
        if descriptor_axis_id != self.metadata["descriptor_axis_id"]:
            raise ValueError("Descriptor-axis identity differs from the trained checkpoint")
        values = np.asarray(values)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Descriptors must be a finite row-by-feature matrix")
        if self.kind == "ensemble":
            outputs = [member(values, descriptor_axis_id) for member in self.members]
            output = np.zeros_like(outputs[0])
            for member in outputs:
                output += member / len(outputs)
            return output
        a = self.arrays
        if values.shape[1] != self.metadata["n_input_features"]:
            raise ValueError("Descriptor width differs from the checkpoint")
        projected = values @ a["pca_components"].T
        projected -= a["pca_mean"].reshape(1, -1) @ a["pca_components"].T
        if self.kind == "ridge":
            output = projected @ a["ridge_coef"].T + a["ridge_intercept"]
        else:
            output = projected.astype(np.float32)
            for i in range(self.metadata["n_layers"]):
                output = output @ a[f"layer{i}_weight"].T + a[f"layer{i}_bias"]
                if i + 1 != self.metadata["n_layers"]:
                    output = np.maximum(output, 0)
        if output.shape != (len(values), len(self.feature_names)) or not np.isfinite(output).all():
            raise ValueError("Invalid checkpoint prediction")
        return output


def convert_matched(checkpoint_dir, model, axes, output, metadata=None):
    checkpoint_dir = Path(checkpoint_dir)
    with np.load(checkpoint_dir / "ridge_pca.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    with np.load(axes, allow_pickle=False) as data:
        names = data["feature_names"].astype(str)
        axis_id = str(data["descriptor_axis_id"].item())
    info = {"kind": model, "source_format": "matched_transfer_ridge_pca_mlp",
            "source_sha256": {"ridge_pca.npz": sha256(checkpoint_dir / "ridge_pca.npz")},
            "axes_sha256": sha256(axes), **(metadata or {})}
    if model == "mlp":
        import torch
        info["source_sha256"]["mlp.pt"] = sha256(checkpoint_dir / "mlp.pt")
        state = torch.load(checkpoint_dir / "mlp.pt", map_location="cpu", weights_only=True)
        arrays.pop("ridge_coef")
        arrays.pop("ridge_intercept")
        indices = sorted(int(k.split(".")[0]) for k in state if k.endswith(".weight"))
        for index, layer in enumerate(indices):
            arrays[f"layer{index}_weight"] = state[f"{layer}.weight"].cpu().numpy()
            arrays[f"layer{index}_bias"] = state[f"{layer}.bias"].cpu().numpy()
        info["n_layers"] = len(indices)
    return save_arrays(arrays, info, output, names, axis_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True, type=Path)
    parser.add_argument("--model", choices=("ridge", "mlp"), required=True)
    parser.add_argument("--axes", type=Path, required=True,
                        help="NPZ with feature_names and descriptor_axis_id")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(convert_matched(args.checkpoint_dir, args.model, args.axes, args.out_dir))


if __name__ == "__main__":
    main()
