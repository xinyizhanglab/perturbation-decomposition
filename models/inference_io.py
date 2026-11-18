"""Truth-optional prediction I/O with explicit gene and descriptor axes."""

import numpy as np

from reproduction.io import new_directory, sha256, write_json


def prediction_labels(arrays):
    keys = [key for key in ("test_perts", "perturbations") if key in arrays]
    if not keys:
        raise ValueError("Prediction archives require test_perts or perturbations.")
    names = np.asarray(arrays[keys[0]], dtype=str)
    if len(keys) == 2 and not np.array_equal(names, np.asarray(arrays[keys[1]], dtype=str)):
        raise ValueError("Prediction identifier aliases disagree.")
    if names.ndim != 1 or not len(names) or len(set(names)) != len(names):
        raise ValueError("Prediction perturbation identifiers must be unique and nonempty.")
    return names


def read_inputs(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    if "test_perts" not in arrays:
        raise ValueError("Inputs require ordered test_perts")
    names = arrays["test_perts"].astype(str)
    if names.ndim != 1 or not len(names) or len(set(names)) != len(names):
        raise ValueError("Perturbation names must be unique and nonempty")
    for key in ("features", "descriptors", "controls", "conditioning_mean", "control_mean"):
        if key in arrays and not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite inputs: {key}")
    return arrays


def write_predictions(directory, inputs, predictions, feature_names, metadata):
    predictions = np.asarray(predictions)
    names = np.asarray(feature_names, dtype=str)
    perturbations = inputs["test_perts"].astype(str)
    if names.ndim != 1 or len(set(names)) != len(names):
        raise ValueError("Gene names must be unique and ordered")
    if predictions.shape != (len(perturbations), len(names)) or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite perturbation-by-gene values")
    if "feature_names" in inputs and not np.array_equal(inputs["feature_names"].astype(str), names):
        raise ValueError("Output gene identities/order differ from the prepared input contract")
    arrays = {"predictions": predictions, "test_perts": perturbations, "feature_names": names}
    for key in ("true_deltas", "reference", "train_mean"):
        if key in inputs:
            expected = predictions.shape if key == "true_deltas" else (len(names),)
            if inputs[key].shape != expected or not np.isfinite(inputs[key]).all():
                raise ValueError(f"Invalid optional evaluation array: {key}")
            arrays[key] = inputs[key]
    directory = new_directory(directory)
    np.savez_compressed(directory / "predictions.npz", **arrays)
    write_json(directory / "metadata.json", {**metadata,
        "has_evaluation_truth": "true_deltas" in arrays,
        "predictions_sha256": sha256(directory / "predictions.npz")})
    return directory
