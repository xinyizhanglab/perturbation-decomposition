"""Model-neutral prediction schema and native archive adapters.

The reference is explicit: it is NOT necessarily a training centroid.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import read_json, write_json


@dataclass
class Predictions:
    perturbations: np.ndarray
    features: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    reference: np.ndarray
    metadata: dict

    def validate(self):
        p, g = len(self.perturbations), len(self.features)
        for labels in (self.perturbations, self.features):
            if labels.ndim != 1 or len(set(labels.astype(str))) != len(labels):
                raise ValueError("Identifiers must be unique one-dimensional arrays")
        if min(p, g) < 1:
            raise ValueError("Empty prediction axes")
        if self.predicted.shape != (p, g) or self.observed.shape != (p, g):
            raise ValueError("Prediction/truth axes do not match labels")
        if self.reference.shape != (g,):
            raise ValueError("Reference must be a gene vector")
        if any(not np.isfinite(x).all() for x in (self.predicted, self.observed, self.reference)):
            raise ValueError("Nonfinite predictions/truth/reference")
        required = {"model", "target", "setting", "repeat", "protocol",
                    "reference_definition", "control_subtraction", "repeat_kind"}
        if required - self.metadata.keys():
            raise ValueError(f"Missing metadata: {sorted(required - self.metadata.keys())}")
        if self.metadata["model"] == "scgen" and self.metadata["setting"] == "target30":
            raise ValueError("Binary scGen target30 is not applicable")
        return self

    def save(self, directory):
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "predictions.npz",
                            perturbations=self.perturbations.astype(str),
                            feature_names=self.features.astype(str),
                            predictions=self.predicted, true_deltas=self.observed,
                            reference=self.reference)
        write_json(directory / "metadata.json", self.metadata)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with np.load(directory / "predictions.npz", allow_pickle=False) as data:
            result = cls(data["perturbations"], data["feature_names"],
                         data["predictions"], data["true_deltas"], data["reference"],
                         read_json(directory / "metadata.json"))
        return result.validate()


def native_archive(path, metadata, reference=None):
    """Read final native archives; never infer missing gene identities."""
    with np.load(path, allow_pickle=False) as data:
        names = data["feature_names"].astype(str)
        perts = data["test_perts"].astype(str)
        predicted, observed = data["predictions"].copy(), data["true_deltas"].copy()
        if reference is None:
            if "train_mean" not in data:
                raise ValueError("Provide the complete-cohort reference for binary scGen")
            reference = data["train_mean"].copy()
    return Predictions(perts, names, predicted, observed, np.asarray(reference), metadata).validate()


def aggregate_binary(paths, metadata):
    """Aggregate one complete target/repeat cohort, not arbitrary fit subsets."""
    predictions, truths, labels, features = [], [], [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            names = data["feature_names"].astype(str)
            if features is not None and not np.array_equal(features, names):
                raise ValueError("Gene identity/order differs across binary fits")
            features = names
            if data["predictions"].shape != (1, len(names)) or data["true_deltas"].shape != (1, len(names)):
                raise ValueError("A binary fit must contain one complete gene vector")
            labels.extend(data["test_perts"].astype(str))
            predictions.append(data["predictions"].copy())
            truths.append(data["true_deltas"].copy())
    if not predictions:
        raise ValueError("No binary fits")
    observed = np.concatenate(truths)
    return Predictions(np.asarray(labels), features, np.concatenate(predictions), observed,
                       observed.mean(axis=0), metadata).validate()
