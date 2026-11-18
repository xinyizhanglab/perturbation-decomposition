"""Explicit-reference metrics with named constant-vector policies."""

import numpy as np


def gene_correlations(a, b, constant="undefined"):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite, identically shaped gene vectors required")
    if constant not in ("undefined", "legacy_zero"):
        raise ValueError("Unknown constant-vector policy")
    ac, bc = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
    an, bn = np.linalg.norm(ac, axis=-1), np.linalg.norm(bc, axis=-1)
    if constant == "legacy_zero":
        valid = (a.std(axis=-1) >= 1e-12) & (b.std(axis=-1) >= 1e-12)
    else:
        valid = ((an > 1e-12 * np.maximum(1, np.linalg.norm(a, axis=-1))) &
                 (bn > 1e-12 * np.maximum(1, np.linalg.norm(b, axis=-1))))
    result = np.full(an.shape, 0. if constant == "legacy_zero" else np.nan)
    np.divide((ac * bc).sum(axis=-1), an * bn, out=result, where=valid)
    return np.clip(result, -1, 1)


def evaluate(predictions):
    predictions.validate()
    return {
        "std_r": float(gene_correlations(predictions.observed, predictions.predicted,
                                          "legacy_zero").mean()),
        "pref": float(gene_correlations(predictions.observed - predictions.reference,
                                         predictions.predicted - predictions.reference,
                                         "legacy_zero").mean()),
        "n_perturbations": len(predictions.perturbations),
        "reference_definition": predictions.metadata["reference_definition"],
    }
