"""Independent balanced-panel ANOVA; no fitted-model dependencies."""

import numpy as np

from .metrics import gene_correlations


def decompose(tensor):
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim != 3 or min(tensor.shape[:2]) < 2 or tensor.shape[-1] < 2:
        raise ValueError("Need a balanced context × perturbation × gene tensor")
    if not np.isfinite(tensor).all():
        raise ValueError("Missing/nonfinite tensor entries cannot be filled or dropped")
    mu = tensor.mean(axis=(0, 1))
    alpha = tensor.mean(axis=1) - mu
    beta = tensor.mean(axis=0) - mu
    gamma = tensor - mu - alpha[:, None, :] - beta[None, :, :]
    return {"mu": mu, "alpha": alpha, "beta": beta, "gamma": gamma}


def reconstruct(components):
    return (components["mu"] + components["alpha"][:, None, :] +
            components["beta"][None, :, :] + components["gamma"])


def validate_decomposition(tensor, components):
    np.testing.assert_allclose(reconstruct(components), tensor, atol=1e-10, rtol=1e-10)
    for values in (components["alpha"].mean(axis=0), components["beta"].mean(axis=0),
                   components["gamma"].mean(axis=0), components["gamma"].mean(axis=1)):
        np.testing.assert_allclose(values, 0, atol=1e-10, rtol=0)
    return float(np.max(np.abs(reconstruct(components) - tensor)))


def compare(predicted_tensor, observed_tensor):
    if np.shape(predicted_tensor) != np.shape(observed_tensor):
        raise ValueError("Predicted and observed panels must match exactly")
    predicted, observed = decompose(predicted_tensor), decompose(observed_tensor)
    validate_decomposition(predicted_tensor, predicted)
    validate_decomposition(observed_tensor, observed)
    pairs = {name: gene_correlations(predicted[name], observed[name])
             for name in ("alpha", "beta", "gamma")}
    return ({"alpha": pairs["alpha"], "beta": float(pairs["beta"].mean()),
             "gamma": pairs["gamma"].mean(axis=1)}, pairs)


def second_moment_fractions(components):
    """Orthogonal energy fractions, including mu; not misleading scalar variances."""
    energy = {name: float(np.mean(value ** 2)) for name, value in components.items()}
    total = sum(energy.values())
    return {name: value / total if total else None for name, value in energy.items()}


def remove_template(responses, template, method):
    responses, template = np.asarray(responses), np.asarray(template)
    if responses.shape[-1:] != template.shape or not np.isfinite(template).all():
        raise ValueError("Template must match the gene axis")
    if method == "subtract":
        return responses - template
    if method == "project":
        norm = np.linalg.norm(template)
        if norm <= 1e-12:
            raise ValueError("Cannot project onto a zero template")
        direction = template / norm
        return responses - (responses @ direction)[..., None] * direction
    raise ValueError("Select 'subtract' or 'project' explicitly")
