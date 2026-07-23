"""Shared evaluation metrics for perturbation prediction."""
import numpy as np


def pearson(a, b):
    """Pearson correlation, safe for constant vectors."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def std_r(true_deltas, pred_deltas):
    """Mean per-perturbation Pearson (standard correlation).
    Inflated by template — use pref as primary metric."""
    return float(np.mean([
        pearson(true_deltas[i], pred_deltas[i])
        for i in range(len(true_deltas))
    ]))


def pref(true_deltas, pred_deltas, train_centroid):
    """Perturbed-reference Pearson (template-removed).
    Primary metric: measures perturbation-specific prediction accuracy.

    Args:
        true_deltas: (n_perts, n_genes) ground truth
        pred_deltas: (n_perts, n_genes) predictions
        train_centroid: (n_genes,) mean delta of TRAINING perturbations
    """
    return float(np.mean([
        pearson(true_deltas[i] - train_centroid, pred_deltas[i] - train_centroid)
        for i in range(len(true_deltas))
    ]))


def centroid_accuracy(true_deltas, pred_deltas, pert_names):
    """Fraction of perturbations whose predicted centroid is closest
    to the correct true centroid (1-NN classification)."""
    n = len(pert_names)
    if n < 2:
        return 0.0
    correct = 0
    for i in range(n):
        dists = [np.linalg.norm(pred_deltas[i] - true_deltas[j]) for j in range(n)]
        if np.argmin(dists) == i:
            correct += 1
    return correct / n


def median_rank(true_deltas, pred_deltas, pert_names):
    """Median rank of the correct perturbation when sorting by distance
    to the predicted centroid."""
    n = len(pert_names)
    if n < 2:
        return n
    ranks = []
    for i in range(n):
        dists = [np.linalg.norm(pred_deltas[i] - true_deltas[j]) for j in range(n)]
        rank = np.argsort(dists).tolist().index(i) + 1
        ranks.append(rank)
    return float(np.median(ranks))


def evaluate_predictions(npz_path):
    """Load a predictions.npz and compute all metrics."""
    d = np.load(npz_path, allow_pickle=True)
    yt = d["true_deltas"]
    yp = d["predictions"]
    T = d["train_mean"]
    perts = list(d["test_perts"])

    return {
        "n_perts": len(perts),
        "std_r": std_r(yt, yp),
        "pref": pref(yt, yp, T),
        "centroid_accuracy": centroid_accuracy(yt, yp, perts),
        "median_rank": median_rank(yt, yp, perts),
    }


def save_predictions(path, test_perts, predictions, true_deltas, train_centroid):
    """Save predictions in the standard format."""
    np.savez(path,
             test_perts=np.array(test_perts),
             predictions=predictions,
             true_deltas=true_deltas,
             train_mean=train_centroid)
