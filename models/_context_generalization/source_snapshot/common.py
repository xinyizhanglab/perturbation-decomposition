"""Shared data handling and paper-compatible metrics for context benchmarks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np


CELL_LINES = ("k562", "rpe1", "hepg2", "jurkat")
CONTROL_LABEL = "non-targeting"


def dataset_path(data_root: Path, cell_line: str) -> Path:
    path = data_root / cell_line / "replogle.h5ad"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset for {cell_line}: {path}")
    return path


def valid_perturbations(path: Path, min_cells: int) -> set[str]:
    """Return non-control perturbations with enough cells without loading X."""
    data = ad.read_h5ad(path, backed="r")
    try:
        labels = data.obs["gene"].astype(str).to_numpy()
    finally:
        data.file.close()
    return {
        perturbation
        for perturbation, count in Counter(labels).items()
        if perturbation != CONTROL_LABEL and count >= min_cells
    }


def hvg_names(data: ad.AnnData, n_genes: int) -> list[str]:
    """Obtain the names matching X_hvg, falling back to stable feature names."""
    if "highly_variable" in data.var and int(data.var["highly_variable"].sum()) == n_genes:
        return data.var_names[data.var["highly_variable"].to_numpy()].astype(str).tolist()
    if data.n_vars == n_genes:
        return data.var_names.astype(str).tolist()
    return [f"hvg_{index}" for index in range(n_genes)]


def sample_groups(
    path: Path,
    perturbations: Iterable[str],
    max_cells_per_group: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load bounded X_hvg samples for requested perturbation groups.

    Reading in backed mode keeps each worker from materialising the full raw
    expression matrix.  Sampling is deterministic for a given seed and group.
    """
    requested = set(perturbations)
    data = ad.read_h5ad(path, backed="r")
    try:
        if "X_hvg" not in data.obsm:
            raise KeyError(f"{path} has no obsm['X_hvg']")
        labels = data.obs["gene"].astype(str).to_numpy()
        matrix = data.obsm["X_hvg"]
        rng = np.random.RandomState(seed)
        groups: dict[str, np.ndarray] = {}
        for perturbation in sorted(requested):
            indices = np.flatnonzero(labels == perturbation)
            if len(indices) == 0:
                continue
            if len(indices) > max_cells_per_group:
                indices = np.sort(rng.choice(indices, size=max_cells_per_group, replace=False))
            groups[perturbation] = np.asarray(matrix[indices], dtype=np.float32)
        names = hvg_names(data, next(iter(groups.values())).shape[1]) if groups else []
    finally:
        data.file.close()
    return groups, names


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def std_r(true_deltas: np.ndarray, predicted_deltas: np.ndarray) -> float:
    return float(np.mean([pearson(true, predicted) for true, predicted in zip(true_deltas, predicted_deltas)]))


def pref(true_deltas: np.ndarray, predicted_deltas: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.mean(
            [
                pearson(true - reference, predicted - reference)
                for true, predicted in zip(true_deltas, predicted_deltas)
            ]
        )
    )
