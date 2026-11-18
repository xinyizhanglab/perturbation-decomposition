"""Recover Figure 1b/c plotting values from trusted processed responses."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import pickle

import numpy as np

from reproduction.io import new_directory, sha256, write_json


CELLS = ("k562", "rpe1", "hepg2", "jurkat")


def _effective_rank(M):
    from sklearn.decomposition import PCA
    n_comp = min(100, M.shape[0] - 1, M.shape[1])
    pca = PCA(n_components=n_comp).fit(M)
    ev = pca.explained_variance_
    ev = ev[ev > 1e-12]
    p = ev / ev.sum()
    return np.exp(-np.sum(p * np.log(p)))


def recover(data_dir, destination, trust_processed=False):
    from benchmarks._runtime import require_compute
    require_compute("Figure 1 plotting-input recovery")
    if not trust_processed:
        raise ValueError("Processed pickle inputs require --trust_processed.")
    data_dir, destination = Path(data_dir).resolve(), Path(destination).resolve()
    if destination.is_relative_to(data_dir):
        raise ValueError("Output must be outside the read-only input directory.")
    if destination.exists():
        raise FileExistsError(destination)
    paths = {cell: data_dir / f"{cell}.pkl" for cell in CELLS}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("All four processed genetic response files are required.")
    template, full, residual, inputs = {}, {}, {}, {}
    rng = np.random.get_state()
    for cell, path in paths.items():
        before = path.stat()
        digest = sha256(path)
        with path.open("rb") as stream:
            saved = pickle.load(stream)
        names = sorted(name for name in saved["deltas"]
                       if saved.get("cell_counts", {}).get(name, 5) >= 5)
        D = np.array([saved["deltas"][name] for name in names])
        if D.ndim != 2 or min(D.shape) < 2 or not np.isfinite(D).all():
            raise ValueError(f"Invalid processed response matrix for {cell}.")
        T = D.mean(0)
        total_var = np.mean(np.sum(D ** 2, axis=1))
        template_component = np.mean(np.sum(
            (np.outer(D @ T / (np.dot(T, T) + 1e-30), T)) ** 2, axis=1))
        template[cell] = float(template_component / total_var * 100)
        full[cell] = float(_effective_rank(D))
        Tn = T / (np.linalg.norm(T) + 1e-30)
        alpha = D @ Tn
        R = D - np.outer(alpha, Tn)
        residual[cell] = float(_effective_rank(R))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input changed during recovery: {cell}")
        inputs[cell] = {"sha256": digest, "n_perturbations": len(names),
                        "n_genes": D.shape[1], "dtype": str(D.dtype)}
    if not all(np.isfinite(value) for group in (template, full, residual) for value in group.values()):
        raise ValueError("Recovered plot values are not finite.")
    output = new_directory(destination)
    write_json(output / "fig1b_template_variance_results.json", {"template_var": template})
    write_json(output / "fig1c_erank_results.json", {"erank_full": full, "erank_resid": residual})
    write_json(output / "provenance.json", {
        "inputs": inputs, "source": "Processed all-perturbation responses, minimum five cells.",
        "formula_sources": ["fig1_b.py: template variance block", "fig1_c.py: effective_rank and projection blocks"],
        "numpy": np.__version__, "scikit_learn": version("scikit-learn"),
        "source_sha256": sha256(__file__), "rng_seed_changed": False,
        "initial_numpy_rng_state": {"algorithm": rng[0], "keys": rng[1],
                                    "position": rng[2], "has_gauss": rng[3], "cached_gaussian": rng[4]},
        "PCA_fits": 8, "models_trained": False, "prediction_metrics_recomputed": False,
        "plot_inputs_recomputed": True, "paper_rendering_match_verified": False,
    })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--trust_processed", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"execute": False, "data_dir": str(args.data_dir),
                          "out_dir": str(args.out_dir), "PCA_fits": 8,
                          "models_trained": False, "rng_seed_changed": False}, indent=2))
        return 0
    try:
        try:
            version("scikit-learn")
        except PackageNotFoundError as exc:
            raise RuntimeError("Run uv sync --locked --extra analysis before submitting the recovery job.") from exc
        print(recover(args.data_dir, args.out_dir, args.trust_processed))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
