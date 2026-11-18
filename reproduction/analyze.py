"""Recompute independent component matching from portable balanced tensors."""

from pathlib import Path

import numpy as np

from .catalog import CELLS
from .decomposition import compare, decompose, second_moment_fractions, validate_decomposition
from .io import check_output_location, new_directory, read_json, verify, write_json


def component_scores(root, destination):
    """Reaggregate released per-pair correlations, NOT recompute ANOVA tensors.

    This lightweight audit works with the figure bundle and preserves all
    strict-weight/undefined rules of the original independent decomposition.
    """
    root = Path(root)
    verify(root)
    check_output_location(root, destination)
    expected = read_json(root / "final/components/results.json")
    assembly = read_json(root / "final/components/assembly_manifest.json")
    if assembly["contexts"] != list(CELLS):
        raise ValueError("Unexpected component context order")
    summary, n_checked = {}, 0
    with np.load(root / "final/components/component_correlations.npz", allow_pickle=False) as saved:
        expected_keys = set()
        for model, targets in expected.items():
            for target, settings in targets.items():
                ci = list(CELLS).index(target)
                for setting, reference in settings.items():
                    entry = {"n_repeats": 5, "test_sizes": [
                        len(assembly["panels"][str(r)]) for r in range(5)]}
                    for metric in ("alpha", "beta", "gamma"):
                        values = []
                        for repeat in range(5):
                            key = f"{model}__{setting}__repeat{repeat}__{metric}"
                            expected_keys.add(key)
                            pairs = saved[key]
                            p = len(assembly["panels"][str(repeat)])
                            shape = {"alpha": (4,), "beta": (p,), "gamma": (4, p)}[metric]
                            if pairs.shape != shape or np.isinf(pairs).any():
                                raise ValueError(f"Invalid saved component pairs: {key}")
                            value = pairs[ci] if metric == "alpha" else (
                                pairs.mean() if metric == "beta" else pairs[ci].mean())
                            values.append(value)
                        values = np.asarray(values)
                        entry[metric] = float(values.mean())
                        entry[f"{metric}_std"] = float(values.std(ddof=0))
                        entry[f"{metric}_n_defined_repeats"] = int(np.isfinite(values).sum())
                        for field in (metric, f"{metric}_std"):
                            value = reference[field]
                            np.testing.assert_allclose(entry[field], np.nan if value is None else value,
                                                       atol=1e-12, rtol=1e-12, equal_nan=True)
                            n_checked += 1
                        if entry[f"{metric}_n_defined_repeats"] != reference[f"{metric}_n_defined_repeats"]:
                            raise ValueError("Defined repeat counts changed")
                    if entry["test_sizes"] != reference["test_sizes"]:
                        raise ValueError("Saved component cohort sizes changed")
                    summary.setdefault(model, {}).setdefault(target, {})[setting] = entry
        if set(saved.files) != expected_keys:
            raise ValueError("Saved component-pair coverage differs from the reported model/settings")
    output = new_directory(destination)
    write_json(output / "results.json", summary)
    write_json(output / "validation.json", {
        "all_reference_metrics_match": True, "n_reference_statistics_checked": n_checked,
        "mode": "reaggregation_of_saved_pair_correlations",
        "ANOVA_tensors_recomputed": False,
        "note": "Use --analysis matching with an analysis bundle to independently recompute tensor components."})
    return output


def matching(root, destination):
    root = Path(root)
    release = verify(root)
    check_output_location(root, destination)
    if release["profile"] not in ("analysis", "full"):
        raise ValueError("Component recomputation needs an analysis/full bundle, not just figure summaries")
    output = new_directory(destination)
    expected = read_json(root / "final/components/results.json")
    per_group, panels = {}, {}
    for truth_file in sorted((root / "matched").glob("repeat*/truth.npz")):
        repeat = int(truth_file.parent.name.removeprefix("repeat"))
        with np.load(truth_file, allow_pickle=False) as saved:
            panels[repeat] = {k: saved[k].copy() for k in saved.files}
    if sorted(panels) != list(range(5)):
        raise ValueError("Exactly five balanced truth panels required")
    validation = []
    for path in sorted((root / "matched").glob("*/*/repeat*.npz")):
        model, setting = path.parent.parent.name, path.parent.name
        repeat = int(path.stem.removeprefix("repeat"))
        truth = panels[repeat]
        with np.load(path, allow_pickle=False) as saved:
            for name in ("contexts", "perturbations", "feature_names"):
                if not np.array_equal(saved[name], truth[name]):
                    raise ValueError(f"Panel order mismatch: {path}/{name}")
            predicted = saved["predictions"].copy()
        scores, _ = compare(predicted, truth["true_deltas"])
        if truth["contexts"].astype(str).tolist() != list(CELLS):
            raise ValueError("Unexpected context order")
        # Strict means retain undefined values; no nanmean or hidden filtering.
        for i, target in enumerate(CELLS):
            per_group.setdefault((model, target, setting), {})[repeat] = {
                "alpha": float(scores["alpha"][i]), "beta": scores["beta"],
                "gamma": float(scores["gamma"][i]), "n_test": len(truth["perturbations"])}
        components = decompose(predicted)
        validation.append({"model": model, "setting": setting, "repeat": repeat,
                           "reconstruction_error": validate_decomposition(predicted, components)})
    summary = {}
    expected_groups = {(m, t, s) for m, targets in expected.items()
                       for t, settings in targets.items() for s in settings}
    if set(per_group) != expected_groups:
        raise ValueError("Model/target/setting coverage differs from the reference")
    for (model, target, setting), repeats in per_group.items():
        if sorted(repeats) != list(range(5)):
            raise ValueError("Missing/duplicate repeats")
        entry = {"n_repeats": 5, "test_sizes": [repeats[r]["n_test"] for r in range(5)]}
        for metric in ("alpha", "beta", "gamma"):
            values = np.asarray([repeats[r][metric] for r in range(5)])
            entry[metric], entry[f"{metric}_std"] = float(values.mean()), float(values.std(ddof=0))
            entry[f"{metric}_n_defined_repeats"] = int(np.isfinite(values).sum())
            for key in (metric, f"{metric}_std"):
                reference = expected[model][target][setting][key]
                np.testing.assert_allclose(entry[key], np.nan if reference is None else reference,
                                           atol=1e-10, rtol=1e-10, equal_nan=True)
        if entry["test_sizes"] != expected[model][target][setting]["test_sizes"]:
            raise ValueError("Reference panel sizes differ")
        summary.setdefault(model, {}).setdefault(target, {})[setting] = entry
    write_json(output / "results.json", summary)
    write_json(output / "validation.json", {"all_reference_metrics_match": True,
        "prediction_checks": validation, "independent_truth_and_prediction_decomposition": True,
        "uncertainty": "population SD over native-fit/cohort repeats, not CV",
        "undefined_policy": "null; strict equal-weight means"})
    return output


def genetic(root, destination):
    root = Path(root)
    verify(root)
    check_output_location(root, destination)
    tables, features = {}, None
    for cell in CELLS:
        with np.load(root / "genetic" / f"{cell}.npz", allow_pickle=False) as data:
            if features is not None and not np.array_equal(features, data["feature_names"]):
                raise ValueError("Genetic gene identities/order differ across contexts")
            features = data["feature_names"].copy()
            names = data["perturbations"].astype(str).tolist()
            if len(set(names)) != len(names):
                raise ValueError("Duplicate genetic perturbations")
            tables[cell] = dict(zip(names, data["true_deltas"]))
    common = sorted(set.intersection(*(set(table) for table in tables.values())))
    tensor = np.asarray([[tables[cell][p] for p in common] for cell in CELLS])
    components = decompose(tensor)
    error = validate_decomposition(tensor, components)
    output = new_directory(destination)
    np.savez_compressed(output / "components.npz", **components, contexts=np.asarray(CELLS),
                        perturbations=np.asarray(common), feature_names=features)
    write_json(output / "summary.json", {"n_shared": len(common),
        "orthogonal_second_moment_fractions": second_moment_fractions(components),
        "reconstruction_error": error, "protocol": "balanced_observed_ANOVA_no_noise_correction",
        "note": "Not the historical split-half noise-corrected Figure 1f calculation."})
    return output
