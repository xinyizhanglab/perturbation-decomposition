"""Audit all author-aligned binary fits, report scores and render a new Figure 3."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from common import CELL_LINES, CONTROL_LABEL, pref, std_r
from scgen_author_aligned_v1 import OUTPUT, OLD_RESULTS, PROJECT, PROTOCOL, read_tasks


STEM = "fig3_crosscl_scgen_author_aligned_v1"


def validate_fit(directory, task, task_index, seed):
    meta = json.loads((directory / "metadata.json").read_text())
    expected = {"protocol": PROTOCOL, "task_index": task_index, "seed": seed,
                "effective_training_seed": seed + task_index, "sampling_seed": 42,
                "max_cells_per_group": 128, "target": task["target"],
                "sources": task["sources"], "perturbation": task["perturbation"],
                "batch_size": 32, "max_epochs": 100, "early_stopping": True,
                "numpy_compatibility": "temporary_np_Inf_alias_during_native_train",
                "early_stopping_patience": 25, "target_perturbed_cells_in_fitting": 0,
                "target_controls_in_model_anndata": True, "arithmetic": "native_default_all",
                "control_subtraction": "observed_target_control_mean", "old_truth_exact_match": True}
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(f"Invalid {key}: {directory}")
    if meta["initial_state_sample_sha256"] == meta["trained_state_sample_sha256"]:
        raise ValueError("Parameters were not updated")
    if not 1 <= meta["epochs_completed"] <= 100 or not meta["n_validation_cells"]:
        raise ValueError("Invalid training duration or missing validation")
    if "EarlyStopping" not in meta["early_stopping_callback"]:
        raise ValueError("Missing native early stopping")
    with np.load(directory / "membership.npz", allow_pickle=False) as data:
        target = data["cell_line"] == task["target"]
        if not target.any() or not np.all(data["perturbation"][target] == CONTROL_LABEL):
            raise ValueError("Missing target controls or target-perturbed-cell leakage")
        if not np.array_equal(target, data["role"] == "target_control"):
            raise ValueError("Incorrect target-control roles")
        if set(data["cell_line"].astype(str)) != {task["target"], *task["sources"]}:
            raise ValueError("Cell-line identities were collapsed or unexpected")
        for cell in task["sources"]:
            labels = set(data["perturbation"][data["cell_line"] == cell].astype(str))
            if labels != {CONTROL_LABEL, task["perturbation"]}:
                raise ValueError("Non-binary or missing source conditions")
        n = len(target)
        optimizer, validation = data["optimizer_indices"], data["validation_indices"]
        if (set(optimizer) & set(validation) or set(optimizer) | set(validation) != set(range(n))
                or len(optimizer) != meta["n_optimizer_cells"]
                or len(validation) != meta["n_validation_cells"] or n != meta["n_model_cells"]):
            raise ValueError("Invalid native training/validation split")
        actual = [f"{cell}:{row}" for cell, row in zip(data["cell_line"], data["original_row_index"])]
        if len(set(actual)) != n or hashlib.sha256("\n".join(actual).encode()).hexdigest() != meta["training_membership_sha256"]:
            raise ValueError("Corrupt training membership")
        if set(data["original_row_index"][target]) & set(data["evaluation_perturbed_row_indices"]):
            raise ValueError("Target evaluation rows entered model AnnData")
    with np.load(directory / "predictions.npz", allow_pickle=False) as data:
        if data["test_perts"].astype(str).tolist() != [task["perturbation"]]:
            raise ValueError("Wrong predicted perturbation")
        prediction, truth = data["predictions"].copy(), data["true_deltas"].copy()
        features = data["feature_names"].copy()
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid prediction array")
        if not np.allclose(prediction[0], data["predicted_expression_mean"] - data["observed_control_mean"]):
            raise ValueError("Observed-control subtraction changed")
    old_path = OLD_RESULTS / f"seed{seed}" / task["target"] / task["perturbation"] / "predictions.npz"
    with np.load(old_path, allow_pickle=False) as old:
        if not np.array_equal(old["true_deltas"], truth):
            raise ValueError("New and old evaluation truth differs")
        old_prediction = old["predictions"].copy()
    return prediction, truth, old_prediction, features, meta


def collect(root):
    tasks = read_tasks(root / "inputs/manifest.jsonl")
    aggregated = {}
    rows = []
    fit_rows = []
    feature_reference = None
    fingerprints = {}
    for target in CELL_LINES:
        selected = [(index, task) for index, task in enumerate(tasks) if task["target"] == target]
        entries = []
        for seed in range(5):
            predictions, truths, old_predictions = [], [], []
            for index, task in selected:
                directory = root / f"seed{seed}" / target / task["perturbation"]
                p, y, old, features, meta = validate_fit(directory, task, index, seed)
                if feature_reference is not None and not np.array_equal(features, feature_reference):
                    raise ValueError("Prediction features differ across fits")
                feature_reference = features
                fingerprints.setdefault(index, set()).add(meta["initial_state_sample_sha256"])
                predictions.append(p); truths.append(y); old_predictions.append(old)
                fit_rows.append({key: meta[key] for key in ("target", "perturbation", "seed", "seconds", "epochs_completed", "n_model_cells")})
            prediction = np.concatenate(predictions)
            truth = np.concatenate(truths)
            old_prediction = np.concatenate(old_predictions)
            reference = truth.mean(0)
            score = pref(truth, prediction, reference)
            old_score = pref(truth, old_prediction, reference)
            old_meta = json.loads((OLD_RESULTS / f"seed{seed}" / target / "metrics.json").read_text())
            if not np.isclose(old_score, old_meta["pref"], atol=1e-10):
                raise ValueError("Matched old pref differs from saved Figure 3 score")
            entry = {"target": target, "seed": seed, "n_perturbations": len(selected),
                     "pref": score, "std_r": std_r(truth, prediction), "old_pref": old_score}
            entries.append(entry); rows.append(entry)
        values = [entry["pref"] for entry in entries]
        aggregated[target] = {"values": values, "mean": float(np.mean(values)),
                              "std": float(np.std(values)), "n_runs": 5}
    if any(len(values) != 5 for values in fingerprints.values()):
        raise ValueError("Training seeds did not yield five distinct initializations")
    return aggregated, rows, fit_rows


def plot(payload, output):
    sys.path.insert(0, str(PROJECT / "scripts/paper_figures_june"))
    from fig3_crosscl_with_biolord_scgen import MODEL_STYLE, CELL_LABELS

    plt.rcParams.update({"font.family": payload.get("font", "DejaVu Sans"), "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    figure.subplots_adjust(left=.11, right=.57, bottom=.23, top=.96)
    positions = np.arange(4) * 1.2
    offsets = np.linspace(-.38, .38, len(MODEL_STYLE))
    handles = []
    for index, (model, (color, marker, filled, label)) in enumerate(MODEL_STYLE.items()):
        label = model if model in ("BioLORD", "scGen") else label
        for ti, target in enumerate(CELL_LINES):
            entry = payload["results"][model][target]
            position = positions[ti] + offsets[index]
            if entry["std"] is not None:
                axis.errorbar(position, entry["mean"], yerr=entry["std"], fmt="none", ecolor=color, capsize=2, linewidth=1)
            axis.scatter(position, entry["mean"], s=45, marker=marker,
                         facecolors=color if filled else "white", edgecolors="white" if filled else color,
                         linewidths=.5 if filled else 1.2, zorder=3)
        handles.append(Line2D([0], [0], marker=marker, color=color, linestyle="none", label=label,
                              markerfacecolor=color if filled else "white", markersize=6))
    axis.axhline(0, color="#1F2937", linewidth=.5, alpha=.4)
    axis.set_xticks(positions, CELL_LABELS, rotation=45, ha="right", rotation_mode="anchor")
    axis.set_xlim(positions[0] - .62, positions[-1] + .62)
    axis.set_ylabel("Perturbed-reference Pearson")
    axis.yaxis.grid(True, color="#E5E7EB", linewidth=.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(handles=handles, frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.04, .5))
    for extension in ("pdf", "png", "svg"):
        figure.savefig(output / f"{STEM}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def report(root):
    output = root / "report"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    provenance = json.loads((root / "inputs/provenance.json").read_text())
    original_bytes = (root / "inputs/original_figure3.json").read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != provenance["original_figure3_sha256"]:
        raise ValueError("Original Figure 3 snapshot changed")
    if hashlib.sha256((root / "inputs/manifest.jsonl").read_bytes()).hexdigest() != provenance["manifest_sha256"]:
        raise ValueError("Original manifest snapshot changed")
    aggregated, rows, fit_rows = collect(root)
    original = json.loads(original_bytes)
    payload = {**original, "results": {**original["results"], "scGen": aggregated},
               "scgen_protocol": PROTOCOL,
               "layout_revision": {"source": provenance["original_figure3"],
                                   "changed_model": "scGen", "x_label_rotation": 45,
                                   "other_model_metrics_unchanged": True},
               "scgen_metadata": {"n_validated_fits": len(fit_rows), "provenance": provenance}}
    for model in original["results"]:
        if model != "scGen" and payload["results"][model] != original["results"][model]:
            raise ValueError("A non-scGen model changed")
    output.mkdir(exist_ok=False)
    for name, value in (("results.json", payload), ("provenance.json", provenance)):
        with (output / name).open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")
    for name, data in (("metrics.csv", rows), ("fit_timings.csv", fit_rows)):
        with (output / name).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    plot(payload, output)
    lines = ["# Figure 3: author-aligned binary scGen", "",
             f"Validated {len(fit_rows)} separate binary fits. Only scGen scores change; all other model values are preserved.", "",
             "Target controls are included in the model AnnData; target perturbed cells are held out. Actual cell-line labels, native default arithmetic over all registered conditions, and tutorial batch size 32 / up to 100 epochs / early stopping patience 25 are used. The default 90%/10% cell split is retained. Some target controls can be in inner validation.", "",
             "Benchmark-specific adaptations remain: original aligned X_hvg, <=128 cells per group, original source-observed perturbation cohorts, sampled target truth, observed-control response subtraction, and pref. This is not a claim of reproducing the authors' original dataset or scores.", "",
             "Five seeded repeats, not five-fold CV. Cell samples, target truth, perturbation cohorts and pref reference are matched exactly to the old scGen Figure 3 runs. Non-scGen model cohorts remain as before and are not globally harmonized.", "",
             "| Target | Old scGen pref | Author-aligned scGen pref mean ± SD |", "|---|---:|---:|"]
    for target in CELL_LINES:
        old_mean = np.mean([row["old_pref"] for row in rows if row["target"] == target])
        entry = aggregated[target]
        lines.append(f"| {target} | {old_mean:.4f} | {entry['mean']:.4f} ± {entry['std']:.4f} |")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(f"REPORT_PASS {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    report(parser.parse_args().root)
