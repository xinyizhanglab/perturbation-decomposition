"""Validate all 20 BioLORD GO/cell-line fits and create a new Figure 3."""

from portable_paths import (ROOT as PORTABLE_ROOT, PROJECT as PORTABLE_PROJECT, DATA_ROOT, REFERENCE_ROOT, BENCHMARK_ROOT, GO_FILE as PORTABLE_GO_FILE, RESULT_ROOT)

import csv
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np

from biolord_cellline_go_v1 import (
    CELL_LINES, CONTROL_LABEL, GO_KEY, MODULE, OLD, PLAN, PROJECT, PROTOCOL, TRAIN,
    assert_matches_old, verify_inputs, write_json,
)
from common import pref, std_r


STEM = "fig3_crosscl_biolord_cellline_go_author_v1"


def validate_fit(root, target, seed):
    directory = root / f"seed{seed}" / target
    metadata = json.loads((directory / "metrics.json").read_text())
    expected = {"protocol": PROTOCOL, "target": target, "seed": seed, "sampling_seed": 42,
                "module": {**MODULE, "seed": seed}, "plan": PLAN, "training": TRAIN,
                "categorical_attributes": ["cell_line"], "ordered_attributes": [GO_KEY],
                "unknown_attributes": False, "target_perturbed_cells_in_fitting": 0,
                "old_truth_exact_match": True,
                "control_subtraction": "reconstructed_target_control_mean"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Invalid {key}: {directory}")
    if metadata["initial_state_sample_sha256"] == metadata["trained_state_sample_sha256"]:
        raise ValueError("Model parameters did not change")
    if not 1 <= metadata["epochs_completed"] <= TRAIN["max_epochs"]:
        raise ValueError("Unexpected training duration")
    if not Path(metadata["checkpoint"]).is_dir():
        raise ValueError("Model checkpoint missing")
    data = ad.read_h5ad(root / "inputs" / target / "training.h5ad")
    with np.load(directory / "membership.npz", allow_pickle=False) as member:
        n = len(member["role"])
        target_rows = member["cell_line"] == target
        if (target_rows.sum() != 1 or not np.all(member["perturbation"][target_rows] == CONTROL_LABEL)
                or not np.all(member["role"][target_rows] == "target_control")):
            raise ValueError("Target perturbations leaked into model membership")
        training, validation, test = (member[key] for key in
                                     ("train_indices", "validation_indices", "test_indices"))
        if len(test) or not len(validation) or set(training) & set(validation):
            raise ValueError("Wrong training/validation/test split")
        if set(training) | set(validation) != set(range(n)):
            raise ValueError("Membership does not cover all condition means")
        if (not set(np.flatnonzero(target_rows)).issubset(set(training))
                or np.any(target_rows[validation])):
            raise ValueError("Target controls were not in training only")
        if (len(training) != metadata["n_training_means"]
                or len(validation) != metadata["n_validation_means"]
                or n != metadata["n_condition_means"]):
            raise ValueError("Membership counts differ")
        for column in ("cell_line", "perturbation", "role"):
            if not np.array_equal(member[column], data.obs[column].astype(str).to_numpy()):
                raise ValueError("Actual fit membership differs from prepared inputs")
        if (not np.array_equal(training, np.flatnonzero(data.obs.split == "train"))
                or not np.array_equal(validation, np.flatnonzero(data.obs.split == "validation"))):
            raise ValueError("Fit split differs from prepared inputs")
    with np.load(directory / "predictions.npz", allow_pickle=False) as file:
        arrays = {key: file[key].copy() for key in file.files}
    prediction, truth = arrays["predictions"], arrays["true_deltas"]
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid predictions")
    if not np.array_equal(prediction, (arrays["predicted_expression_means"]
                                      - arrays["reconstructed_control_mean"]).astype(np.float32)):
        raise ValueError("Control subtraction changed")
    assert_matches_old(arrays["test_perts"], truth, arrays["feature_names"],
                       OLD / f"seed{seed}" / target / "predictions.npz")
    with np.load(root / "inputs" / target / "evaluation.npz", allow_pickle=False) as prepared:
        for key in prepared.files:
            if not np.array_equal(arrays[key], prepared[key]):
                raise ValueError("Evaluation differs from prepared inputs")
    if (not np.array_equal(arrays["train_mean"], truth.mean(0))
            or len(prediction) != metadata["n_perturbations"]):
        raise ValueError("Metric reference/cohort changed")
    score, standard = pref(truth, prediction, truth.mean(0)), std_r(truth, prediction)
    if not np.isclose(score, metadata["pref"], atol=1e-12, rtol=0) or not np.isclose(
        standard, metadata["std_r"], atol=1e-12, rtol=0
    ):
        raise ValueError("Saved metrics do not reproduce")
    return metadata


def plot(payload, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from training_style import CELL_LABELS, MODEL_STYLE

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
                axis.errorbar(position, entry["mean"], yerr=entry["std"], fmt="none",
                              ecolor=color, capsize=2, linewidth=1)
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
    provenance = verify_inputs(root)
    original = json.loads((root / "inputs/original_figure3.json").read_text())
    aggregated, rows = {}, []
    for target in CELL_LINES:
        runs = [validate_fit(root, target, seed) for seed in range(5)]
        if len({run["initial_state_sample_sha256"] for run in runs}) != 5:
            raise ValueError("The five seeds do not have distinct initializations")
        values = [run["pref"] for run in runs]
        aggregated[target] = {"values": values, "mean": float(np.mean(values)),
                              "std": float(np.std(values)), "n_runs": 5}
        for run in runs:
            rows.append({key: run[key] for key in ("target", "seed", "pref", "std_r",
                        "epochs_completed", "seconds", "n_perturbations")})
    payload = {**original, "results": {**original["results"], "BioLORD": aggregated},
               "biolord_protocol": PROTOCOL, "biolord_metadata": provenance,
               "layout_revision": {"changed_model": "BioLORD", "other_model_metrics_unchanged": True,
                                   "source": provenance["original_figure3"], "x_label_rotation": 45}}
    for model in original["results"]:
        if model != "BioLORD" and original["results"][model] != payload["results"][model]:
            raise ValueError("A non-BioLORD result changed")
    output.mkdir(exist_ok=False)
    write_json(output / "results.json", payload)
    write_json(output / "provenance.json", provenance)
    with (output / "metrics.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(payload, output)
    lines = ["# BioLORD GO + cell-line Figure 3", "",
             "Validated 20 fits: five initialization seeds per target, not five-fold CV.", "",
             "Author Norman GO-neighbor ordered attributes, condition-mean expression and unknown_attributes=False. "
             "Categorical cell_line is the explicit cross-cell-line extension. Target controls are fitted; "
             "target perturbed expression is excluded from both fitting and validation.", "",
             "Old BioLORD cohorts, sampled target truth, 64-cell cap, aligned X_hvg, reconstructed-control "
             "subtraction, and pref are retained exactly. Other model values, including the completed "
             "author-aligned scGen rerun, are unchanged. Cross-model cohorts/sampling are not harmonized.", "",
             "GO descriptors: GEARS Jaccard >0.1, top 21 including self, then weighted column sum >1. "
             "No expression information enters GO construction. Zero descriptors are retained and reported.", "",
             "| Target | Old BioLORD pref | GO + cell-line pref mean ± SD |", "|---|---:|---:|"]
    for target in CELL_LINES:
        entry = aggregated[target]
        lines.append(f"| {target} | {original['results']['BioLORD'][target]['mean']:.4f} | "
                     f"{entry['mean']:.4f} ± {entry['std']:.4f} |")
    lines.extend(["", "## GO coverage", "", json.dumps(provenance["go_coverage"], indent=2)])
    with (output / "README.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"REPORT_PASS {output}", flush=True)
