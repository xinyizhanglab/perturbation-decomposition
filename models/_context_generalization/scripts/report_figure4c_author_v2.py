"""Validate new BioLORD fits, rescore frozen binary scGen, render Figure 4c."""

from portable_paths import (ROOT as PORTABLE_ROOT, PROJECT as PORTABLE_PROJECT, DATA_ROOT, REFERENCE_ROOT, BENCHMARK_ROOT, GO_FILE as PORTABLE_GO_FILE, RESULT_ROOT)

import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from biolord_cellline_go_v1 import MODULE, PLAN, TRAIN, sha256, write_json
from calibration_v1 import SETTINGS, evaluate
from common import CELL_LINES
from figure4c_author_v2 import (
    NA_REASON, PROTOCOL, SCGEN, STEM, assert_training, load_descriptors, task_spec, verify_inputs,
)
from report_scgen_author_aligned_v1 import validate_fit as validate_scgen
from scgen_author_aligned_v1 import read_tasks


METRICS = ("pref", "beta", "gamma")
LABELS = dict(zip(CELL_LINES, ("K562", "RPE1", "HepG2", "Jurkat")))


def aggregate(entries):
    if len(entries) != 5 or sorted(e["seed"] for e in entries) != list(range(5)):
        raise ValueError("Exactly five distinct seed/split repeats are required")
    result = {"status": "available", "n_seeds": 5, "values": entries}
    for metric in METRICS:
        values = [e[metric] for e in entries]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite metrics")
        result[metric] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values, ddof=0))
    return result


def unavailable_scgen():
    return {"status": "not_applicable", "reason": NA_REASON, "n_seeds": 0,
            **{key: None for metric in METRICS for key in (metric, f"{metric}_std")}}


def validate_biolord(root, task, manifest, anova, descriptors):
    target, seed, setting = task_spec(task)
    split = manifest["splits"][target][str(seed)]
    directory = root / f"biolord/{target}/seed{seed}/{setting}"
    meta = json.loads((directory / "metrics.json").read_text())
    for key, expected in {
        "protocol": PROTOCOL, "target": target, "seed": seed, "setting": setting, "task": task,
        "module": {**MODULE, "seed": seed}, "plan": PLAN, "training": TRAIN,
        "control_subtraction": "reconstructed_target_control_mean", "target_test_in_fitting": 0,
        "target_validation_in_fitting": 0, "n_test_perturbations": len(split["test"]),
        "n_target_calibration_means": len(split["calibration"]) if setting == "target30" else 0,
    }.items():
        if meta[key] != expected:
            raise ValueError(f"Invalid BioLORD {key}: {directory}")
    if meta["initial_state_sample_sha256"] == meta["trained_state_sample_sha256"]:
        raise ValueError("BioLORD parameters were not updated")
    if meta["epochs_completed"] != TRAIN["max_epochs"]:
        raise ValueError("Unexpected BioLORD training duration")
    for name in ("predictions", "membership"):
        if sha256(directory / f"{name}.npz") != meta[f"{name}_sha256"]:
            raise ValueError(f"Changed BioLORD {name}")
    if sha256(Path(meta["checkpoint"]) / "model.pt") != meta["checkpoint_sha256"]:
        raise ValueError("Changed checkpoint")
    data = ad.read_h5ad(root / f"inputs/{target}/seed{seed}/{setting}.h5ad")
    assert_training(data, target, split, setting, descriptors)
    with np.load(directory / "membership.npz", allow_pickle=False) as arrays:
        if arrays["condition_ids"].astype(str).tolist() != data.obs_names.tolist():
            raise ValueError("BioLORD condition membership changed")
        for name, label in (("optimizer_indices", "train"), ("validation_indices", "validation")):
            if not np.array_equal(np.sort(arrays[name]), np.flatnonzero(data.obs["split"].to_numpy() == label)):
                raise ValueError("BioLORD optimizer/validation split changed")
        if len(arrays["test_indices"]):
            raise ValueError("Unexpected native test condition means")
    with np.load(directory / "predictions.npz", allow_pickle=False) as arrays:
        if arrays["test_perts"].astype(str).tolist() != split["test"]:
            raise ValueError("BioLORD test cohort changed")
        if not np.array_equal(arrays["feature_names"], anova["feature_names"]):
            raise ValueError("BioLORD features differ from ANOVA truth")
        prediction = arrays["predictions"]
        if arrays["reconstructed_control_mean"].shape != (data.n_vars,):
            raise ValueError("BioLORD singleton-axis export regression")
        if not np.allclose(prediction, arrays["predicted_expression_means"] - arrays["reconstructed_control_mean"]):
            raise ValueError("BioLORD control subtraction changed")
        scores = evaluate(prediction, split["test"], target, anova)
    if not all(np.isclose(scores[key], meta[key], atol=1e-12, rtol=0) for key in METRICS):
        raise ValueError("BioLORD metrics do not recompute")
    return {"seed": seed, "n_test_perturbations": len(split["test"]), **scores}, meta


def rescore_scgen(root, manifest, anova, audit_dir):
    """Use saved binary fits, never a shared VAE or a target30 surrogate."""
    tasks = read_tasks(root / "inputs/scgen_manifest.jsonl")
    lookup = {(task["target"], task["perturbation"]): (index, task) for index, task in enumerate(tasks)}
    result, rows, audit = {}, [], {}
    source_root = Path(manifest["scgen_root"])
    if source_root != SCGEN:
        raise ValueError("Unexpected scGen source root")
    n_fits = 0
    for target in CELL_LINES:
        entries = []
        for seed in range(5):
            split = manifest["splits"][target][str(seed)]
            predictions = []
            for perturbation in split["test"]:
                task_index, task = lookup[(target, perturbation)]
                directory = source_root / f"seed{seed}/{target}/{perturbation}"
                names = ("metadata.json", "membership.npz", "predictions.npz")
                before = {name: sha256(directory / name) for name in names}
                prediction, _, _, features, meta = validate_scgen(directory, task, task_index, seed)
                if any(sha256(directory / name) != checksum for name, checksum in before.items()):
                    raise ValueError("Original scGen artifact changed during validation")
                if not np.array_equal(features, anova["feature_names"]):
                    raise ValueError("scGen features differ from original Figure 4c ANOVA")
                predictions.append(prediction)
                audit[f"{target}/seed{seed}/{perturbation}"] = {
                    "source": str(directory), "sha256": before,
                    "initial_state_sample_sha256": meta["initial_state_sample_sha256"],
                    "trained_state_sample_sha256": meta["trained_state_sample_sha256"],
                }
                n_fits += 1
            predictions = np.concatenate(predictions)
            scores = evaluate(predictions, split["test"], target, anova)
            entry = {"seed": seed, "n_test_perturbations": len(split["test"]), **scores}
            entries.append(entry)
            rows.append({"model": "scgen", "target": target, "setting": "source_only", **entry})
            directory = audit_dir / target / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(directory / "predictions.npz", predictions=predictions,
                                test_perts=np.asarray(split["test"]), feature_names=anova["feature_names"])
            write_json(directory / "metrics.json", {"protocol": PROTOCOL, "retrained": False, **entry})
            print(f"SCGEN_RESCORE_PASS {target} seed{seed}: {scores}", flush=True)
        result[target] = {"source_only": aggregate(entries), "target30": unavailable_scgen()}
    write_json(audit_dir / "source_audit.json", {"retrained": False, "n_validated_binary_fits": n_fits, "fits": audit})
    return result, rows, n_fits


def plot(payload, output):
    rows = []
    for target in CELL_LINES:
        rows.append(("STATE", LABELS[target], [("source_only", payload["state"][target]["zeroshot"]),
                                                ("target30", payload["state"][target]["calibrated"])]))
    for source, target in (("k562", "rpe1"), ("rpe1", "k562"), ("hepg2", "jurkat"), ("jurkat", "hepg2")):
        entry = payload["morph"][f"{source}→{target}"]
        rows.append(("MORPH", f"{LABELS[source]}→{LABELS[target]}",
                     [(key, entry[name]) for key, name in (("source_only", "Source only"),
                      ("controls", "Fine-tune ctrl"), ("target30", "+30% target"))]))
    for model, label in (("biolord", "BioLORD"), ("scgen", "scGen")):
        for target in CELL_LINES:
            rows.append((label, LABELS[target], [(setting, payload[model][target][setting])
                                                 for setting in SETTINGS
                                                 if payload[model][target][setting].get("status") != "not_applicable"]))
    styles = {"source_only": ("#6F5CC4", "o", "No target perturbed cells"),
              "controls": ("#C9CED6", "D", "Target-control fine-tuning"),
              "target30": ("#D26A3A", "o", "+30% target perturbations")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 8.7), sharey=True)
    for axis, metric, title, xlabel in zip(axes, ("pref", "gamma"),
                                         ("Prediction performance", "Interaction recovery"),
                                         ("Perturbed-reference Pearson", "corr(δ̂, γ)")):
        for index, (_, _, entries) in enumerate(rows):
            axis.plot([entry[metric] for _, entry in entries], [index] * len(entries),
                      color="#C9CED6", linewidth=1.2, zorder=1)
            for setting, entry in entries:
                color, marker, _ = styles[setting]
                if entry.get(f"{metric}_std", 0) > 0:
                    axis.errorbar(entry[metric], index, xerr=entry[f"{metric}_std"], fmt="none",
                                  ecolor=color, capsize=2, linewidth=.9)
                axis.scatter(entry[metric], index, s=40, color=color, marker=marker,
                             edgecolor="white", linewidth=.5, zorder=3)
        # N/A is explicitly non-numeric: no orange point, line, or zero score.
        axis.text(.98, 13.5, "+30%: N/A", transform=axis.get_yaxis_transform(),
                  ha="right", va="center", color="#6B7280", fontsize=8)
        axis.set_ylim(len(rows) - .5, -.5)
        axis.set_yticks(range(len(rows)), [])
        axis.axvline(0, color="#1F2937", alpha=.4, linewidth=.6)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel, fontsize=10)
        axis.xaxis.grid(True, color="#E5E7EB", linewidth=.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        for separator in (3.5, 7.5, 11.5):
            axis.axhline(separator, color="#E5E7EB", linewidth=.8)
    for index, (_, label, _) in enumerate(rows):
        axes[0].text(-.06, index, label, transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=8)
    for index, group in enumerate(("STATE", "MORPH", "BioLORD", "scGen")):
        axes[0].text(-.36, 1.5 + index * 4, group, rotation=90,
                     transform=axes[0].get_yaxis_transform(), ha="center", va="center", fontsize=9)
    handles = [Line2D([0], [0], marker=marker, color=color, linestyle="none", label=label,
                      markeredgecolor="white", markersize=6) for color, marker, label in styles.values()]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(.55, .01))
    figure.subplots_adjust(left=.27, right=.98, top=.95, bottom=.11, wspace=.28)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(output / f"{STEM}.{extension}", dpi=250, bbox_inches="tight")
    plt.close(figure)


def report(root):
    manifest = verify_inputs(root, full=True)
    output = root / "report"
    audit_dir = root / "scgen_rescored"
    if output.exists() or audit_dir.exists():
        raise FileExistsError("Refusing to overwrite an existing/incomplete report or scGen rescore")
    with np.load(root / "inputs/anova.npz", allow_pickle=False) as data:
        anova = {key: data[key].copy() for key in data.files}
    descriptors = load_descriptors(root / "inputs/go_features.npz")
    original = json.loads((root / "inputs/original_figure4c.json").read_text())
    if sha256(root / "inputs/original_figure4c.json") != manifest["baseline_sha256"]:
        raise ValueError("Original Figure 4c baseline changed")
    result = {"biolord": {target: {} for target in CELL_LINES}}
    rows, fingerprints = [], {}
    for target in CELL_LINES:
        for setting in SETTINGS:
            entries = []
            for task in range(40):
                cell, _, condition = task_spec(task)
                if (cell, condition) != (target, setting):
                    continue
                entry, meta = validate_biolord(root, task, manifest, anova, descriptors)
                entries.append(entry)
                fingerprints.setdefault((target, setting), set()).add(meta["initial_state_sample_sha256"])
                rows.append({"model": "biolord", "target": target, "setting": setting, **entry})
            result["biolord"][target][setting] = aggregate(entries)
    if any(len(values) != 5 for values in fingerprints.values()):
        raise ValueError("BioLORD repeats lack five distinct initialization fingerprints")
    result["scgen"], scgen_rows, n_scgen = rescore_scgen(root, manifest, anova, audit_dir)
    rows.extend(scgen_rows)
    payload = {**original, **result}
    if any(payload[key] != value for key, value in original.items()):
        raise ValueError("Original STATE/MORPH results changed")
    output.mkdir(exist_ok=False)
    write_json(output / "results.json", payload)
    with (output / "metrics.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "provenance.json", {"protocol": PROTOCOL, "n_new_biolord_fits": 40,
               "n_validated_reused_scgen_binary_fits": n_scgen, "scgen_retrained": False,
               "original_state_morph_unchanged": True, "manifest": manifest})
    plot(payload, output)
    lines = ["# Figure 4c: corrected author-aligned models", "",
             "40 new BioLORD fits validated: GO-neighbor ordered attributes, categorical cell_line, "
             "condition means, unknown_attributes=False, original author configuration; singleton gene-axis fix included.", "",
             "BioLORD pairs use identical source/control samples and source-only inner validation. "
             "target30 adds only the original STATE split's target calibration perturbations to optimizer training. "
             "Target test and original target validation perturbations are excluded from the complete model AnnData.", "",
             f"scGen reuses {n_scgen} completed author-aligned binary fits with observed-control subtraction "
             "and native default latent arithmetic. No scGen models were retrained. " + NA_REASON, "",
             "Scoring matches original Figure 4c: original full-pseudobulk ANOVA truth; target test-cohort "
             "centroid subtracted from truth/prediction for pref; interaction recovery is corr(predicted delta, gamma). "
             "Target truth/centroids/gamma are evaluation-only. Original STATE/MORPH values are unchanged.", "",
             "Mean ± population SD over five paired STATE-calibration split/model-seed repeats, not five-fold CV. "
             "For reused scGen, test membership and fitted initialization both vary between repeats. "
             "Cross-model cell sampling remains different: <=64 cells/group for BioLORD versus the existing "
             "<=128-cell binary scGen fits. Native BioLORD/scGen test cohorts are paired per target/seed; "
             "the original STATE/MORPH cohorts and uncertainty schemes are not globally harmonized.", "",
             "| Model | Target | Setting | pref mean ± SD | gamma mean ± SD |", "|---|---|---|---:|---:|"]
    for model in result:
        for target in CELL_LINES:
            for setting in SETTINGS:
                entry = result[model][target][setting]
                if entry["status"] == "not_applicable":
                    lines.append(f"| {model} | {target} | {setting} | N/A | N/A |")
                else:
                    lines.append(f"| {model} | {target} | {setting} | {entry['pref']:.4f} ± "
                                 f"{entry['pref_std']:.4f} | {entry['gamma']:.4f} ± {entry['gamma_std']:.4f} |")
    (output / "README.md").write_text("\n".join(lines) + "\n")
    (root / "report_pass").touch(exist_ok=False)
    print(f"REPORT_PASS {output}", flush=True)
