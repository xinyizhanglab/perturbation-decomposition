"""Snapshot original Figure 3 inputs and run four full-epoch GPU pilot fits."""

import hashlib
import json
from pathlib import Path
import argparse

import torch

from scgen_author_aligned_v1 import (CELL_LINES, MANIFEST, N_SHARDS, OUTPUT, PROJECT,
                                     PROTOCOL, load_cache, read_tasks, run_one)


BASELINE = PROJECT / ("scripts/paper_figures_june/figure3_crosscl_biolord_corrected_v5_compact/"
                      "fig3_crosscl_transfer_biolord_corrected_v5_compact_results.json")


def prepare(root, log_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("Run the pilot through SLURM with a GPU")
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    tasks = read_tasks()
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    for source, name in ((MANIFEST, "manifest.jsonl"), (BASELINE, "original_figure3.json")):
        with (inputs / name).open("xb") as handle:
            handle.write(source.read_bytes())
    sources = [Path(__file__), Path(__file__).with_name("scgen_author_aligned_v1.py"),
               Path(__file__).with_name("report_scgen_author_aligned_v1.py")]
    provenance = {"protocol": PROTOCOL, "n_shards": N_SHARDS, "n_fits": len(tasks) * 5,
                  "original_manifest": str(MANIFEST), "original_figure3": str(BASELINE),
                  "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  "original_figure3_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
                  "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                  "reference_tutorial": "https://scgen.readthedocs.io/en/stable/tutorials/scgen_perturbation_prediction.html",
                  "benchmark_adaptations": "Original X_hvg, <=128 cells/group, source-observed cohort, observed-control delta and pref"}
    with (inputs / "provenance.json").open("x") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    cache = load_cache()
    for target in CELL_LINES:
        index = next(index for index, task in enumerate(tasks) if task["target"] == target)
        run_one(cache, tasks[index], index, 0, root / "pilot", log_dir)
    (inputs / "pilot_pass").touch(exist_ok=False)
    print(f"PILOT_PASS: four real binary fits, native early stopping and prediction; {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--training-log-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.training_log_dir)
