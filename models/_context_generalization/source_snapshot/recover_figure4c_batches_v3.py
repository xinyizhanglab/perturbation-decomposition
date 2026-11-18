"""Recover only failed Figure 4c fits, merging a trailing singleton minibatch.

Frozen v2 training/report sources, successful fits, all split memberships,
and the nominal author batch size remain unchanged. No samples are dropped
or repeated: an otherwise final singleton is appended to the preceding batch
(32 + 1 becomes 33). Every complete optimizer-data pass is audited.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

import figure4c_author_v2 as original
from biolord_cellline_go_v1 import TRAIN, sha256, write_json


OUTPUT = original.ROOT / "results/figure4c_author_aligned_v2_retry1"
TASKS = (5, 25)
BATCH_PROTOCOL = "merge_final_singleton_into_previous_batch_no_drop_no_repeat_v3"


class MergeSingletonSampler(Sampler):
    """Wrap native BatchSampler without altering its shuffle/permutation."""

    def __init__(self, sampler):
        self.native = sampler
        self.n_samples = len(sampler.sampler)
        self.batch_size = sampler.batch_size
        self.drop_last = sampler.drop_last
        self.audit = []
        if self.n_samples < 2 or self.batch_size < 2 or self.drop_last:
            raise ValueError("Require >=2 samples, batch size >=2 and native drop_last=False")

    def __len__(self):
        return len(self.native) - int(self.n_samples % self.batch_size == 1)

    def __iter__(self):
        pending, seen, sizes = None, [], []
        for batch in self.native:
            batch = list(batch)
            if pending is None:
                pending = batch
                continue
            if len(batch) == 1:
                pending += batch
            else:
                sizes.append(len(pending))
                seen.extend(pending)
                yield pending
                pending = batch
        if pending is not None:
            sizes.append(len(pending))
            seen.extend(pending)
        if sorted(seen) != list(range(self.n_samples)) or min(sizes) < 2:
            raise ValueError("Optimizer pass dropped/repeated samples or retained a singleton")
        self.audit.append({"n_samples": len(seen), "n_distinct_samples": len(set(seen)),
                           "n_batches": len(sizes), "min_batch_size": min(sizes),
                           "max_batch_size": max(sizes),
                           "ordered_indices_sha256": hashlib.sha256(
                               np.asarray(seen, dtype=np.int64).tobytes()).hexdigest()})
        # Lightning stops after len(loader) batches without necessarily asking
        # for one extra item/StopIteration. Audit before issuing the final batch.
        if pending is not None:
            yield pending


def safe_loader(native):
    if native.num_workers != 0:
        raise ValueError("Recovery expects the frozen zero-worker loader")
    sampler = MergeSingletonSampler(native.sampler)
    loader = DataLoader(native.dataset, batch_size=None, sampler=sampler,
                        num_workers=0, collate_fn=native.collate_fn, pin_memory=native.pin_memory)
    return loader, sampler


@contextmanager
def patched_training_batches():
    """Patch only this process's splitter binding; never edit installed code."""
    import biolord._model as native

    parent = native.AnnDataSplitter
    samplers = []

    class SingletonSafeSplitter(parent):
        def train_dataloader(self):
            loader, sampler = safe_loader(super().train_dataloader())
            samplers.append(sampler)
            return loader

    with patch.object(native, "AnnDataSplitter", SingletonSafeSplitter):
        yield samplers


def protected_artifacts(root):
    """Hash only the 38 already completed fits, excluding both recovery tasks."""
    hashes, count = {}, 0
    for path in sorted((root / "biolord").glob("*/seed*/*/metrics.json")):
        meta = json.loads(path.read_text())
        if meta["task"] in TASKS:
            continue
        count += 1
        for name in ("metrics.json", "predictions.npz", "membership.npz"):
            file = path.parent / name
            hashes[str(file.resolve())] = sha256(file)
        checkpoint = Path(meta["checkpoint"]) / "model.pt"
        hashes[str(checkpoint)] = sha256(checkpoint)
    if count != 38:
        raise ValueError(f"Expected 38 protected successful fits, found {count}")
    return hashes


def verify_protected(hashes):
    if any(sha256(Path(path)) != checksum for path, checksum in hashes.items()):
        raise ValueError("A protected successful-fit artifact changed")


def recover_fit(root, task, log_dir):
    if task not in TASKS:
        raise ValueError("Only the two failed tasks 5 and 25 may be recovered")
    protected = protected_artifacts(root)
    with patched_training_batches() as samplers:
        original.fit(root, task, log_dir)
    passes = [audit for sampler in samplers for audit in sampler.audit]
    if len(passes) < TRAIN["max_epochs"]:
        raise ValueError("Fewer fully audited training passes than requested epochs")
    if any(p["min_batch_size"] < 2 or p["max_batch_size"] != TRAIN["batch_size"] + 1
           or p["n_samples"] != p["n_distinct_samples"] for p in passes):
        raise ValueError("Invalid merged-batch audit")
    verify_protected(protected)
    target, seed, setting = original.task_spec(task)
    directory = root / f"biolord/{target}/seed{seed}/{setting}"
    write_json(directory / "batching_recovery.json", {
        "protocol": BATCH_PROTOCOL, "task": task, "target": target, "seed": seed, "setting": setting,
        "nominal_batch_size": TRAIN["batch_size"], "final_batch_size": TRAIN["batch_size"] + 1,
        "samples_dropped": 0, "samples_repeated": 0,
        "model_parameters_splits_and_scoring_unchanged": True,
        "recovery_source_sha256": sha256(Path(__file__)),
        "protected_artifacts_sha256": protected, "complete_optimizer_passes": passes,
    })
    print(f"BATCH_RECOVERY_PASS task{task}: {len(passes)} audited passes; all 38 fits unchanged", flush=True)


def recovery_report(root):
    original.verify_inputs(root, full=True)
    records = {}
    for task in TASKS:
        target, seed, setting = original.task_spec(task)
        path = root / f"biolord/{target}/seed{seed}/{setting}/batching_recovery.json"
        record = json.loads(path.read_text())
        if (record["protocol"] != BATCH_PROTOCOL or record["task"] != task
                or record["recovery_source_sha256"] != sha256(Path(__file__))
                or record["samples_dropped"] != 0 or record["samples_repeated"] != 0):
            raise ValueError("Invalid recovery provenance")
        meta = json.loads((path.parent / "metrics.json").read_text())
        with np.load(path.parent / "membership.npz", allow_pickle=False) as membership:
            n_training = len(membership["optimizer_indices"])
        passes = record["complete_optimizer_passes"]
        if len(passes) < meta["epochs_completed"] or n_training % TRAIN["batch_size"] != 1:
            raise ValueError("Invalid recovery epoch/sample count")
        if any(p["n_samples"] != n_training or p["n_distinct_samples"] != n_training
               or p["min_batch_size"] < 2 or p["max_batch_size"] != 33 for p in passes):
            raise ValueError("Invalid complete optimizer-data coverage")
        verify_protected(record["protected_artifacts_sha256"])
        records[str(task)] = {"path": str(path), "sha256": sha256(path), "n_training_means": n_training,
                              "n_audited_optimizer_passes": len(passes)}
    from report_figure4c_author_v2 import report
    report(root)
    write_json(root / "report/batching_recovery_provenance.json", {
        "protocol": BATCH_PROTOCOL, "recovery_source_sha256": sha256(Path(__file__)),
        "recovered_tasks": records, "n_successful_fits_preserved": 38,
        "nominal_batch_size": 32, "merged_final_batch_size": 33,
        "model_parameters_splits_and_scoring_unchanged": True,
    })
    (root / "report/BATCHING_RECOVERY.md").write_text(
        "# Singleton minibatch recovery\n\n"
        "Only K562 seed2 target30 and HepG2 seed2 target30 were rerun. The native "
        "batch sampler's final singleton was merged into its preceding batch "
        "(32 + 1 = 33). All other minibatches retain nominal size 32. No training "
        "condition means were dropped or repeated; every complete optimizer-data "
        "pass was audited. Author model/configuration, BatchNorm, train/validation "
        "memberships, initialization seeds, epochs, inference and scoring were "
        "unchanged. SHA256 checks confirm all 38 existing fits and checkpoints "
        "were preserved. No installed package or frozen v2 source was edited.\n")
    (root / "batching_recovery_pass").touch(exist_ok=False)
    print(f"BATCH_RECOVERY_REPORT_PASS {root / 'report'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "report"))
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--task", type=int)
    parser.add_argument("--training-log-dir", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Recovery fitting and full report require SLURM")
    if args.action == "train":
        if args.task is None or args.training_log_dir is None:
            parser.error("train requires --task and --training-log-dir")
        recover_fit(args.root, args.task, args.training_log_dir)
    else:
        recovery_report(args.root)


if __name__ == "__main__":
    main()
