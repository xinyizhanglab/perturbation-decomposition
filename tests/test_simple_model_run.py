"""Model aliases and environment dispatch on tiny fixtures, without native SDKs."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from models import run
from reproduction.io import record_file
from reproduction.predictions import Predictions


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "ridge": "ridge/matched/3to1/k562/seed0/source_only",
    "mlp": "mlp/matched/3to1/k562/seed0/source_only",
    "morph": "morph/within/k562/fold0",
    "state": "state/source_only/k562/seed0",
    "scgpt": "scgpt/within/k562/seed0",
    "biolord": "biolord/transfer/k562/seed0",
}


def manifest(root, records):
    (root / "registry.json").write_text(json.dumps({"schema_version": 1, "records": records}))
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json")
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "version": "fixture",
        "files": [record_file(root, path) for path in paths],
    }))


def released(root, model="ridge", targets=("k562",), repeat=0, checkpoint=True):
    root.mkdir()
    rows = []
    for number, target in enumerate(targets):
        source = root / f"{target}.npz"
        values = np.arange(6).reshape(2, 3).astype(np.float32) + number
        np.savez(source, test_perts=["A", "B"], feature_names=["g0", "g1", "g2"],
                 predictions=values, true_deltas=values + 1)
        rows.append({
            "id": run.identifier(model, target, repeat), "model": model,
            "target": target, "repeat": repeat, "saved": source.name,
            "experiment": "recorded_fixture",
            "inference": {"model": model, "target": target, "seed": repeat,
                          "inputs": source.name, "checkpoint": "model.npz"} if checkpoint else None,
        })
    manifest(root, rows)
    return rows


def binary_scgen(root, targets=("k562",), repeat=0):
    root.mkdir()
    for target in targets:
        values = np.arange(6).reshape(2, 3).astype(np.float32)
        data = Predictions(
            perturbations=np.asarray(["A", "B"]), features=np.asarray(["g0", "g1", "g2"]),
            predicted=values, observed=values + 1, reference=np.ones(3),
            metadata={"model": "scgen", "target": target, "repeat": repeat,
                      "setting": "source_only", "protocol": "complete_native_cohort",
                      "reference_definition": "observed_cohort_mean",
                      "control_subtraction": "observed_target_control",
                      "repeat_kind": "initialization_seed"},
        )
        data.save(root / f"predictions/scgen/{target}/repeat{repeat}")
    (root / "unrelated.npz").write_bytes(b"unrelated large release object")
    manifest(root, [])


def invoke(arguments):
    output = StringIO()
    with redirect_stdout(output):
        code = run.main(arguments)
    return code, output.getvalue()


class SimpleModelRunTests(unittest.TestCase):
    def arguments(self, root, output, model="ridge", target="k562"):
        return ["--model", model, "--target", target, "--results_dir", str(root),
                "--out_dir", str(output)]

    def test_aliases_preserve_all_six_previous_release_examples(self):
        for model, expected in EXAMPLES.items():
            with self.subTest(model=model):
                self.assertEqual(run.identifier(model, "k562", 0), expected)

    def test_repeat_selects_seed_or_fold_and_does_not_fake_state_repeats(self):
        self.assertEqual(run.identifier("morph", "jurkat", 3), "morph/within/jurkat/fold3")
        self.assertEqual(run.identifier("ridge", "rpe1", 2), "ridge/matched/3to1/rpe1/seed2/source_only")
        with self.assertRaisesRegex(ValueError, "repeat 0 only"):
            run.identifier("state", "k562", 1)

    def test_saved_and_portable_inference_use_the_core_python(self):
        for model in run.MODELS:
            self.assertEqual(run.interpreter(model, saved=True), Path(sys.executable).absolute())
        for model in ("ridge", "mlp"):
            self.assertEqual(run.interpreter(model), Path(sys.executable).absolute())

    def test_native_interpreters_are_selected_without_resolving_venv_paths(self):
        for model in ("morph", "state", "scgpt", "biolord", "scgen"):
            self.assertEqual(run.interpreter(model), ROOT / f"models/{model}/.venv/bin/python")

    def test_default_is_a_plan_and_does_not_launch_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files")
            with patch("models.run.subprocess.run") as launch:
                code, text = invoke(self.arguments(base / "files", base / "output"))
            self.assertEqual(code, 0)
            plan = json.loads(text)
            self.assertEqual(plan["selections"][0]["id"], EXAMPLES["ridge"])
            self.assertEqual(plan["device"], "cpu")
            self.assertFalse((base / "output").exists())
            launch.assert_not_called()

    def test_saved_dry_run_does_not_load_or_copy_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files", model="morph")
            with patch("models.run.release_main") as copy:
                code, text = invoke(self.arguments(base / "files", base / "output", "morph")
                                    + ["--saved", "--dry_run"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(text)["saved"])
            copy.assert_not_called()
            self.assertFalse((base / "output").exists())

    def test_saved_all_targets_preserves_the_legacy_prediction_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files", targets=run.TARGETS)
            code, _ = invoke(self.arguments(base / "files", base / "output", target="all") + ["--saved"])
            self.assertEqual(code, 0)
            for target in run.TARGETS:
                with np.load(base / f"files/{target}.npz") as old, \
                        np.load(base / f"output/{target}/predictions.npz") as new:
                    for key in old.files:
                        np.testing.assert_array_equal(old[key], new[key])
            selection = json.loads((base / "output/selection.json").read_text())
            self.assertFalse(selection["checkpoint_inference"])
            self.assertFalse(selection["training_performed"])

    def test_saved_scgen_uses_the_full_native_cohort_without_sdk_or_global_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            binary_scgen(base / "files")
            (base / "files/unrelated.npz").write_bytes(b"not selected, do not scan this")
            with patch.dict(sys.modules, {"scgen": None, "torch": None}), \
                    patch("models.run.subprocess.run") as launch:
                code, _ = invoke(self.arguments(base / "files", base / "output", "scgen") + ["--saved"])
            self.assertEqual(code, 0)
            launch.assert_not_called()
            old = Predictions.load(base / "files/predictions/scgen/k562/repeat0")
            new = Predictions.load(base / "output/k562")
            self.assertEqual(new.metadata, old.metadata)
            for name in ("perturbations", "features", "predicted", "observed", "reference"):
                np.testing.assert_array_equal(getattr(new, name), getattr(old, name))

    def test_tampered_selected_scgen_archive_is_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            binary_scgen(base / "files")
            (base / "files/predictions/scgen/k562/repeat0/predictions.npz").write_bytes(b"tampered")
            with redirect_stderr(StringIO()) as errors, self.assertRaises(SystemExit):
                invoke(self.arguments(base / "files", base / "output", "scgen") + ["--saved"])
            self.assertIn("checksum", errors.getvalue())
            self.assertFalse((base / "output").exists())

    def test_missing_scgen_metadata_is_not_replaced_by_the_rescored_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            binary_scgen(base / "files")
            data = json.loads((base / "files/manifest.json").read_text())
            data["files"] = [row for row in data["files"] if not row["path"].endswith("metadata.json")]
            (base / "files/manifest.json").write_text(json.dumps(data))
            with redirect_stderr(StringIO()) as errors, self.assertRaises(SystemExit):
                invoke(self.arguments(base / "files", base / "output", "scgen") + ["--saved"])
            self.assertIn("complete saved scGen cohort", errors.getvalue())
            self.assertFalse((base / "output").exists())

    def test_missing_weights_are_not_replaced_by_saved_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files", checkpoint=False)
            with redirect_stderr(StringIO()) as errors, self.assertRaises(SystemExit):
                invoke(self.arguments(base / "files", base / "output"))
            self.assertIn("not an inference fallback", errors.getvalue())
            self.assertFalse((base / "output").exists())

    def test_historical_scgen_inference_is_rejected_without_a_new_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stderr(StringIO()) as errors, self.assertRaises(SystemExit):
                invoke(["--model", "scgen", "--out_dir", directory + "/output"])
            self.assertIn("Historical scGen weights", errors.getvalue())

    def test_missing_native_environment_does_not_fall_back_or_install(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files", model="state")
            with patch("models.run.interpreter", return_value=base / "models/state/.venv/bin/python"), \
                    patch("benchmarks._runtime.require_compute"), \
                    patch("models.run.subprocess.run") as launch, \
                    redirect_stderr(StringIO()) as errors, self.assertRaises(SystemExit):
                invoke(self.arguments(base / "files", base / "output", "state") + ["--execute"])
            self.assertIn("uv sync --locked --project models/state", errors.getvalue())
            launch.assert_not_called()
            self.assertFalse((base / "output").exists())

    def test_inference_still_requires_a_compute_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files")
            with patch("benchmarks._runtime.require_compute", side_effect=RuntimeError("allocation required")), \
                    patch("models.run.subprocess.run") as launch, \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                invoke(self.arguments(base / "files", base / "output") + ["--execute"])
            launch.assert_not_called()

    def test_inference_dispatches_the_existing_recipe_with_explicit_trust(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files", model="scgpt", repeat=2)
            python = base / "native/.venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("not executed")
            with patch("models.run.interpreter", return_value=python), \
                    patch("benchmarks._runtime.require_compute"), \
                    patch("models.run.subprocess.run") as launch:
                code, _ = invoke(self.arguments(base / "files", base / "output", "scgpt") + [
                    "--repeat", "2", "--execute", "--trust_checkpoint",
                    "--limit_perts", "2", "--limit_controls", "32",
                    "--batch_size", "4", "--control_batches", "3",
                ])
            self.assertEqual(code, 0)
            command = launch.call_args.args[0]
            self.assertEqual(command[:4], [str(python), "-B", "-m", "models.release"])
            self.assertIn("--trust_checkpoint", command)
            self.assertEqual(command[command.index("--id") + 1], "scgpt/within/k562/seed2")
            for name, value in (("limit_perts", "2"), ("limit_controls", "32"),
                                ("batch_size", "4"), ("control_batches", "3")):
                self.assertEqual(command[command.index("--" + name) + 1], value)
            self.assertEqual(launch.call_args.kwargs["env"]["PYTHONPATH"], str(ROOT))
            self.assertEqual(launch.call_args.kwargs["env"]["WANDB_MODE"], "disabled")

    def test_trust_is_never_enabled_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files", model="biolord")
            code, text = invoke(self.arguments(base / "files", base / "output", "biolord"))
            self.assertEqual(code, 0)
            self.assertNotIn("--trust_checkpoint", json.loads(text)["selections"][0]["arguments"])

    def test_new_scgen_fit_selects_the_native_interpreter_without_a_release_id(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            code, text = invoke([
                "--model", "scgen", "--target", "k562", "--checkpoint", str(base / "fit/checkpoint"),
                "--inputs", str(base / "inputs/request.npz"), "--out_dir", str(base / "output"),
                "--trust_checkpoint",
            ])
            self.assertEqual(code, 0)
            plan = json.loads(text)
            self.assertEqual(plan["interpreter"], str(ROOT / "models/scgen/.venv/bin/python"))
            self.assertEqual(plan["selections"][0]["module"], "models.predict")
            self.assertNotIn("--id", plan["selections"][0]["arguments"])
            self.assertFalse((base / "output").exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files")
            (base / "output").mkdir()
            (base / "output/keep.txt").write_text("preserve")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                invoke(self.arguments(base / "files", base / "output") + ["--saved"])
            self.assertEqual((base / "output/keep.txt").read_text(), "preserve")

    def test_release_metadata_and_recipe_mismatches_are_rejected(self):
        for field in ("target", "inference"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                rows = released(base / "files")
                if field == "target":
                    rows[0]["target"] = "jurkat"
                else:
                    rows[0]["inference"]["seed"] = 4
                manifest(base / "files", rows)
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    invoke(self.arguments(base / "files", base / "output"))
                self.assertFalse((base / "output").exists())

    def test_source_directory_cannot_be_used_for_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            released(base / "files")
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                invoke(self.arguments(base / "files", base / "files/output") + ["--saved"])

    def test_saved_and_inference_cannot_be_requested_together(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run.main(["--model", "ridge", "--out_dir", "unused", "--saved", "--execute"])

    def test_help_needs_no_native_sdk_or_installed_data(self):
        result = subprocess.run([sys.executable, "-B", "-m", "models.run", "--help"],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)
        self.assertNotIn("--id ", result.stdout)


if __name__ == "__main__":
    unittest.main()
