"""Small native-setup tests; real training and research datasets are not used."""

import argparse
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from models import training_runtime, upstream_train
from reproduction.io import sha256, write_json


ROOT = Path(__file__).resolve().parents[1]


class NativeTrainingSetupTests(unittest.TestCase):
    def fixture(self, base, model):
        data = base / "data"
        data.mkdir()
        (data / "matrix.h5ad").write_bytes(b"unchanged expression")
        python = base / "runtime/.venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("fixture")
        config = base / "config.json"
        write_json(config, {"arguments": {}})
        return argparse.Namespace(model=model, data_dir=data, out_dir=base / "out",
                                  native_python=python, source_dir=None, pretrained_dir=None,
                                  driver=None, config=config)

    def test_plans_select_the_repository_native_environments(self):
        for model in ("morph", "state", "scgpt"):
            with self.subTest(model=model):
                output = StringIO()
                with redirect_stdout(output), patch("models.upstream_train.stage_and_run") as execute:
                    result = upstream_train.main(model, [
                        "--config", "unused.json", "--data_dir", "unused",
                        "--out_dir", "unused-output",
                    ])
                self.assertEqual(result, 0)
                execute.assert_not_called()
                plan = json.loads(output.getvalue())
                self.assertEqual(plan["native_python"], str(ROOT / f"models/{model}/.venv/bin/python"))
                self.assertFalse(plan["check"])
                if model == "morph":
                    self.assertEqual(plan["source_dir"], str(ROOT / "external/MORPH"))
                if model == "scgpt":
                    self.assertEqual(plan["driver"], str(ROOT / "data/inputs/scgpt-pretrained/driver.py"))

    def test_relative_interpreter_is_absolute_without_resolving_venv_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = self.fixture(base, "state")
            args.native_python.unlink()
            args.native_python.symlink_to(sys.executable)
            args.native_python = Path(os.path.relpath(args.native_python))
            python = upstream_train._python(args)
            self.assertTrue(python.is_absolute())
            self.assertEqual(python.parent.parent.name, ".venv")
            self.assertNotEqual(python, python.resolve())

    def test_missing_environment_names_the_exact_setup_command(self):
        args = argparse.Namespace(native_python=ROOT / "missing/.venv/bin/python", model="morph")
        with self.assertRaisesRegex(ValueError, "uv sync --locked --project models/morph"):
            upstream_train._python(args)

    def test_state_uses_the_compatibility_entrypoint_and_current_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), "state")
            before = (args.data_dir / "matrix.h5ad").read_bytes()
            with patch("benchmarks._runtime.require_compute"), patch("subprocess.call", return_value=0) as run:
                self.assertEqual(upstream_train.stage_and_run(args), 0)
            command = run.call_args.args[0]
            self.assertIn("models.training_runtime", command)
            self.assertIn("output_dir=" + str(args.out_dir.resolve()), command)
            self.assertIn("name=run", command)
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], str(ROOT))
            self.assertEqual((args.data_dir / "matrix.h5ad").read_bytes(), before)

    def test_state_staging_rebases_lexical_and_resolved_input_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            actual = base / "actual"
            actual.mkdir()
            alias = base / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            args = self.fixture(alias, "state")
            config = args.data_dir / "config.toml"
            config.write_text(
                f'lexical="{args.data_dir}/matrix.h5ad"\n'
                f'resolved="{args.data_dir.resolve()}/matrix.h5ad"\n'
            )
            before = config.read_bytes()
            with patch("benchmarks._runtime.require_compute"), patch("subprocess.call", return_value=0):
                self.assertEqual(upstream_train.stage_and_run(args), 0)
            staged = (args.out_dir / "data").resolve()
            self.assertEqual(
                (staged / "config.toml").read_text(),
                f'lexical="{staged}/matrix.h5ad"\nresolved="{staged}/matrix.h5ad"\n',
            )
            self.assertEqual(config.read_bytes(), before)

    def test_check_does_not_stage_data_or_require_an_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), "state")
            args.data_dir = Path(directory) / "not-installed"
            with patch("benchmarks._runtime.require_compute") as require, \
                    patch("subprocess.call", return_value=0) as call:
                self.assertEqual(upstream_train.check_runtime(args), 0)
            require.assert_not_called()
            self.assertIn("--check", call.call_args.args[0])
            self.assertFalse(args.out_dir.exists())

    def test_missing_inputs_fail_before_creating_training_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), "state")
            args.data_dir = Path(directory) / "not-installed"
            with patch("benchmarks._runtime.require_compute"), patch("subprocess.call") as run:
                with self.assertRaisesRegex(FileNotFoundError, "data.training_inputs"):
                    upstream_train.stage_and_run(args)
            self.assertFalse(args.out_dir.exists())
            run.assert_not_called()

    def test_state_import_check_composes_config_but_does_not_train(self):
        module = ModuleType("state.__main__")
        from unittest.mock import Mock
        module.load_hydra_config = Mock()
        module.main = Mock(side_effect=AssertionError("training must not run"))
        with patch("models.state.compat.install", return_value={"fixture": True}), \
                patch("models.training_runtime._state_logging_config", return_value=True), \
                patch.dict(sys.modules, {"state.__main__": module}), \
                patch("models.training_runtime.version", return_value="0.10.2"):
            result = training_runtime.run(
                "state", None, ["tx", "train", "training.train_seed=0"], check=True
            )
        self.assertEqual(result["configuration"], "valid")
        module.load_hydra_config.assert_called_once_with("tx", ["training.train_seed=0"])
        module.main.assert_not_called()

    def test_state_logging_fallback_is_in_memory_only(self):
        from unittest.mock import Mock
        state = ModuleType("state")
        state.__file__ = "/tmp/not-installed-state/__init__.py"
        store = ModuleType("hydra.core.config_store")
        store.ConfigStore = Mock()
        with patch.dict(sys.modules, {"state": state, "hydra.core.config_store": store}):
            self.assertTrue(training_runtime._state_logging_config())
        arguments = store.ConfigStore.instance.return_value.store.call_args.kwargs
        self.assertEqual(arguments["group"], "wandb")
        self.assertEqual(set(arguments["node"]), {"project", "entity", "local_wandb_dir"})
        self.assertIsNone(arguments["node"]["entity"])

    def test_script_import_check_never_uses_main(self):
        for model in ("morph", "scgpt"):
            with self.subTest(model=model), patch.object(sys, "argv", list(sys.argv)), \
                    patch.object(sys, "path", list(sys.path)), \
                    patch("runpy.run_path") as run, \
                    patch("models.training_runtime.version", return_value="fixture"):
                result = training_runtime.run(model, "/tmp/native-driver.py", [], check=True)
                self.assertEqual(result["imports"], "valid")
                self.assertEqual(run.call_args.kwargs["run_name"], "__native_import_check__")

    def test_scgpt_copies_only_the_selected_dataset_and_keeps_driver_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = self.fixture(base, "scgpt")
            selected = args.data_dir / "selected"
            other = args.data_dir / "other"
            for path in (selected, other):
                path.mkdir()
                (path / "cell_graphs.pkl").write_bytes(b"recorded control pairings")
            args.driver = base / "driver.py"
            args.driver.write_text("raise AssertionError('must not train')\n")
            args.pretrained_dir = base / "pretrained"
            args.pretrained_dir.mkdir()
            for name in ("args.json", "vocab.json", "best_model.pt"):
                (args.pretrained_dir / name).write_bytes(b"fixture")
            args.config.write_text(json.dumps({"arguments": {"dataset": "selected"}}))
            with patch("benchmarks._runtime.require_compute"), patch("subprocess.call", return_value=0):
                upstream_train.stage_and_run(args)
            self.assertEqual((args.out_dir / "driver.py").read_bytes(), args.driver.read_bytes())
            self.assertEqual((args.out_dir / "data/selected/cell_graphs.pkl").read_bytes(),
                             (selected / "cell_graphs.pkl").read_bytes())
            self.assertFalse((args.out_dir / "data/other").exists())

    def test_morph_stages_only_verified_code_and_the_recorded_loader_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = self.fixture(base, "morph")
            args.source_dir = base / "source"
            (args.source_dir / "morph").mkdir(parents=True)
            raw = "a = random.sample(training_candidates, validation_set_size)\n"
            patched = "a = random.sample(list(training_candidates), validation_set_size)\n"
            (args.source_dir / "morph/utils.py").write_text(raw)
            (args.source_dir / "morph/run.py").write_text("raise AssertionError('must not train')\n")
            (args.source_dir / ".env").write_text("not copied")
            files = {str(path.relative_to(args.source_dir)): sha256(path)
                     for path in (args.source_dir / "morph").iterdir()}
            evidence = {"files": files,
                        "recorded_utils_sha256": hashlib.sha256(patched.encode()).hexdigest()}
            with patch("benchmarks._runtime.require_compute"), patch("subprocess.call", return_value=0), \
                    patch("models.upstream_train._morph_source", return_value=evidence):
                upstream_train.stage_and_run(args)
            self.assertEqual((args.source_dir / "morph/utils.py").read_text(), raw)
            self.assertEqual((args.out_dir / "native_source/morph/utils.py").read_text(), patched)
            self.assertFalse((args.out_dir / "native_source/.env").exists())
            self.assertEqual((args.out_dir / "native_source/data/matrix.h5ad").read_bytes(),
                             (args.data_dir / "matrix.h5ad").read_bytes())

    def test_unrecognized_morph_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "morph").mkdir()
            (base / "morph/model.py").write_text("unexpected model\n")
            with patch("models.upstream_train.read_json",
                       return_value={"files": {"morph/model.py": "not-the-source-hash"}}):
                with self.assertRaisesRegex(ValueError, "source revision mismatch"):
                    upstream_train._morph_source(base)

    def test_readme_native_training_has_no_external_environment_placeholders(self):
        text = (ROOT / "README.md").read_text()
        native = text.split("MORPH, STATE, and scGPT use", 1)[1].split("**Plot model outputs", 1)[0]
        self.assertNotIn("/path/to/", native)
        self.assertIn("1ad06d4591434cc4ab03c12c461fe0d80b087b11", native)
        self.assertIn("uv sync --locked --project models/morph", text)
        self.assertIn("--check", native)


if __name__ == "__main__":
    unittest.main()
