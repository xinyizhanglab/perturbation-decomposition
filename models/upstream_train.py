"""Stage isolated native training workspaces and execute recorded upstream commands."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from reproduction.io import new_directory, path_bindings, read_json, sha256, write_json


PROJECT = Path(__file__).resolve().parents[1]


def _python(args):
    python = Path(args.native_python).absolute()
    if python.parent.name != "bin" or python.parent.parent.name != ".venv" or not python.is_file():
        raise ValueError(
            f"Missing native environment at {python}; run "
            f"uv sync --locked --project models/{args.model} from the repository root."
        )
    return python


def _morph_source(source):
    evidence = read_json(PROJECT / "models/morph/training_source.json")
    actual = {}
    for relative, expected in evidence["files"].items():
        path = Path(source) / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; use the MORPH checkout command in README.md."
            )
        actual[relative] = sha256(path)
        allowed = {expected}
        if relative == "morph/utils.py":
            allowed.add(evidence["recorded_utils_sha256"])
        if actual[relative] not in allowed:
            raise ValueError(f"MORPH source revision mismatch: {relative}")
    return {**evidence, "actual_sha256": actual}


def _runtime_command(python, model, pairs, output, driver=None, check=False):
    command = [str(python), "-B", "-m", "models.training_runtime", "--model", model]
    if driver:
        command += ["--driver", str(driver)]
    if check:
        command += ["--check"]
    if model == "state":
        native = ["tx", "train", *(f"{key}={value}" for key, value in pairs),
                  f"output_dir={output}", "name=run", "use_wandb=false",
                  "training.wandb_track=false"]
    else:
        native = [value for key, text in pairs for value in (f"--{key}", text)]
    return command + ["--", *native]


def _environment(output=None):
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled",
                   "MPLBACKEND": "Agg"}
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT), *filter(None, [environment.get("PYTHONPATH")])]
    )
    if output:
        cache = Path(output) / ".cache"
        environment.update(XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
                           TORCH_HOME=str(cache / "torch"),
                           MPLCONFIGDIR=str(cache / "matplotlib"),
                           WANDB_DIR=str(cache / "wandb"))
    return environment


def check_runtime(args):
    python = _python(args)
    if args.model == "morph":
        _morph_source(args.source_dir)
        driver = Path(args.source_dir).absolute() / "morph/run.py"
    else:
        driver = Path(args.driver).absolute() if args.driver else None
    config = read_json(args.config)
    pairs = _arguments(config["arguments"],
                       {"{data_dir}": str(Path(args.data_dir).absolute())}, set())
    command = _runtime_command(python, args.model, pairs, Path(args.out_dir).absolute(),
                               driver=driver, check=True)
    return subprocess.call(command, cwd=PROJECT, env=_environment())


def _arguments(values, replacements, forbidden):
    result = []
    for key, value in values.items():
        if key in forbidden or any(word in key.lower() for word in ("api_key", "password", "secret")):
            raise ValueError(f"Unsupported native override: {key}")
        text = str(value).lower() if isinstance(value, bool) else str(value)
        for source, target in replacements.items():
            text = text.replace(source, target)
        result.append((key, text))
    return result


def stage_and_run(args):
    from benchmarks._runtime import require_compute
    require_compute("Native training")
    output = Path(args.out_dir).resolve()
    sources = [Path(p).resolve() for p in (args.data_dir, args.source_dir, args.pretrained_dir) if p]
    if any(output.is_relative_to(p) for p in sources):
        raise ValueError("Training output must be outside native source/data roots")
    python = _python(args)
    source_evidence = None
    if args.model == "morph" and not args.source_dir:
        raise ValueError("MORPH requires --source_dir")
    if args.model == "scgpt" and (not args.driver or not args.pretrained_dir):
        raise ValueError("scGPT requires --driver and --pretrained_dir matching the original training setup")
    config = read_json(args.config)
    if args.model == "morph":
        source_evidence = _morph_source(args.source_dir)
    if args.model == "scgpt":
        if not Path(args.driver).is_file():
            raise FileNotFoundError("Install the scGPT input bundles to provide the recorded driver.")
        for filename in ("args.json", "vocab.json", "best_model.pt"):
            if not (Path(args.pretrained_dir) / filename).is_file():
                raise FileNotFoundError(f"Missing scGPT pretrained companion: {filename}")
    forbidden = {"output_dir", "outdir", "data_dir", "scgpt_ckpt", "name", "wandb.entity",
                 "wandb.project", "use_wandb", "training.wandb_track"}
    _arguments(config["arguments"], {}, forbidden)
    data_copy = output / "data"
    source_data, destination_data = Path(args.data_dir), data_copy
    if args.model == "scgpt":
        dataset = config["arguments"]["dataset"]
        if not dataset or Path(dataset).name != dataset or dataset in {".", ".."}:
            raise ValueError("scGPT dataset must be one directory name.")
        source_data = source_data / dataset
        destination_data = data_copy / dataset
    if not source_data.is_dir():
        raise FileNotFoundError(
            f"Missing prepared inputs at {source_data}; install --model {args.model} "
            "with data.training_inputs before training."
        )
    output = new_directory(output)
    shutil.copytree(source_data, destination_data, symlinks=False,
        ignore=shutil.ignore_patterns(".git", ".venv", ".cache", "__pycache__", ".env", ".env.*",
                                     "*.pem", "*.key", "credentials*"))
    replacements = {**path_bindings(args.data_dir, data_copy), "{data_dir}": str(data_copy)}
    if args.model == "morph":
        workspace = output / "native_source"
        for relative in source_evidence["files"]:
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(args.source_dir) / relative, target)
        utils = workspace / "morph/utils.py"
        if sha256(utils) != source_evidence["recorded_utils_sha256"]:
            text = utils.read_text().replace(
                "random.sample(training_candidates, validation_set_size)",
                "random.sample(list(training_candidates), validation_set_size)",
            ).replace(
                "random.sample(val_ptb_list, validation_ood_size)",
                "random.sample(list(val_ptb_list), validation_ood_size)",
            )
            utils.write_text(text)
        if sha256(utils) != source_evidence["recorded_utils_sha256"]:
            raise ValueError("MORPH compatibility copy does not match the recorded loader.")
        data_copy.rename(workspace / "data")
        data_copy = workspace / "data"
        replacements.update(path_bindings(args.source_dir, workspace))
        replacements.update(path_bindings(args.data_dir, data_copy))
        replacements["{data_dir}"] = str(workspace / "data")
        driver = workspace / "morph/run.py"
        if not driver.is_file():
            raise FileNotFoundError("MORPH source must contain morph/run.py")
    else:
        workspace = output
    # Rebase only configuration text; expression matrices and checkpoints are not edited.
    for directory in [data_copy]:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".toml", ".yaml", ".yml", ".csv", ".txt") and path.stat().st_size < 2*1024**2:
                text = path.read_text()
                for source, target in replacements.items():
                    text = text.replace(source, target)
                path.write_text(text)
    pairs = _arguments(config["arguments"], replacements, forbidden)
    if args.model == "scgpt":
        driver = output / "driver.py"
        shutil.copyfile(args.driver, driver)
        pairs += [("data_dir", str(data_copy)),
                  ("scgpt_ckpt", str(Path(args.pretrained_dir).resolve())),
                  ("outdir", str(output / "run"))]
    command = _runtime_command(python, args.model, pairs, output,
                               driver=driver if args.model != "state" else None)
    write_json(output / "launch.json", {"model": args.model, "command": command,
        "config_sha256": sha256(args.config), "driver_sha256": sha256(driver) if args.model != "state" else None,
        "staged_data": str(data_copy), "working_directory": str(workspace),
        "source": source_evidence,
        "environment_project": f"models/{args.model}"})
    environment = _environment(output)
    status = subprocess.call(command, cwd=workspace, env=environment)
    if status == 0:
        selected = []
        for path in output.rglob("*"):
            if path.is_file() and not path.is_relative_to(data_copy) and ".cache" not in path.parts:
                if path.name in ("best_model.pt", "best_model_val.pt", "last_model.pt", "model.pt",
                                 "final.ckpt", "last.ckpt", "predictions.npz", "pert_onehot_map.pt"):
                    selected.append({"path": str(path.relative_to(output)), "size": path.stat().st_size,
                                     "sha256": sha256(path)})
        write_json(output / "outputs.json", {"status": "native_command_completed", "artifacts": selected})
    return status


def main(model, argv=None):
    parser = argparse.ArgumentParser(description=f"Native {model} training with isolated outputs")
    parser.add_argument("--config", required=True, type=Path, help="JSON containing native arguments")
    parser.add_argument("--data_dir", required=True, type=Path, help="Self-contained prepared native data layout")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--native_python", type=Path, default=PROJECT / f"models/{model}/.venv/bin/python")
    parser.add_argument("--source_dir", type=Path, default=PROJECT / "external/MORPH" if model == "morph" else None)
    parser.add_argument("--driver", type=Path, default=PROJECT / "data/inputs/scgpt-pretrained/driver.py" if model == "scgpt" else None)
    parser.add_argument("--pretrained_dir", type=Path, default=PROJECT / "data/inputs/scgpt-pretrained" if model == "scgpt" else None)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--check", action="store_true", help="Check native imports/configuration without loading data or training")
    args = parser.parse_args(argv)
    args.model = model
    try:
        if args.check:
            return check_runtime(args)
        if not args.execute:
            print(json.dumps({key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()}, indent=2))
            return 0
        return stage_and_run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
