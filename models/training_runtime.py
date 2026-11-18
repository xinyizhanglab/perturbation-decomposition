"""Invoke native training with the recorded compatibility adapters."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import runpy
import sys


def _state_logging_config():
    import state
    from hydra.core.config_store import ConfigStore
    path = Path(state.__file__).parent / "configs/wandb/default.yaml"
    if path.is_file():
        return False
    ConfigStore.instance().store(
        group="wandb", name="default",
        node={"project": "perturbation-decomposition", "entity": None, "local_wandb_dir": "."},
    )
    return True


def run(model, driver, arguments, check=False):
    if model == "state":
        from models.state.compat import install
        compatibility = install()
        logging_fallback = _state_logging_config()
        from state.__main__ import load_hydra_config, main
        if check:
            load_hydra_config("tx", arguments[2:])
            return {"model": model, "configuration": "valid",
                    "arc_state": version("arc-state"), "compatibility": compatibility,
                    "logging_schema_in_memory": logging_fallback}
        sys.argv = ["state", *arguments]
        main()
    else:
        path = Path(driver).absolute()
        sys.path.insert(0, str(path.parent))
        sys.argv = [str(path), *arguments]
        runpy.run_path(str(path), run_name="__native_import_check__" if check else "__main__")
        if check:
            package = "scanpy" if model == "morph" else "scgpt"
            return {"model": model, "imports": "valid", package: version(package),
                    "torch": version("torch"), "driver": str(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("morph", "state", "scgpt"))
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.model != "state" and not args.driver:
        parser.error("--driver is required for MORPH and scGPT")
    result = run(args.model, args.driver, arguments, args.check)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
