"""Plan or explicitly execute a portable native experiment. Never submits jobs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=(
        "biolord", "scgen", "scgen_prepare", "biolord_export_recovery", "figure4c", "batch_recovery"))
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="Read-only source/reference snapshot with agop-relative layout")
    parser.add_argument("--data-root", type=Path, required=True, help="Raw benchmark data directory")
    parser.add_argument("--results-root", type=Path, required=True, help="NEW writable results tree")
    parser.add_argument("--go-file", type=Path)
    parser.add_argument("--execute", action="store_true", help="Requires a SLURM compute allocation")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Native arguments after --")
    args = parser.parse_args(argv)
    protected = [args.reference_root.resolve(), args.data_root.resolve(), args.data_root.resolve().parent]
    results = args.results_root.resolve()
    if any(results.is_relative_to(p) for p in protected):
        parser.error("Results must not be inside a read-only source project")
    scripts = Path(__file__).resolve().parent / "scripts"
    modules = {"biolord": "biolord_cellline_go_v1.py", "scgen": "scgen_author_aligned_v1.py",
               "scgen_prepare": "prepare_scgen_author_aligned_v1.py",
               "biolord_export_recovery": "recover_biolord_cellline_go_export_v2.py",
               "figure4c": "figure4c_author_v2.py", "batch_recovery": "recover_figure4c_batches_v3.py"}
    rest = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    writable_flags = {"--root", "--training-log-dir", "--checkpoint-root"}
    for index, argument in enumerate(rest):
        flag, separator, inline = argument.partition("=")
        if flag in writable_flags:
            value = inline if separator else (rest[index+1] if index+1 < len(rest) else "")
            if not value:
                parser.error(f"{flag} needs a path")
            candidate = Path(value).expanduser().resolve()
            if not candidate.is_relative_to(results):
                parser.error(f"{flag} must remain inside --results-root")
    command = [sys.executable, "-B", str(scripts / modules[args.model]), *rest]
    environment = {**os.environ, "PD_REFERENCE_ROOT": str(args.reference_root.resolve()),
                   "PD_RAW_DATA": str(args.data_root.resolve()), "PD_RESULT_ROOT": str(results),
                   "PYTHONDONTWRITEBYTECODE": "1"}
    from models.prepared_inputs import reference_scgen
    scgen_reference = reference_scgen(args.reference_root)
    if scgen_reference:
        environment["PD_SCGEN_REFERENCE_ROOT"] = str(scgen_reference)
    if args.go_file:
        environment["PD_GO_FILE"] = str(args.go_file.resolve())
    print(json.dumps({"model": args.model, "command": command, "results_root": str(results),
                      "read_only_roots": [str(p) for p in protected], "execute": args.execute,
                      "save_scgen_checkpoints": os.environ.get("PD_SAVE_SCGEN_CHECKPOINTS") == "1",
                      "checkpoint_tasks": os.environ.get("PD_SCGEN_CHECKPOINT_TASKS", "")}, indent=2))
    if args.execute:
        if not os.environ.get("SLURM_JOB_ID"):
            parser.error("--execute requires a SLURM allocation; no jobs are submitted by this command")
        if not rest:
            parser.error("Supply the native task arguments after --")
        return subprocess.call(command, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
