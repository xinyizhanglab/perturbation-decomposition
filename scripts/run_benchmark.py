"""Run one benchmark with its normal command-line arguments."""

import argparse
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("within_cl", "cross_cl", "norman_doubles",
                                             "tahoe_drugs", "pathway_prediction"))
    args, rest = parser.parse_known_args()
    module = importlib.import_module(f"benchmarks.{args.benchmark}")
    sys.argv = [f"benchmarks.{args.benchmark}", *rest]
    module.main()


if __name__ == "__main__":
    main()
