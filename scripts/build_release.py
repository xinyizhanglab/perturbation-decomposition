"""Standalone SLURM dispatch for the reproduction release builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproduction.__main__ import main

if __name__ == "__main__":
    main(["build", *sys.argv[1:]])
