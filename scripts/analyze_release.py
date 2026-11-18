"""Standalone SLURM dispatch for portable analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.__main__ import main

if __name__ == "__main__":
    main(["analyze", *sys.argv[1:]])
