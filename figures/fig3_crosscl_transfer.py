"""Render fig3_crosscl_transfer from saved analysis results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.render import figure_main

if __name__ == "__main__":
    figure_main("fig3_crosscl_transfer")
