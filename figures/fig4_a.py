"""Render paper Figure 4a: within-cell-line template/residual recovery."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.render import figure_main

if __name__ == "__main__":
    figure_main("fig4_a")
