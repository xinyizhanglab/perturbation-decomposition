"""Recover missing plotting numbers in a prepared analysis allocation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.main_figure_inputs import main


if __name__ == "__main__":
    raise SystemExit(main())
