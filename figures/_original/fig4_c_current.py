"""Saved annotation-free author-aligned Figure 4c drawing."""
from pathlib import Path
import runpy
from unittest.mock import patch
from matplotlib.axes import Axes

_base = runpy.run_path(str(Path(__file__).with_name("fig4_c.py")))

def plot(payload, output):
    plot = _base["plot"]
    original_text = Axes.text
    removed = []
    def without_annotation(axis, x, y, text, *positional, **kwargs):
        if text == "+30%: N/A":
            removed.append(text)
            return None
        return original_text(axis, x, y, text, *positional, **kwargs)

    with patch.object(Axes, "text", without_annotation):
        plot(payload, output)
    if len(removed) != 2:
        raise RuntimeError(f"Expected two annotations, removed {len(removed)}")

