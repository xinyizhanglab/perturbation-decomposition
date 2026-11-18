"""Saved within-cell-line attribution plot with cell-line-template labels."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

STEM = "fig4a_within_decomp_labels_v1"

def plot(payload, OUTPUT):
    source = Path(__file__).with_name("fig4_a.py")
    code = source.read_text().split("fig.savefig(", 1)[0]
    namespace = {"results": payload}
    exec(compile(code, str(source), "exec"), namespace)
    figure, axis = namespace["fig"], namespace["ax"]
    position = axis.get_position().bounds
    points = [collection.get_offsets().copy() for collection in axis.collections]
    replacements = {}
    for artist in axis.get_legend().get_texts():
        old = artist.get_text()
        new = old.replace("template (α)", "cell-line template").replace("residual (ε)", "residual")
        if old == new:
            raise ValueError(f"Unexpected legend entry: {old}")
        replacements[old] = new
        artist.set_text(new)
    if len(replacements) != 6:
        raise ValueError("Expected exactly six changed legend entries")
    for extension in ("pdf", "svg", "png"):
        figure.savefig(OUTPUT / f"{STEM}.{extension}", dpi=300, bbox_inches="tight")
    np.testing.assert_array_equal(axis.get_position().bounds, position)
    for collection, expected in zip(axis.collections, points):
        np.testing.assert_array_equal(collection.get_offsets(), expected)
    plt.close(figure)
