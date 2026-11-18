"""Draw the supplementary transfer panels from saved inputs."""
from pathlib import Path
import runpy

from figures.original import _prepare


def plot(data, out_dir):
    sources = Path(__file__).resolve().parent
    for identifier, key in (
        ("fig3_supp1_heatmaps", "tables"),
        ("fig3_supp1_slopes", "slopes"),
        ("fig3_supp1_calibration", "calibration"),
        ("fig3_supp1_basis", "tables"),
        ("fig3_supp1_baselines", "tables"),
    ):
        namespace = {**_prepare(identifier, data[key]), "OUT": out_dir}
        runpy.run_path(str(sources / (identifier + ".py")),
                       init_globals=namespace, run_name="preserved_plot")
