"""Plot component-analysis outputs without model loading or retraining."""

import argparse
from pathlib import Path

import numpy as np

from reproduction.io import new_directory, read_json, sha256, write_json


def render(results_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures.style import apply_style
    apply_style()
    results_dir = Path(results_dir)
    summary = read_json(results_dir / "summary.json")
    columns = 2 if "matching_components" in summary["results"] else 1
    figure, axes = plt.subplots(1, columns, figsize=(6*columns, 4), squeeze=False)
    figure.suptitle(f"{len(summary['contexts'])} contexts, {summary['n_perturbations']} perturbations")
    axis = axes[0, 0]
    rows = [key for key in ("true_deltas", "predictions") if key in summary["results"]]
    for i, row in enumerate(rows):
        left = 0.
        for component, value in summary["results"][row]["second_moment_fractions"].items():
            if value is not None:
                axis.barh(i, value, left=left, label=component if i == 0 else None)
                left += value
    axis.set_yticks(range(len(rows)), rows)
    axis.set_xlabel("Orthogonal second-moment fraction")
    axis.legend(frameon=False)
    if columns == 2:
        axis = axes[0, 1]
        scores = summary["results"]["matching_components"]
        for offset, name in enumerate(("alpha", "gamma")):
            axis.plot(np.arange(len(summary["contexts"]))+(offset-.5)*.1,
                      [np.nan if v is None else v for v in scores[name]], "o", label=name)
        if scores["beta"] is not None:
            axis.axhline(scores["beta"], label="beta", linestyle="--")
        axis.set_xticks(range(len(summary["contexts"])), summary["contexts"])
        axis.set_ylabel("Independent matching-component Pearson")
        axis.legend(frameon=False)
    figure.tight_layout()
    output = new_directory(output)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(output / f"components.{suffix}", bbox_inches="tight", dpi=150)
    plt.close(figure)
    write_json(output / "plot_data.json", summary)
    write_json(output / "provenance.json", {"input_sha256": sha256(results_dir / "summary.json")})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(render(args.results_dir, args.out_dir))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
