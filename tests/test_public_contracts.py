"""Documentation layout and protected scientific implementation contracts."""

import hashlib
import io
from contextlib import redirect_stderr
from pathlib import Path
import re
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
README_HEADINGS = [
    "#  Response decomposition enables biologically aligned generalization to unseen perturbations and cellular contexts",
    "## Quick start",
    "## Data",
    "## Pre-computed results",
    "## Main analyses",
    "### Decomposition",
    "### Hallmark and pathway figures",
    "### Other response figures",
    "## Models",
    "### External model setup",
    "## Metrics",
    "## Repository structure",
    "## Citation",
    "## License",
]
PROTECTED = {
    "evaluation/metrics.py": "e1e4d1a69a8db0ff4474f6fb6a4252d4810cf5ed2abf416224e840d84a4e2964",
    "decomposition/anova.py": "3c8e26a5fd8a33e7a24627cd07b39acdf4bba7f07cc2fd19ec62adcece0d7d4d",
}


def markdown_headings(text):
    result, fenced = [], False
    for line in text.splitlines():
        if line.startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced and re.match(r"^#{1,3} ", line):
            result.append(line)
    return result


class PublicContractTests(unittest.TestCase):
    def test_readme_orders_data_and_model_free_analyses_before_models(self):
        self.assertEqual(markdown_headings((ROOT / "README.md").read_text()), README_HEADINGS)

    def test_existing_metrics_and_anova_are_unchanged(self):
        for name, digest in PROTECTED.items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_public_documentation_links_resolve(self):
        paths = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")),
                 *sorted((ROOT / "models").glob("*/README.md"))]
        for path in paths:
            for target in re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", path.read_text()):
                link = urlsplit(target)
                if link.scheme or link.netloc or not link.path:
                    continue
                with self.subTest(document=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue((path.parent / unquote(link.path)).exists())

    def test_standalone_bundle_destinations_are_consistent(self):
        readme = (ROOT / "README.md").read_text()
        for destination in ("results/precomputed", "data/processed", "results/analysis-inputs",
                            "results/predictions", "checkpoints/released-models"):
            self.assertIn(f"`{destination}`", readme)

    def test_lockfiles_do_not_depend_on_a_research_checkout(self):
        projects = [ROOT, *(ROOT / f"models/{name}" for name in ("biolord", "scgen", "morph", "state", "scgpt"))]
        for project in projects:
            for name in ("pyproject.toml", "uv.lock"):
                path = project / name
                with self.subTest(path=str(path.relative_to(ROOT))):
                    text = path.read_text()
                    self.assertNotIn("/mnt/", text)
                    self.assertNotIn("perturbation-decomposition-repro", text)

    def test_transfer_prediction_default_matches_the_documented_bundle(self):
        from models._native_execution import predictions
        for model in ("biolord", "scgen"):
            with self.subTest(model=model), redirect_stderr(io.StringIO()):
                with patch("reproduction.io.verify", side_effect=ValueError("stop before loading")) as verify:
                    with self.assertRaises(SystemExit) as error:
                        predictions(model, ["--target", "k562"])
                self.assertEqual(error.exception.code, 2)
                verify.assert_called_once_with(Path("results/predictions"))

    def test_legacy_combined_prediction_bundles_remain_explicitly_selectable(self):
        from models._native_execution import predictions
        with redirect_stderr(io.StringIO()):
            with patch("reproduction.io.verify", side_effect=ValueError("stop before loading")) as verify:
                with self.assertRaises(SystemExit):
                    predictions("biolord", ["--target", "k562", "--results_dir", "legacy/full"])
        verify.assert_called_once_with(Path("legacy/full"))


if __name__ == "__main__":
    unittest.main()
