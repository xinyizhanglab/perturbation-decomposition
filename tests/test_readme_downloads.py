"""Exercise the README download instructions using tiny offline release assets."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from data.artifacts import create_example
from data.release_check import registry
from data.training_inputs import for_model, selected_bundles
from reproduction.download import download, pack


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MOCK_GH = """#!/PLACEHOLDER_PYTHON
import fnmatch
import os
from pathlib import Path
import shutil
import sys

args = sys.argv[1:]
assert args[:3] == ["release", "download", "rep4rev-v1"], args
assert args[args.index("--repo") + 1] == "xinyizhanglab/perturbation-decomposition", args
assert "--skip-existing" in args
pattern = args[args.index("--pattern") + 1]
out = Path(args[args.index("--dir") + 1])
out.mkdir(parents=True, exist_ok=True)
files = sorted(p for p in Path(os.environ["FAKE_RELEASE"]).iterdir()
               if fnmatch.fnmatchcase(p.name, pattern))
assert files, pattern
for path in files:
    if path.name == os.environ.get("FAIL_ASSET"):
        sys.exit(17)
    destination = out / path.name
    if not destination.exists():
        shutil.copyfile(path, destination)
        with Path(os.environ["FAKE_DOWNLOAD_LOG"]).open("a") as log:
            log.write(path.name + "\\n")
"""


class ReadmeDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.release = self.root / "release"
        self.release.mkdir()
        self.work = self.root / "work"
        self.work.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "requests.log"
        gh = self.bin / "gh"
        gh.write_text(MOCK_GH.replace("PLACEHOLDER_PYTHON", sys.executable))
        gh.chmod(0o755)
        self.environment = {
            **os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "FAKE_RELEASE": str(self.release), "FAKE_DOWNLOAD_LOG": str(self.log),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.text = README.read_text()
        self.helper = next(block for block in re.findall(
            r"```bash\n(.*?)```", self.text, flags=re.S) if block.startswith("download()"))
        self.spec = registry()
        source = create_example(self.root / "source")
        self.archives = {}
        for index in self.spec["assets"]:
            if not index.endswith(".tar.gz.json"):
                continue
            archive = index.removesuffix(".json")
            path = self.release / archive
            pack(source, path)
            self.archives[archive] = path.read_bytes()
            parts = sorted(name for name in self.spec["assets"]
                           if name.startswith(archive + ".part-"))
            if parts:
                data = self.archives[archive]
                for number, name in enumerate(parts):
                    (self.release / name).write_bytes(
                        data[len(data) * number // len(parts):
                             len(data) * (number + 1) // len(parts)])
                path.unlink()

    def run_downloads(self, commands, **environment):
        return subprocess.run(
            ["bash", "-c", self.helper + "\n" + commands], cwd=self.work,
            env={**self.environment, **environment}, capture_output=True, text=True,
        )

    def test_all_thirteen_archive_downloads_and_exact_alphabetical_reassembly(self):
        for name, data in self.archives.items():
            with self.subTest(archive=name):
                result = self.run_downloads(f"download {name.removesuffix('.tar.gz')}")
                self.assertEqual(result.returncode, 0, result.stderr)
                path = self.work / "reproduction_files" / name
                self.assertEqual(path.read_bytes(), data)
                self.assertTrue(path.with_suffix(".gz.json").is_file())
                self.assertFalse(path.with_suffix(".gz.partial").exists())
        expected = set(self.spec["assets"]) - {"inputs.json"}
        self.assertEqual(set(self.log.read_text().splitlines()), expected)

    def test_existing_downloads_are_reused_without_replacing_the_complete_archive(self):
        self.assertEqual(self.run_downloads("download models-v1").returncode, 0)
        path = self.work / "reproduction_files/models-v1.tar.gz"
        before = path.stat().st_mtime_ns
        requests = self.log.read_text()
        self.assertEqual(self.run_downloads("download models-v1").returncode, 0)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(self.log.read_text(), requests)

    def test_a_failed_download_does_not_promote_a_partial_archive(self):
        result = self.run_downloads(
            "download models-v1", FAIL_ASSET="models-v1.tar.gz.part-aac")
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertFalse((self.work / "reproduction_files/models-v1.tar.gz").exists())
        result = self.run_downloads("download models-v1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.work / "reproduction_files/models-v1.tar.gz").read_bytes(),
                         self.archives["models-v1.tar.gz"])

    def test_reassembled_predictions_install_using_the_documented_index(self):
        self.assertEqual(self.run_downloads("download predictions-v1").returncode, 0)
        files = self.work / "reproduction_files"
        output = self.work / "results/predictions"
        download(files / "predictions-v1.tar.gz.json", output, files / "predictions-v1.tar.gz")
        self.assertEqual((output / "manifest.json").read_bytes(),
                         (self.root / "source/manifest.json").read_bytes())

    def test_corrupt_reassembly_is_rejected_by_the_existing_installer(self):
        name = "predictions-v1.tar.gz.part-aab"
        (self.release / name).write_bytes(b"bad")
        self.assertEqual(self.run_downloads("download predictions-v1").returncode, 0)
        files = self.work / "reproduction_files"
        output = self.work / "results/predictions"
        with self.assertRaises(ValueError):
            download(files / "predictions-v1.tar.gz.json", output,
                     files / "predictions-v1.tar.gz")
        self.assertFalse(output.exists())

    def test_native_table_downloads_include_every_required_companion(self):
        catalog = json.loads((ROOT / "reproduction_files/inputs.json").read_text())
        table = self.text.split("**Native training inputs**", 1)[1].split(
            "For example,", 1)[0]
        covered = set()
        for row in table.splitlines():
            if not row.startswith("|") or "download " not in row:
                continue
            resources = re.findall(r"`download ([a-z0-9-]+)`", row)
            model = re.search(r"--model ([a-z0-9]+)", row)
            selection = re.search(r"--bundles ([a-z0-9-]+)", row)
            target = re.search(r"--target ([a-z0-9]+)", row)
            names = for_model(model[1], target[1] if target else "all") if model else [selection[1]]
            required = selected_bundles(catalog, names)
            expected = {catalog["bundles"][name]["archive"].removesuffix(".tar.gz")
                        for name in required}
            self.assertEqual(set(resources), expected, row)
            covered.update(required)
        self.assertEqual(covered, {
            "genetic", "morph", "scgpt-k562", "scgpt-pretrained",
            "biolord", "references", "calibration",
        })
        for target in ("k562", "rpe1", "hepg2", "jurkat"):
            self.assertIn(f"inputs-scgpt-{target}-v1.tar.gz", self.archives)

    def test_model_routes_download_before_installing_and_preserve_the_figure_route(self):
        for label, resource in (("1. Saved predictions", "predictions"), ("2. Inference", "models")):
            section = self.text.split(f"**{label}**", 1)[1].split("\n**", 1)[0]
            self.assertLess(section.index(f"download {resource}-v1"),
                            section.index("python -m data.artifacts install"))
        figures = self.text.split("## Pre-computed results", 1)[1].split("## Main analyses", 1)[0]
        self.assertNotIn("\ndownload ", figures)
        self.assertNotIn("gh ", figures)


if __name__ == "__main__":
    unittest.main()
