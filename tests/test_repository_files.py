"""Repository-local plotting/processed downloads are pinned and installable."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from data.training_inputs import install


ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "reproduction_files"
SHIPPED = {
    "figures-v4.tar.gz", "figures-v4.tar.gz.json",
    "inputs-processed-v2.tar.gz", "inputs-processed-v2.tar.gz.json", "inputs.json",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class RepositoryFileTests(unittest.TestCase):
    def test_readme_uses_the_source_distributed_folder(self):
        text = (ROOT / "README.md").read_text()
        self.assertIn("RESULTS_DOWNLOADS=reproduction_files", text)
        self.assertIn("DATA_DOWNLOADS=reproduction_files", text)
        self.assertNotIn("RESULTS_DOWNLOADS=/path/to/", text)
        self.assertNotIn("DATA_DOWNLOADS=/path/to/", text)
        self.assertIn("**Optional model files**", text)
        self.assertIn("not needed to redraw saved figures", text)

    def test_five_required_files_exist_and_match_their_pinned_indexes(self):
        self.assertTrue(all((FILES / name).is_file() for name in SHIPPED))
        for stem in ("figures-v4", "inputs-processed-v2"):
            archive = FILES / f"{stem}.tar.gz"
            index = json.loads((FILES / f"{stem}.tar.gz.json").read_text())
            with self.subTest(archive=archive.name):
                self.assertEqual(index["archive"], archive.name)
                self.assertEqual(index["size"], archive.stat().st_size)
                self.assertEqual(index["sha256"], digest(archive))
        catalog = json.loads((FILES / "inputs.json").read_text())
        entry = catalog["bundles"]["processed"]
        self.assertEqual(entry["requires"], [])
        self.assertEqual(entry["index_sha256"], digest(FILES / entry["index"]))
        index = json.loads((FILES / entry["index"]).read_text())
        for key in ("sha256", "size", "manifest_sha256", "version", "archive"):
            self.assertEqual(entry[key], index[key])

    def test_source_distributed_files_are_separate_from_the_python_package(self):
        self.assertIn('"reproduction_files*"', (ROOT / "pyproject.toml").read_text())
        self.assertEqual(len(SHIPPED), 5)
        self.assertTrue(SHIPPED <= {path.name for path in FILES.iterdir()})

    def test_processed_install_selects_the_local_archive_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data"
            expected = (output / "processed").resolve()
            with patch("data.training_inputs.download", return_value=expected) as download:
                result = install(FILES / "inputs.json", output, ["processed"])
            self.assertEqual(result, [expected])
            download.assert_called_once_with(
                FILES / "inputs-processed-v2.tar.gz.json", expected,
                FILES / "inputs-processed-v2.tar.gz")
            self.assertFalse(output.exists())

    def test_figure_install_cli_selects_the_repository_local_files(self):
        from data.artifacts import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figures"
            with patch("data.artifacts.download", return_value=output) as download, \
                    redirect_stdout(StringIO()):
                main([
                    "install", "--index", str(FILES / "figures-v4.tar.gz.json"),
                    "--archive", str(FILES / "figures-v4.tar.gz"), "--out_dir", str(output),
                ])
            download.assert_called_once_with(
                FILES / "figures-v4.tar.gz.json", output, FILES / "figures-v4.tar.gz")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
