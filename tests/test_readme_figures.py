"""Check the README's per-plot commands without fitting or drawing models."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import runpy
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from figures.original import catalog
from figures.rendering import profile
from reproduction.figures import coverage


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ALIASES = {
    "fig2_d2": "fig2_d",
    "fig3_crosscl_transfer": "fig3_a",
    "fig4_d": "fig4_c",
}


def commands():
    rows = []
    for line in README.read_text().splitlines():
        if not line.startswith("|"):
            continue
        match = re.search(r"`(uv run python -m figures\.[^`\n]+)`", line)
        if not match:
            continue
        parts = shlex.split(match.group(1))
        module = parts[parts.index("-m") + 1]
        arguments = parts[parts.index("-m") + 2:]
        rows.append({
            "label": line.split("|")[1].strip(), "command": match.group(1),
            "module": module, "id": module.removeprefix("figures."),
            "arguments": arguments,
            "mode": arguments[arguments.index("--mode") + 1],
            "out_dir": arguments[arguments.index("--out_dir") + 1],
        })
    return rows


def grouped_commands():
    groups = {}
    for block in re.findall(r"```bash\n(.*?)```", README.read_text(), flags=re.S):
        for line in block.replace("\\\n", " ").splitlines():
            if not line.startswith("uv run python -m figures.render "):
                continue
            parts = shlex.split(line)
            selected = [parts[index + 1] for index, value in enumerate(parts) if value == "--figure"]
            if selected:
                groups[parts[parts.index("--out_dir") + 1]] = selected
    return groups


class ReadmeFigureTests(unittest.TestCase):
    def test_every_supported_original_plot_has_a_direct_command(self):
        originals = {row["id"] for row in commands() if row["mode"] == "original"}
        self.assertEqual(len(originals), 40)
        documented = originals | {alias for alias, target in ALIASES.items() if target in originals}
        expected = {name for name in catalog()
                    if profile(name, require_verified=False)["compatibility"] == "matched"}
        self.assertEqual(documented, expected)
        for alias, target in ALIASES.items():
            self.assertEqual(profile(alias)["assets"], profile(target)["assets"])
            self.assertIn(f"`{alias}`", README.read_text())

    def test_copy_only_commands_cover_gaps_without_claiming_redraw(self):
        copied = {row["id"] for row in commands() if row["mode"] == "reference"}
        expected = {row["id"] for row in coverage() if row["reference"] and not row["original"]}
        self.assertEqual(copied, expected)
        for row in commands():
            with self.subTest(command=row["command"]):
                self.assertIn(row["mode"], {"original", "reference"})
                if row["mode"] == "original":
                    self.assertEqual(profile(row["id"])["compatibility"], "matched")
                else:
                    self.assertTrue(profile(row["id"], require_verified=False)["assets"])
        text = README.read_text()
        self.assertIn("Commands with `--mode reference` export saved figures.", text)
        self.assertIn("**Supplementary and Tahoe figures**", text)

    def test_figure1_uses_paper_letters_not_the_old_script_letters(self):
        expected = {
            "1c": ("fig1_f", "original"), "1d": ("fig1_b", "original"),
            "1e": ("fig1_c", "reference"), "1f": ("fig1_d", "original"),
            "1g": ("fig1_dsup", "original"), "1i": ("fig1_e", "original"),
        }
        found = {row["label"].split(" ", 1)[0]: (row["id"], row["mode"])
                 for row in commands() if re.match(r"^1[cdefgi] — ", row["label"])}
        self.assertEqual(found, expected)
        text = README.read_text()
        self.assertIn("Panels 1a–b are schematics", text)
        self.assertNotRegex(text, r"(?m)\bfig1_h\b|^\| 1h\b")

    def test_whole_figure_commands_select_only_supported_panels(self):
        groups = grouped_commands()
        self.assertEqual(groups, {
            "results/figures/figure1": ["fig1_f", "fig1_b", "fig1_d", "fig1_dsup", "fig1_e"],
            "results/figures/figure2": ["fig2_a", "fig2_b", "fig2_c", "fig2_d", "fig2_e"],
            "results/figures/figure3": ["fig3_a", "fig3_c", "fig3_d", "fig3_e"],
            "results/figures/figure4": ["fig4_a", "fig4_b", "fig4_c"],
        })
        for ids in groups.values():
            for identifier in ids:
                self.assertEqual(profile(identifier)["compatibility"], "matched")

    def test_each_documented_entrypoint_parses_and_dispatches_its_exact_command(self):
        import figures.render as launcher
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "inputs"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": []}))
            for row in commands():
                with self.subTest(command=row["command"]), \
                        patch.object(launcher, "DEFAULT_RESULTS", source), \
                        patch.object(launcher, "available", return_value=[row["id"]]), \
                        patch.object(launcher, "render", return_value=Path(directory) / "not-written") as render, \
                        patch.object(sys, "argv", [row["module"], *row["arguments"]]), \
                        patch.object(sys, "path", sys.path.copy()), redirect_stdout(StringIO()):
                    runpy.run_module(row["module"], run_name="__main__")
                    render.assert_called_once_with(
                        source, Path(row["out_dir"]), [row["id"]], row["mode"], None)
            self.assertFalse((Path(directory) / "not-written").exists())

    def test_commands_are_explicit_portable_and_do_not_share_output_directories(self):
        rows = commands()
        self.assertEqual(len(rows), 44)
        self.assertEqual(len({row["out_dir"] for row in rows}), len(rows))
        for row in rows:
            with self.subTest(command=row["command"]):
                self.assertEqual(row["arguments"], [
                    "--mode", row["mode"], "--out_dir", row["out_dir"]])
                self.assertTrue((ROOT / (row["module"].replace(".", "/") + ".py")).is_file())
                self.assertTrue(row["out_dir"].startswith("results/figures/"))
                self.assertNotIn("..", Path(row["out_dir"]).parts)
                self.assertNotIn("/mnt/", row["command"])


if __name__ == "__main__":
    unittest.main()
