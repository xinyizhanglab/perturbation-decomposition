"""Rendering-profile checks use small drawings, not scientific recomputation."""

import json
from pathlib import Path
import tempfile
import unittest

from figures.rendering import (
    PROFILE_PATH, drawing_sha256, pdf_fonttype, profile, saved_font_families, saved_settings, svg_evidence,
)


SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="100pt" height="80pt" viewBox="0 0 100 80">
<metadata><date>{date}</date>{extra}</metadata>
<defs><clipPath id="{identifier}"><rect width="90" height="70"/></clipPath>
<path id="DejaVuSans-41" d="M 0 0 L 1 1"/></defs>
<g><path clip-path="url(#{identifier})" d="M 1 2 L 9 10" style="fill: none;stroke: {color}"/></g>
</svg>"""


class RenderingProfileTests(unittest.TestCase):
    def test_saved_profile_coverage_includes_all_available_reference_figures(self):
        registry = json.loads(PROFILE_PATH.read_text())["figures"]
        self.assertEqual(len(registry), 48)
        self.assertEqual(sum(bool(entry["assets"]) for entry in registry.values()), 47)
        self.assertEqual(sum(entry["compatibility"] == "matched" for entry in registry.values()), 43)
        self.assertEqual(registry["fig3_enrichment_barplot"]["status"], "no_saved_reference")

    def test_saved_font_and_export_types_are_per_figure_not_global(self):
        self.assertEqual(profile("fig_pathway_crosscl")["rc_overrides"], {
            "font.family": ["DejaVu Sans"], "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42, "svg.fonttype": "path"})
        self.assertEqual(profile("fig1_e")["rc_overrides"]["pdf.fonttype"], 3)
        self.assertEqual(profile("fig4_b")["rc_overrides"]["svg.fonttype"], "none")

    def test_font_requests_in_source_cannot_override_saved_profile(self):
        import matplotlib
        original = matplotlib.rcParams["font.family"].copy()
        with saved_settings({"font.family": ["DejaVu Sans"], "pdf.fonttype": 42}):
            matplotlib.rcParams.update({"font.family": "Arial", "font.size": 13})
            self.assertEqual(matplotlib.rcParams["font.family"], ["DejaVu Sans"])
            self.assertEqual(matplotlib.rcParams["font.size"], 13)
            matplotlib.rcParams["pdf.fonttype"] = 3
            self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
        self.assertEqual(matplotlib.rcParams["font.family"], original)

    def test_metadata_and_random_ids_do_not_change_the_drawing_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first, second = base / "first.svg", base / "second.svg"
            first.write_text(SVG.format(date="yesterday", extra="", identifier="p123", color="#000001"))
            second.write_text(SVG.format(date="today", extra="<creator>different export</creator>",
                                         identifier="p456", color="#000001"))
            self.assertEqual(drawing_sha256(first), drawing_sha256(second))
            evidence = svg_evidence(first)
            self.assertEqual(evidence["font_families"], ["DejaVu Sans"])
            self.assertEqual(evidence["viewBox"], "0 0 100 80")

    def test_drawing_signature_detects_color_and_geometry_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first, second = base / "first.svg", base / "second.svg"
            text = SVG.format(date="same", extra="", identifier="p123", color="#000001")
            first.write_text(text)
            second.write_text(text.replace("#000001", "#000002"))
            self.assertNotEqual(drawing_sha256(first), drawing_sha256(second))
            second.write_text(text.replace("L 9 10", "L 10 10"))
            self.assertNotEqual(drawing_sha256(first), drawing_sha256(second))

    def test_pdf_font_embedding_types_are_observed_not_assumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pdf"
            path.write_bytes(b"/Subtype /Type3")
            self.assertEqual(pdf_fonttype(path), 3)
            path.write_bytes(b"/Subtype /Type0 /Subtype /CIDFontType2")
            self.assertEqual(pdf_fonttype(path), 42)
            path.write_bytes(b"/Subtype /Type3 /Subtype /CIDFontType2")
            self.assertIsNone(pdf_fonttype(path))

    def test_reference_only_routines_do_not_claim_original_rendering_compatibility(self):
        with self.assertRaisesRegex(ValueError, "Use --mode reference"):
            profile("fig1_c")
        self.assertEqual(profile("fig1_c", require_verified=False)["status"], "reference_only")

    def test_recorded_font_lookup_keeps_original_svg_family_names_and_restores_manager(self):
        import matplotlib
        from matplotlib import font_manager
        original_fonts = font_manager.fontManager.ttflist
        with matplotlib.rc_context(), saved_font_families(["DejaVu Sans"]):
            matplotlib.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
            self.assertEqual(matplotlib.rcParams["font.family"], ["Arial", "DejaVu Sans"])
            self.assertEqual({font.name for font in font_manager.fontManager.ttflist}, {"DejaVu Sans"})
        self.assertIs(font_manager.fontManager.ttflist, original_fonts)

    def test_unknown_recorded_font_does_not_substitute(self):
        with self.assertRaisesRegex(ValueError, "no replacement"):
            with saved_font_families(["A definitely unavailable original font"]):
                pass


if __name__ == "__main__":
    unittest.main()
