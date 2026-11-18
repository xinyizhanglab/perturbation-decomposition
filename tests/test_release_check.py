"""Public release checks use mocked metadata, never large downloads or credentials."""

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from data.release_check import check, main, registry


def fixture():
    data = b'{"archive":"test.tar.gz"}'
    checksum = hashlib.sha256(data).hexdigest()
    spec = {
        "schema_version": 1, "repository": "example/project", "tag": "review-v1",
        "assets": {
            "test.tar.gz.part-aaa": {"size": 5, "kind": "archive_part", "sha256": None},
            "test.tar.gz.json": {"size": len(data), "kind": "index", "sha256": checksum},
        },
    }
    rows = [{"name": name, "size": entry["size"], "state": "uploaded",
             "digest": "sha256:" + entry["sha256"] if entry["sha256"] else None}
            for name, entry in spec["assets"].items()]
    calls = []

    def read(url):
        calls.append(url)
        if "/releases/tags/" in url:
            return json.dumps({"id": 42, "tag_name": spec["tag"], "draft": False}).encode()
        if "/assets?" in url:
            return json.dumps(rows).encode()
        if url.endswith("test.tar.gz.json"):
            return data
        raise AssertionError(f"Unexpected or large asset request: {url}")

    return spec, rows, calls, read


class ReleaseCheckTests(unittest.TestCase):
    def test_real_registry_pins_the_supplied_org_and_tag_and_all_assets(self):
        spec = registry()
        self.assertEqual(spec["repository"], "xinyizhanglab/perturbation-decomposition")
        self.assertEqual(spec["tag"], "rep4rev-v1")
        self.assertEqual(len(spec["assets"]), 38)
        for stem, count in (("models-v1.tar.gz", 6), ("predictions-v1.tar.gz", 2)):
            parts = [name for name in spec["assets"] if name.startswith(stem + ".part-")]
            self.assertEqual(len(parts), count)
        self.assertNotIn("figures-v4.tar.gz", spec["assets"])
        self.assertNotIn("inputs-processed-v2.tar.gz", spec["assets"])
        for name, entry in spec["assets"].items():
            self.assertLess(entry["size"], 2 * 1024 ** 3, name)

    def test_complete_release_checks_only_metadata_and_small_json(self):
        spec, rows, calls, read = fixture()
        with patch("data.release_check._read", side_effect=read):
            result = check(spec)
        self.assertTrue(result["complete"])
        self.assertEqual(result["json_checksums_verified"], ["test.tar.gz.json"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["large_archive_bytes_downloaded"], 0)
        self.assertFalse(result["archive_contents_independently_verified"])

    def test_missing_asset_is_reported_without_downloading_a_replacement(self):
        spec, rows, calls, read = fixture()
        rows.pop(0)
        with patch("data.release_check._read", side_effect=read):
            result = check(spec)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing"], ["test.tar.gz.part-aaa"])

    def test_incomplete_upload_or_wrong_size_fails(self):
        for change in ({"size": 99}, {"state": "new"}):
            spec, rows, calls, read = fixture()
            rows[0].update(change)
            with patch("data.release_check._read", side_effect=read):
                result = check(spec)
            self.assertFalse(result["complete"])
            self.assertEqual(result["mismatched"][0]["asset"], "test.tar.gz.part-aaa")

    def test_wrong_json_bytes_or_reported_digest_fails(self):
        spec, rows, calls, read = fixture()
        rows[1]["digest"] = "sha256:" + "0" * 64
        with patch("data.release_check._read", side_effect=read):
            self.assertFalse(check(spec)["complete"])
        rows[1]["digest"] = None

        def changed(url):
            return b"changed" if url.endswith("test.tar.gz.json") else read(url)

        with patch("data.release_check._read", side_effect=changed):
            self.assertFalse(check(spec)["complete"])

    def test_draft_private_or_unpublished_release_is_not_claimed_verified(self):
        error = HTTPError("https://api.github.com/", 404, "Not Found", {}, None)
        with patch("data.release_check._read", side_effect=error):
            with self.assertRaisesRegex(ValueError, "No uploaded assets were verified"):
                check(fixture()[0])

    def test_pagination_is_not_limited_to_the_first_page(self):
        spec, rows, calls, read = fixture()
        filler = [{"name": f"extra-{i}", "size": 1, "state": "uploaded"} for i in range(100)]

        def paged(url):
            if url.endswith("&page=1"):
                return json.dumps(filler).encode()
            if url.endswith("&page=2"):
                return json.dumps(rows).encode()
            return read(url)

        with patch("data.release_check._read", side_effect=paged):
            result = check(spec)
        self.assertTrue(result["complete"])
        self.assertEqual(result["listed_assets"], 102)

    def test_plan_is_offline_and_does_not_write_files(self):
        stream = StringIO()
        with patch("data.release_check._read") as request, redirect_stdout(stream):
            self.assertEqual(main(["--plan"]), 0)
        request.assert_not_called()
        self.assertFalse(json.loads(stream.getvalue())["network_access"])

    def test_registry_refuses_unsafe_asset_names(self):
        spec = fixture()[0]
        spec["assets"]["../outside"] = {"size": 1, "sha256": None, "kind": "archive_part"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(spec))
            with patch("data.release_check.REGISTRY", path):
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    registry()


if __name__ == "__main__":
    unittest.main()
