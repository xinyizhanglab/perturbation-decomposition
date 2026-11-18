"""Packaging and early-failure checks for the scGPT CUDA runtime."""

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from models.gpu_runtime import check_gpu


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "models/scgpt/_torchtext/torchtext"


class RecordedVocabularyTests(unittest.TestCase):
    def test_recorded_helper_files_match_their_hashes(self):
        metadata = json.loads((HELPER / "source.json").read_text())
        self.assertEqual(set(metadata["source_files"]), {"__init__.py", "_torchtext.py", "vocab.py"})
        for name, expected in metadata["source_files"].items():
            self.assertEqual(hashlib.sha256((HELPER / name).read_bytes()).hexdigest(), expected)

    def test_recorded_helper_preserves_indices_and_default(self):
        spec = importlib.util.spec_from_file_location("recorded_vocabulary", HELPER / "vocab.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        vocab = module.Vocab()
        for token, index in (("G1", 0), ("G2", 1), ("<pad>", 2)):
            vocab.insert_token(token, index)
        vocab.set_default_index(2)
        self.assertEqual(vocab(["G2", "UNKNOWN", "G1"]), [1, 2, 0])
        self.assertEqual(vocab.get_stoi(), {"G1": 0, "G2": 1, "<pad>": 2})
        self.assertEqual(vocab.get_itos(), ["G1", "G2", "<pad>"])

    def test_lock_uses_matching_cuda_wheels_and_local_helper(self):
        project = (ROOT / "models/scgpt/pyproject.toml").read_text()
        for requirement in ("torch==2.10.0+cu128", "torch-scatter==2.1.2+pt210cu128",
                            "torch-sparse==0.6.18+pt210cu128", "torchtext==0.18.0+recorded"):
            self.assertIn(requirement, project)
        self.assertIn('torchtext = { path = "_torchtext" }', project)
        self.assertNotIn("cu121", project)


class GPUPreflightTests(unittest.TestCase):
    @staticmethod
    def fake_torch():
        return SimpleNamespace(
            __version__="synthetic-torch", version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(is_available=lambda: True,
                get_device_name=lambda device: "synthetic GPU",
                get_device_capability=lambda device: (12, 0),
                get_arch_list=lambda: ["sm_120"], synchronize=lambda device: None),
            inference_mode=nullcontext, eye=lambda *args, **kwargs: np.eye(2))

    @patch("benchmarks._runtime.require_compute")
    def test_cpu_is_not_a_silent_fallback(self, require_compute):
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            check_gpu("cpu")
        require_compute.assert_called_once()

    @patch("benchmarks._runtime.require_compute")
    def test_unavailable_cuda_fails_before_loading_checkpoints(self, require_compute):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                check_gpu()
        require_compute.assert_called_once()

    def test_cluster_preflight_requires_an_allocation(self):
        with patch("benchmarks._runtime.require_compute", side_effect=RuntimeError("allocation required")):
            with self.assertRaisesRegex(RuntimeError, "allocation required"):
                check_gpu()

    @patch("benchmarks._runtime.require_compute")
    def test_pytorch_kernel_probe_reports_hardware(self, require_compute):
        with patch.dict("sys.modules", {"torch": self.fake_torch()}):
            report = check_gpu()
        self.assertEqual(report["capability"], [12, 0])
        self.assertEqual(report["compiled_architectures"], ["sm_120"])
        self.assertTrue(report["kernel_checks_passed"])
        require_compute.assert_called_once()

    @patch("benchmarks._runtime.require_compute")
    def test_kernel_failure_includes_runtime_and_does_not_fall_back(self, require_compute):
        torch = self.fake_torch()
        def incompatible_kernel(*args, **kwargs):
            raise RuntimeError("no kernel image is available")
        torch.eye = incompatible_kernel
        with patch.dict("sys.modules", {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "synthetic GPU") as error:
                check_gpu()
        self.assertIn("no CPU fallback", str(error.exception))
        self.assertIn("no kernel image", str(error.exception))
        require_compute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
