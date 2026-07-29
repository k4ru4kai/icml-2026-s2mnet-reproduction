"""Tests for the executable Claim 2 SSTM audit."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "repro/diagnostics/verify_claim2_sstm.py"
SPEC = importlib.util.spec_from_file_location("verify_claim2_sstm", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class Claim2SstmAuditTests(unittest.TestCase):
    def test_released_mechanism_audit_matches_pinned_source(self) -> None:
        result = AUDIT.audit_released_sstm()
        self.assertTrue(result["audit_pass"])
        self.assertEqual(
            [stage["effective_k"] for stage in result["stage_records"]],
            [32, 32, 32, 22, 11],
        )
        self.assertEqual(result["tensorflow_fft2d_innermost_axes"], ["W", "C"])
        self.assertFalse(result["literal_retained_subset_in_released_path"])

    def test_constant_image_retains_dc_energy(self) -> None:
        image = np.ones((64, 64, 3), dtype=np.float32)
        ratio = AUDIT.centered_spatial_energy_retention(image, 32)
        self.assertTrue(math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12))

    def test_checkerboard_disproves_universal_95_percent_property(self) -> None:
        image = (
            2 * (np.indices((64, 64), dtype=np.int32).sum(axis=0) % 2) - 1
        ).astype(np.float32)
        ratio = AUDIT.centered_spatial_energy_retention(image, 32)
        self.assertLess(ratio, 0.95)

    def test_cost_alternatives_do_not_recover_63_percent(self) -> None:
        result = AUDIT.audit_cost_claim()
        self.assertFalse(result["claim_reproducible_from_released_materials"])
        self.assertFalse(any(result["alternative_matches_63_percent"].values()))

    def test_explicit_npy_image_is_hashed_and_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "permitted_validation_image.npy"
            np.save(path, np.ones((352, 352, 3), dtype=np.float32))
            result = AUDIT.analyze_explicit_images([path])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["summary"]["n"], 1)
        self.assertTrue(result["summary"]["mean_exceeds_95_percent"])
        self.assertEqual(len(result["images"][0]["sha256"]), 64)

    def test_checked_in_json_matches_fresh_deterministic_audit(self) -> None:
        expected_path = REPO_ROOT / "results/audits/claim2_sstm_audit.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(AUDIT.build_audit(), expected)


if __name__ == "__main__":
    unittest.main()
