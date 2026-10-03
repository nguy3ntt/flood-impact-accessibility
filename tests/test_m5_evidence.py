"""Meaningful M5 invariants: group separation, unknowns and coherent alternatives."""

from __future__ import annotations

import unittest
from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np

from flood_access.evidence import (calibrated_probability, calibration_points,
                                   fit_monotone_logistic, hashed_event_rank,
                                   quality_flags, reliability, retained_mask,
                                   scenario_summary, spatial_scenarios)
from flood_access.mapping import MappingTile
from flood_access.cli import local_status


def fixture(tile_id: str = "fixture", split: str = "train") -> MappingTile:
    label = np.zeros((32, 32), dtype=np.int8)
    label[:, 16:] = 1
    label[:4, :4] = -1
    eligible = label != -1
    radar = np.full((2, 32, 32), -20, dtype=np.float32)
    optical = np.full((13, 32, 32), 0.2, dtype=np.float32)
    optical[2, :, :16] = 0.05  # B3 green, high NDWI for dry in this fixture
    optical[7, :, 16:] = 0.05  # B8 NIR, high NDWI for wet
    return MappingTile(tile_id, "fixture-event", split, label, radar, optical, eligible, None, None)


class CalibrationContractTests(unittest.TestCase):
    def test_calibration_samples_do_not_use_label_to_choose_position(self):
        tile = fixture()
        x1, y1 = calibration_points(tile, 8)
        self.assertLessEqual(len(x1), 16)
        tile.label[8:16, 8:16] = 1 - tile.label[8:16, 8:16]
        x2, y2 = calibration_points(tile, 8)
        self.assertTrue(np.array_equal(x1, x2))
        self.assertFalse(np.array_equal(y1, y2))
        with self.assertRaises(ValueError):
            calibration_points(fixture(split="validation"), 8)

    def test_monotone_probability_and_unknowns(self):
        samples = {
            "event_a": (np.array([-0.8, -0.6, 0.6, 0.8]), np.array([0, 0, 1, 1])),
            "event_b": (np.array([-0.7, -0.5, 0.5, 0.7]), np.array([0, 0, 1, 1])),
        }
        fit = fit_monotone_logistic(samples, steps=300, learning_rate=0.25, slope_l2=0.005)
        self.assertGreater(fit["slope"], 0)
        tile = fixture()
        p = calibrated_probability(tile, fit)
        self.assertTrue(np.isnan(p[~tile.eligible]).all())
        self.assertTrue(np.isfinite(p[tile.eligible]).all())
        self.assertGreater(float(p[20, 20]), float(p[20, 5]))

    def test_reliability_rejects_unknown_and_reports_empty(self):
        label = np.array([[1, -1], [0, 1]], dtype=np.int8)
        p = np.array([[0.8, np.nan], [0.2, 0.7]], dtype=np.float32)
        support = label != -1
        result = reliability(label, p, support, 5)
        self.assertEqual(result["pixels"], 3)
        self.assertGreater(result["brier"], 0)
        self.assertEqual(reliability(label, p, np.zeros_like(support), 5)["pixels"], 0)
        support[0, 1] = True
        with self.assertRaises(ValueError):
            reliability(label, p, support, 5)


class ScenarioContractTests(unittest.TestCase):
    def test_block_cases_preserve_unknown_and_are_nested_at_fixed_thresholds(self):
        eligible = np.ones((16, 16), dtype=bool)
        eligible[:2, :2] = False
        p = np.full((16, 16), 0.55, dtype=np.float32)
        p[~eligible] = np.nan
        p[8:, :] = 0.8
        bands, names = spatial_scenarios(p, eligible, "a", block_pixels=8,
                                        thresholds=[0.65, 0.5, 0.35], offset_seeds=[1, 2],
                                        maximum_offset=0.1)
        self.assertTrue(np.all(bands[:, ~eligible] == 255))
        self.assertTrue(np.all(bands[0, eligible] <= bands[1, eligible]))
        self.assertTrue(np.all(bands[1, eligible] <= bands[2, eligible]))
        self.assertTrue(np.array_equal(bands, spatial_scenarios(p, eligible, "a", block_pixels=8,
                                                               thresholds=[0.65, 0.5, 0.35],
                                                               offset_seeds=[1, 2], maximum_offset=0.1)[0]))
        for band in bands:
            self.assertEqual(len(np.unique(band[2:8, 2:8])), 1)
        summary = scenario_summary(bands, eligible, names)
        self.assertIn("not a water probability", summary["frequency_interpretation"])

    def test_spatial_offset_preserves_pixel_detail(self):
        p = np.array([[0.45, 0.55], [0.45, 0.55]], dtype=np.float32)
        eligible = np.ones((2, 2), dtype=bool)
        bands, _ = spatial_scenarios(p, eligible, "pixel-detail", block_pixels=2,
                                     thresholds=[0.5], offset_seeds=[7], maximum_offset=0.0)
        self.assertTrue(np.array_equal(bands[0], np.array([[0, 1], [0, 1]])))
        self.assertTrue(np.array_equal(bands[1], bands[0]))

    def test_quality_abstention_keeps_source_unknown(self):
        tile = fixture()
        fit = {"slope": 2.0, "intercept": 0.0}
        p = calibrated_probability(tile, fit)
        flags = quality_flags(tile, weak_denominator_below=0.05,
                              radiometric_extreme_above=0.9)
        self.assertTrue(np.all(flags[~tile.eligible] == 255))
        retained = retained_mask(tile, p, flags, ambiguity_margin=0.15, require_quality=True)
        self.assertFalse(np.any(retained & ~tile.eligible))
        self.assertFalse(np.any(retained & (flags != 0)))
        self.assertLessEqual(int(retained.sum()), int(tile.eligible.sum()))

    def test_event_ranking_is_order_independent(self):
        events = ["c", "a", "b"]
        self.assertEqual(hashed_event_rank(events, "fixed"), hashed_event_rank(list(reversed(events)), "fixed"))
        with self.assertRaises(ValueError):
            hashed_event_rank(["a", "a"], "fixed")


class MilestoneStatusTests(unittest.TestCase):
    def test_m5_status_requires_hash_valid_run_linked_to_controlled_m4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, reference, control in (("M3-test", None, None),
                                               ("M4-test", "M3-test", None),
                                               ("M4B-test", "M3-test", None),
                                               ("M5-test", "M3-test", "M4B-test")):
                folder = root / "runs" / run_id
                folder.mkdir(parents=True)
                manifest = {"run_id": run_id, "status": "complete"}
                if reference:
                    manifest["m3_reference_run_id"] = reference
                if control:
                    manifest["m4_control_run_id"] = control
                payload = json.dumps(manifest).encode()
                (folder / "run_manifest.json").write_bytes(payload)
                (folder / "status.json").write_text(json.dumps({
                    "run_id": run_id, "status": "complete",
                    "manifest_sha256": hashlib.sha256(payload).hexdigest()}))
            self.assertEqual(local_status(root)["status"], "m5_evidence")
            path = root / "runs/M5-test/run_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["m4_control_run_id"] = "other"
            payload = json.dumps(manifest).encode()
            path.write_bytes(payload)
            status_path = root / "runs/M5-test/status.json"
            status = json.loads(status_path.read_text())
            status["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
            status_path.write_text(json.dumps(status))
            self.assertEqual(local_status(root)["status"], "m4_development")


if __name__ == "__main__":
    unittest.main()
