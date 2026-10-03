"""Frozen final-event boundary and pooled event-metric checks."""

from __future__ import annotations

import sys
from pathlib import Path
import json
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage_m8_final import freeze_plan, frozen_rows


class FinalProtocolTests(unittest.TestCase):
    def test_staging_plan_is_immutable_after_download(self):
        plan = {"schema_version": "m8_final_stage_v1", "plan_id": "M8S-example",
                "config_sha256": "config", "catalog_sha256": "catalog",
                "tiles": [{"tile_id": "a"}], "keys": ["object"],
                "new_objects": {"object": {"bytes": 9, "generation": "1"}},
                "new_bytes_estimate": 9, "max_new_bytes": 100}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            freeze_plan(path, plan, plan["new_objects"])
            first = path.read_bytes()
            repeat = {**plan, "new_objects": {}, "new_bytes_estimate": 0}
            self.assertEqual(freeze_plan(path, repeat, {}), plan)
            self.assertEqual(path.read_bytes(), first)
            with self.assertRaises(ValueError):
                freeze_plan(path, {**repeat, "tiles": []}, {})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), plan)

    def test_only_exact_frozen_events_and_counts_pass(self):
        config = {"final_event_chip_counts": {"event_a": 2, "event_b": 1}}
        rows = [{"tile_id": "a1", "event_id": "event_a", "analysis_split": "final_test"},
                {"tile_id": "a2", "event_id": "event_a", "analysis_split": "final_test"},
                {"tile_id": "b1", "event_id": "event_b", "analysis_split": "final_test"},
                {"tile_id": "dev", "event_id": "event_c", "analysis_split": "validation"}]
        self.assertEqual(len(frozen_rows(rows, config)), 3)
        with self.assertRaises(ValueError):
            frozen_rows(rows[:-2], config)
        with self.assertRaises(ValueError):
            frozen_rows(rows + [{"tile_id": "c1", "event_id": "event_c",
                                "analysis_split": "final_test"}], config)
        with self.assertRaises(ValueError):
            frozen_rows(rows + [rows[0]], config)

    def test_event_confusion_is_pooled_before_macro_average(self):
        try:
            from flood_access.mapping import metrics, sum_confusions
        except ImportError as error:
            self.skipTest(f"Raster dependencies unavailable: {error}")
        event_a = [{"tp": 1, "fp": 0, "fn": 1, "tn": 8, "eligible_pixels": 10},
                   {"tp": 9, "fp": 1, "fn": 0, "tn": 0, "eligible_pixels": 10}]
        event_b = [{"tp": 1, "fp": 1, "fn": 8, "tn": 0, "eligible_pixels": 10}]
        a = metrics(sum_confusions(event_a))["iou"]
        b = metrics(sum_confusions(event_b))["iou"]
        self.assertAlmostEqual(a, 10 / 12)
        self.assertAlmostEqual(b, 1 / 10)
        self.assertAlmostEqual((a + b) / 2, (10 / 12 + 0.1) / 2)
        self.assertNotAlmostEqual((a + b) / 2, metrics(sum_confusions(event_a + event_b))["iou"])


if __name__ == "__main__":
    unittest.main()
