"""Public benchmark numbers remain internally consistent without private data."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicAggregateTests(unittest.TestCase):
    def test_final_event_macro_rebuilds_from_public_event_scores(self):
        result = json.loads((ROOT / "docs/results/final_water_benchmark.json").read_text(encoding="utf-8"))
        events = result["events"]
        self.assertEqual({event["event"] for event in events},
                         {"nigeria_20180921", "somalia_20180507"})
        self.assertEqual(sum(event["chips"] for event in events), 44)
        self.assertEqual(sum(event["common_eligible_pixels"] for event in events), 9_394_609)
        for method, macro in result["event_macro"].items():
            for metric in ("iou", "f1"):
                values = [event["methods"][method][metric] for event in events]
                self.assertTrue(all(0 <= value <= 1 for value in values))
                self.assertAlmostEqual(macro[metric], sum(values) / len(values))
        for event in events:
            self.assertGreater(event["quality_and_ambiguity_retention"]["withheld_observed_water_pixels"], 0)
            self.assertLess(event["quality_and_ambiguity_retention"]["retained_fraction"], 1)


if __name__ == "__main__":
    unittest.main()
