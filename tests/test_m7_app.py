"""M7 bundle integrity and read-only API behavior on an invented clean-clone fixture."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

try:
    from fastapi.testclient import TestClient
    from flood_access.analysis_bundle import build_synthetic, load_bundle
    from flood_access.serving import create_app
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "Install requirements/m7-serve.lock for M7 tests")
class AnalystAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory()
        result = build_synthetic(Path(cls.scratch.name))
        cls.root = Path(cls.scratch.name) / result["run_id"]
        cls.client = TestClient(create_app(cls.root))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.scratch.cleanup()

    def test_synthetic_case_is_explicit_and_consistent(self):
        about = self.client.get("/api/v1/about").json()
        self.assertEqual(about["mode"], "synthetic")
        self.assertIn("invented", about["case_title"].lower())
        self.assertEqual(about["coverage"]["outside_pilot_segments"], 3)
        response = self.client.get("/api/v1/scenarios/demo_water",
                                   params={"policy": "supported_overlap", "bridge_policy": "bridge_exempt"})
        self.assertEqual(response.status_code, 200)
        case = response.json()
        self.assertEqual(case["closed_segment_ids"], ["demo:middle"])
        self.assertEqual(case["summary"]["newly_disconnected"], 1)
        self.assertEqual(sum(row["newly_disconnected"] for row in case["origins"]), 1)
        quiet = self.client.get("/api/v1/scenarios/demo_quiet",
                                params={"policy": "supported_overlap", "bridge_policy": "bridge_exempt"}).json()
        self.assertEqual(quiet["closed_segment_ids"], [])
        self.assertEqual(quiet["summary"]["newly_disconnected"], 0)

    def test_evidence_unknowns_and_invalid_ids(self):
        outside = self.client.get("/api/v1/roads/demo:west").json()
        self.assertEqual(outside["evidence_state"], "outside_image_unknown")
        self.assertIsNone(outside["evidence"])
        exposed = self.client.get("/api/v1/roads/demo:middle").json()
        self.assertEqual(exposed["evidence_state"], "within_image")
        self.assertEqual(exposed["evidence"]["source_supported_water_m_by_scenario"]["demo_water"], 60)
        self.assertEqual(self.client.get("/api/v1/roads/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/scenarios/missing",
                         params={"policy": "supported_overlap", "bridge_policy": "bridge_exempt"}).status_code, 404)
        self.assertEqual(self.client.get("/api/v1/scenarios/demo_water",
                         params={"policy": "unsupported", "bridge_policy": "bridge_exempt"}).status_code, 404)
        self.assertEqual(self.client.get("/api/v1/scenarios/demo_water").status_code, 422)
        self.assertEqual(self.client.get("/api/v1/previews/missing.png").status_code, 404)

    def test_preview_export_and_static_ui(self):
        preview = self.client.get("/api/v1/previews/demo_water.png")
        self.assertEqual(preview.status_code, 200)
        self.assertTrue(preview.content.startswith(b"\x89PNG\r\n\x1a\n"))
        response = self.client.get("/api/v1/export", params={"scenario_id": "demo_water",
            "policy": "supported_overlap", "bridge_policy": "bridge_exempt"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["content-disposition"])
        export = response.json()
        self.assertEqual(export["selected_scenario"]["summary"]["newly_disconnected"], 1)
        self.assertEqual(export["baseline"][0]["estimated_cost_seconds"], 300)
        self.assertNotIn("roads", export)
        self.assertEqual(self.client.get("/static/index.html").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        font = self.client.get("/static/fonts/ibm-plex-sans-condensed-latin-400-normal.woff2")
        self.assertEqual(font.status_code, 200)
        self.assertEqual(font.headers["content-type"], "font/woff2")

    def test_mutated_file_rejected_before_serving(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / self.root.name
            shutil.copytree(self.root, copy)
            evidence_path = copy / "evidence.json"
            content = json.loads(evidence_path.read_text(encoding="utf-8"))
            content["demo:middle"]["source_supported_m"] = 9999
            evidence_path.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output changed"):
                load_bundle(copy)


if __name__ == "__main__":
    unittest.main()
