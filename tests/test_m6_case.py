"""M6 graph, label-independent support and assumption invariants."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from flood_access.case_graph import build_graph, shortest_to_facilities, travel_direction
from flood_access.case_evidence import load_case_observation
from flood_access.case_scenarios import access_rows, closed_segments, summarise_access
from flood_access.raster_pipeline import S1_BANDS, S2_BANDS, VALIDITY_BANDS
from flood_access.cli import local_status


class RoadGraphTests(unittest.TestCase):
    def test_shared_ids_not_geometry_crossings_define_topology_and_direction(self):
        nodes = [{"type": "node", "id": i, "lon": lon, "lat": lat}
                 for i, lon, lat in ((1, -0.8, 38.0), (2, -0.79, 38.01),
                                     (3, -0.8, 38.01), (4, -0.79, 38.0),
                                     (5, -0.78, 38.01))]
        ways = [{"type": "way", "id": 10, "nodes": [1, 2, 5],
                 "tags": {"highway": "residential", "oneway": "yes"}},
                {"type": "way", "id": 11, "nodes": [3, 4],
                 "tags": {"highway": "residential", "bridge": "yes", "layer": "1"}}]
        _, edges, audit = build_graph([*nodes, *ways], {"residential": 30}, "2019-09-12T00:00:00Z")
        self.assertEqual(audit["weak_components"], 2)
        self.assertEqual(len(edges), 4)  # two forward steps, two directions on bridge
        self.assertIsNone(shortest_to_facilities(edges, {"hospital": 5}, set())[0].get(3))
        self.assertIsNone(shortest_to_facilities(edges, {"hospital": 1}, set())[0].get(5))
        self.assertEqual(travel_direction({"highway": "motorway"})[0], 1)
        self.assertIsNone(travel_direction({"highway": "residential", "oneway": "reversible"})[0])

    def test_removed_segments_cannot_improve_cost_and_unknown_is_explicit(self):
        edges = [{"edge_id": "a", "segment_id": "a", "source": 1, "target": 2, "cost_seconds": 5,
                  "bridge": False, "tunnel": False, "length_m": 100},
                 {"edge_id": "b", "segment_id": "b", "source": 2, "target": 3, "cost_seconds": 5,
                  "bridge": True, "tunnel": False, "length_m": 100},
                 {"edge_id": "c", "segment_id": "c", "source": 1, "target": 3, "cost_seconds": 20,
                  "bridge": False, "tunnel": False, "length_m": 100}]
        segments = {row["segment_id"]: row for row in edges}
        evidence = {"b": {"within_tile_m": 100, "retained_m": 20,
                          "source_supported_water_m_by_scenario": {"wet": 25},
                          "retained_water_m_by_scenario": {"wet": 0}}}
        base = access_rows(edges, {"h": 3}, {"o": 1, "isolated": 2}, set())
        by_id = {row["origin_id"]: row for row in base}
        exempt, _ = closed_segments(segments, evidence, "wet", policy="supported_overlap",
                                    bridge_policy="bridge_exempt", minimum_exposed_m=15)
        candidate, _ = closed_segments(segments, evidence, "wet", policy="supported_overlap",
                                       bridge_policy="bridge_candidate", minimum_exposed_m=15)
        retained, _ = closed_segments(segments, evidence, "wet", policy="retained_overlap",
                                      bridge_policy="bridge_candidate", minimum_exposed_m=15)
        self.assertEqual(exempt, set())
        self.assertEqual(candidate, {"b"})
        self.assertEqual(retained, set())
        changed = access_rows(edges, {"h": 3}, {"o": 1, "isolated": 2}, candidate, by_id)
        self.assertEqual(summarise_access(base, changed)["newly_disconnected"], 1)
        self.assertGreater(changed[1]["cost_delta_seconds"], 0)
        all_unknown, _ = closed_segments(segments, evidence, "wet",
                                         policy="supported_plus_all_unknown",
                                         bridge_policy="bridge_candidate", minimum_exposed_m=15)
        self.assertEqual(all_unknown, {"a", "b", "c"})
        disconnected = access_rows(edges, {"h": 3}, {"o": 1}, all_unknown,
                                   {"o": by_id["o"]})
        self.assertTrue(disconnected[0]["newly_disconnected"])
        self.assertIsNone(disconnected[0]["estimated_cost_seconds"])


class ObservationTests(unittest.TestCase):
    def test_case_loader_never_needs_a_label_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({"tile_id": "Spain_fixture",
                "event_id": "spain_20190917", "analysis_split": "case_study_exploratory"}))
            affine = from_origin(-0.8, 38.1, 0.0001, 0.0001)
            def raster(name, array, bands):
                with rasterio.open(root / name, "w", driver="GTiff", width=4, height=4,
                                   count=len(array), dtype=array.dtype, crs="EPSG:4326",
                                   transform=affine) as sink:
                    sink.write(array)
                    sink.descriptions = bands
            radar = np.full((2, 4, 4), -20, dtype=np.float32)
            optical = np.full((13, 4, 4), 0.2, dtype=np.float32)
            optical[:, 0, 0] = np.nan
            radar[:, 0, 0] = np.nan
            validity = np.ones((5, 4, 4), dtype=np.uint8)
            validity[1:5, 0, 0] = 0
            raster("radar_db.tif", radar, S1_BANDS)
            raster("optical_toa.tif", optical, S2_BANDS)
            raster("validity.tif", validity, VALIDITY_BANDS)
            raster("cloud_status.tif", np.full((1, 4, 4), 255, dtype=np.uint8), ("cloud_unknown",))
            self.assertFalse((root / "label.tif").exists())
            tile = load_case_observation(root, "Spain_fixture")
            self.assertEqual(int(tile.eligible.sum()), 15)
            self.assertTrue(np.all(tile.label == -1))
            self.assertFalse(tile.eligible[0, 0])


class MilestoneStatusTests(unittest.TestCase):
    def test_m6_status_requires_matching_m2_and_m5_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw/sen1floods11/v1.1"
            pilot = [raw / "Sen1Floods11_Metadata.geojson"]
            for chip in ("Bolivia_103757", "Ghana_103272", "Spain_7370579"):
                for layer in ("LabelHand", "S1Hand", "S2Hand"):
                    pilot.append(raw / "data/flood_events/HandLabeled" / layer / f"{chip}_{layer}.tif")
            for path in pilot:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / "data/processed/m2_runs/M2-test").mkdir(parents=True)
            (root / "data/processed/m2_runs/M2-test/run_manifest.json").write_text("{}")
            for run_id, fields in (("M2-test", {}), ("M3-test", {}),
                                   ("M4-test", {"m3_reference_run_id": "M3-test"}),
                                   ("M4B-test", {"m3_reference_run_id": "M3-test"}),
                                   ("M5-test", {"m3_reference_run_id": "M3-test",
                                                "m4_control_run_id": "M4B-test"}),
                                   ("M6-test", {"m2_run_id": "M2-test", "m5_run_id": "M5-test"})):
                folder = root / "runs" / run_id
                folder.mkdir(parents=True)
                manifest = {"run_id": run_id, "status": "complete", **fields}
                payload = json.dumps(manifest).encode()
                (folder / "run_manifest.json").write_bytes(payload)
                (folder / "status.json").write_text(json.dumps({"run_id": run_id,
                    "status": "complete", "manifest_sha256": hashlib.sha256(payload).hexdigest()}))
            self.assertEqual(local_status(root)["status"], "m6_accessibility_scenarios")
            status_path = root / "runs/M6-test/status.json"
            status = json.loads(status_path.read_text())
            status["manifest_sha256"] = "0" * 64
            status_path.write_text(json.dumps(status))
            self.assertEqual(local_status(root)["status"], "m5_evidence")


if __name__ == "__main__":
    unittest.main()
