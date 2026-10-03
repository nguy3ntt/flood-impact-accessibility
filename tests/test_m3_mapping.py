import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from flood_access.mapping import (MappingTile, confusion, load_tile, metrics,
                                  optical_index, select_threshold)
from flood_access.cli import local_status
from flood_access.raster_pipeline import S1_BANDS, S2_BANDS, VALIDITY_BANDS


def fixture_tile(split="train"):
    label = np.array([[1, 0], [-1, 0]], dtype=np.int8)
    radar = np.full((2, 2, 2), -15, dtype=np.float32)
    radar[1] = np.array([[-24, -9], [-17, -8]], dtype=np.float32)
    optical = np.full((13, 2, 2), 0.2, dtype=np.float32)
    optical[S2_BANDS.index("B3")] = np.array([[0.5, 0.1], [0.0, 0.1]])
    optical[S2_BANDS.index("B8")] = np.array([[0.1, 0.5], [0.0, 0.5]])
    optical[S2_BANDS.index("B11")] = np.array([[0.1, 0.5], [0.0, 0.5]])
    eligible = np.array([[True, True], [False, True]])
    return MappingTile("fixture_1", "fixture_event", split, label, radar, optical,
                       eligible, from_origin(0, 2, 1, 1), "EPSG:4326")


class MappingContractTests(unittest.TestCase):
    def test_unknown_pixels_cannot_be_scored_and_empty_iou_is_undefined(self):
        tile = fixture_tile()
        prediction = np.array([[1, 0], [1, 0]])
        counts = confusion(tile.label, prediction, tile.eligible)
        self.assertEqual((counts["tp"], counts["tn"], counts["eligible_pixels"]), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "Unknown label"):
            confusion(tile.label, prediction, np.ones((2, 2), dtype=bool))
        self.assertIsNone(metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 3,
                                   "eligible_pixels": 3})["iou"])

    def test_threshold_selection_rejects_validation_and_uses_train_only(self):
        train = fixture_tile()
        threshold, curve = select_threshold([train], "ndwi", [-0.5, 0, 0.5])
        self.assertEqual(threshold, -0.5)  # deterministic smallest-threshold tie break
        self.assertEqual(len(curve), 3)
        with self.assertRaisesRegex(ValueError, "training"):
            select_threshold([train, fixture_tile("validation")], "ndwi", [0])

    def test_index_undefined_area_stays_unknown(self):
        tile = fixture_tile()
        score = optical_index(tile, "ndwi")
        self.assertTrue(np.isnan(score[1, 0]))
        self.assertGreater(score[0, 0], 0)

    def test_loader_rejects_lying_validity_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tile = fixture_tile()
            profile = dict(driver="GTiff", height=2, width=2, crs="EPSG:4326",
                           transform=tile.transform)
            cases = (("label.tif", tile.label[None], ("observed_water_label",), -1),
                     ("radar_db.tif", tile.radar, S1_BANDS, np.nan),
                     ("optical_toa.tif", tile.optical, S2_BANDS, np.nan),
                     ("validity.tif", np.ones((5, 2, 2), dtype=np.uint8), VALIDITY_BANDS, None),
                     ("cloud_status.tif", np.full((1, 2, 2), 255, dtype=np.uint8), ("unknown",), 255))
            for name, data, descriptions, nodata in cases:
                with rasterio.open(root / name, "w", **profile, count=len(data),
                                   dtype=str(data.dtype), nodata=nodata) as sink:
                    sink.write(data)
                    sink.descriptions = descriptions
            (root / "metadata.json").write_text(
                '{"tile_id":"fixture_1","event_id":"fixture_event","analysis_split":"train"}')
            with self.assertRaisesRegex(ValueError, "Label-valid mask"):
                load_tile(root, {"tile_id": "fixture_1", "event_id": "fixture_event",
                                 "analysis_split": "train"})


class SelectionTests(unittest.TestCase):
    def test_nested_selection_never_includes_final_test(self):
        script_dir = Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(script_dir))
        try:
            spec = importlib.util.spec_from_file_location("stage_m3_data", script_dir / "stage_m3_data.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(script_dir))
        catalog = [{"tile_id": f"{event}_{n}", "event_location": event,
                    "analysis_split": split}
                   for split, event in (("train", "A"), ("validation", "B"),
                                        ("final_test", "C")) for n in range(5)]
        small = module.selected_chips(catalog, 2, 2)
        large = module.selected_chips(catalog, 4, 4)
        self.assertEqual(len(small), 4)
        self.assertTrue({row["tile_id"] for row in small}.issubset({row["tile_id"] for row in large}))
        self.assertFalse(any(row["analysis_split"] == "final_test" for row in large))

    def test_status_requires_a_matching_completed_run_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/M3-fixture"
            run.mkdir(parents=True)
            manifest = {"run_id": "M3-fixture", "status": "complete"}
            payload = json.dumps(manifest).encode()
            (run / "run_manifest.json").write_bytes(payload)
            status = {"run_id": "M3-fixture", "status": "complete",
                      "manifest_sha256": hashlib.sha256(payload).hexdigest()}
            (run / "status.json").write_text(json.dumps(status))
            self.assertTrue(local_status(root)["models_trained"])
            status["manifest_sha256"] = "0" * 64
            (run / "status.json").write_text(json.dumps(status))
            self.assertFalse(local_status(root)["models_trained"])


try:
    from flood_access.mapping_train import CompactUNet, predict_unet, train_unet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch training environment is optional in core CI")
class CompactModelTests(unittest.TestCase):
    def test_model_shape_and_training_split_gate(self):
        import torch
        model = CompactUNet(2, 4)
        self.assertEqual(tuple(model(torch.zeros(1, 2, 32, 32)).shape), (1, 1, 32, 32))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "training"):
                train_unet([fixture_tile("validation")], "radar", Path(directory) / "model.pt",
                           seed=1, epochs=1, patches_per_tile=1, patch_size=2, base_channels=4)

    def test_tiny_training_preserves_unknown_prediction_pixels(self):
        small = fixture_tile()
        tile = MappingTile(small.tile_id, small.event_id, small.split,
                           np.tile(small.label, (16, 16)), np.tile(small.radar, (1, 16, 16)),
                           np.tile(small.optical, (1, 16, 16)),
                           np.tile(small.eligible, (16, 16)), small.transform, small.crs)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            model, profile = train_unet([tile], "radar", checkpoint, seed=4,
                                        epochs=1, patches_per_tile=1, patch_size=16,
                                        base_channels=4)
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(np.isfinite(profile["curve"][0]["train_loss"]))
            score = predict_unet(model, tile, "radar")
            self.assertTrue(np.isfinite(score[tile.eligible]).all())
            self.assertTrue(np.isnan(score[~tile.eligible]).all())


if __name__ == "__main__":
    unittest.main()
