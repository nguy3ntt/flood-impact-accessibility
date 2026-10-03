"""M4 split, optical-absence and fused-model contract checks."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np

from flood_access.mapping import MappingTile
from flood_access.cli import local_status

try:
    import torch
    from flood_access.fusion import (late_fusion, model_inputs, nested_train_subset,
                                     predict_model, stress_block, train_model)
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "M4 model tests need the optional training lock")
class FusionContractTests(unittest.TestCase):
    @staticmethod
    def tile(tile_id: str, event: str = "train-event", split: str = "train") -> MappingTile:
        shape = (32, 32)
        label = np.zeros(shape, dtype=np.int8)
        label[5:22, 7:24] = 1
        label[:3, :] = -1
        eligible = label != -1
        radar = np.full((2, *shape), -20, dtype=np.float32)
        optical = np.full((13, *shape), 0.2, dtype=np.float32)
        return MappingTile(tile_id, event, split, label, radar, optical, eligible, None, None)

    def test_nested_budget_is_label_blind_and_split_gated(self):
        tiles = [self.tile(f"a_{index}", "a") for index in range(5)] + \
                [self.tile(f"b_{index}", "b") for index in range(5)]
        small = {tile.tile_id for tile in nested_train_subset(tiles, 2)}
        large = {tile.tile_id for tile in nested_train_subset(list(reversed(tiles)), 4)}
        self.assertEqual(len(small), 4)
        self.assertTrue(small < large)
        tiles[0].label[:] = 1
        self.assertEqual(small, {tile.tile_id for tile in nested_train_subset(tiles, 2)})
        with self.assertRaises(ValueError):
            nested_train_subset(tiles + [self.tile("final", split="final_test")], 2)

    def test_contiguous_missing_optical_is_reproducible_and_explicit(self):
        tile = self.tile("tile")
        block = stress_block(tile.tile_id, tile.label.shape, 0.25)
        self.assertEqual(int(block.sum()), 16 * 16)
        self.assertTrue(np.array_equal(block, stress_block(tile.tile_id, tile.label.shape, 0.25)))
        early, present = model_inputs(tile, "early", "block_missing")
        self.assertEqual(early.shape, (8, 32, 32))
        self.assertTrue(np.array_equal(~present, block))
        self.assertTrue(np.all(early[2:7, block] == 0))
        self.assertTrue(np.all(early[7, block] == 0))
        self.assertTrue(np.all(early[7, present & tile.eligible] == 1))
        self.assertTrue(np.all(early[:, ~tile.eligible] == 0))
        absent, absent_present = model_inputs(tile, "robust_early", "missing_optical")
        self.assertFalse(absent_present.any())
        self.assertTrue(np.all(absent[2:8] == 0))

    def test_late_fusion_falls_back_to_radar_only_where_optical_absent(self):
        radar = np.array([[0.2, 0.8], [0.4, np.nan]], dtype=np.float32)
        optical = np.array([[0.6, 0.1], [0.9, np.nan]], dtype=np.float32)
        present = np.array([[True, False], [True, False]])
        score = late_fusion(radar, optical, present, radar_weight=0.5)
        self.assertAlmostEqual(float(score[0, 0]), 0.4)
        self.assertAlmostEqual(float(score[0, 1]), 0.8)
        self.assertAlmostEqual(float(score[1, 0]), 0.65)
        self.assertTrue(np.isnan(score[1, 1]))

    def test_tiny_fused_training_and_unknown_prediction(self):
        tile = self.tile("train")
        with tempfile.TemporaryDirectory() as directory:
            model, profile = train_model([tile], "robust_early", Path(directory) / "model.pt",
                                         seed=7, epochs=1, patches_per_tile=1, patch_size=16,
                                         base_channels=4, learning_rate=0.001,
                                         full_dropout_probability=1.0,
                                         block_dropout_probability=0.0, block_fraction=0.25)
            self.assertEqual(profile["method"], "robust_early")
            score, present = predict_model(model, tile, "robust_early", "missing_optical",
                                           block_fraction=0.25, brightening_mix=0.35)
            self.assertFalse(present.any())
            self.assertTrue(np.isnan(score[~tile.eligible]).all())
            self.assertTrue(np.isfinite(score[tile.eligible]).all())
        with self.assertRaises(ValueError):
            train_model([self.tile("val", split="validation")], "early", Path("unused.pt"),
                        seed=7, epochs=1, patches_per_tile=1, patch_size=16,
                        base_channels=4, learning_rate=0.001,
                        full_dropout_probability=0.3, block_dropout_probability=0.3,
                        block_fraction=0.25)


class MilestoneStatusTests(unittest.TestCase):
    def test_m4_status_requires_both_hash_valid_runs_on_same_m3_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, reference in (("M3-test", None), ("M4-test", "M3-test"),
                                      ("M4B-test", "M3-test")):
                folder = root / "runs" / run_id
                folder.mkdir(parents=True)
                manifest = {"run_id": run_id, "status": "complete"}
                if reference is not None:
                    manifest["m3_reference_run_id"] = reference
                payload = json.dumps(manifest).encode()
                (folder / "run_manifest.json").write_bytes(payload)
                (folder / "status.json").write_text(json.dumps({"run_id": run_id,
                    "status": "complete", "manifest_sha256": hashlib.sha256(payload).hexdigest()}))
            self.assertEqual(local_status(root)["status"], "m4_development")
            bad = root / "runs/M4B-test/status.json"
            contents = json.loads(bad.read_text())
            contents["manifest_sha256"] = "0" * 64
            bad.write_text(json.dumps(contents))
            self.assertEqual(local_status(root)["status"], "m3_development")


if __name__ == "__main__":
    unittest.main()
