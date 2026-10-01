"""M2 invariants on small, real-format fixtures without external source data."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, mapping

from flood_access.catalog import assign_event_split, boxes_overlap
from flood_access.case_vectors import _project, validate_features
from flood_access.raster_pipeline import S1_BANDS, S2_BANDS, canonicalize_tile, footprint_coverage


def _tiff(path: Path, data: np.ndarray, descriptions: tuple, nodata=None, transform=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis]
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
                       count=data.shape[0], dtype=str(data.dtype), crs="EPSG:4326",
                       transform=transform or from_origin(-1.0, 38.1, 0.001, 0.001),
                       nodata=nodata) as sink:
        sink.write(data)
        sink.descriptions = descriptions


class RasterPipelineTests(unittest.TestCase):
    def test_unknown_labels_nodata_and_cloud_uncertainty_survive_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "data/flood_events/HandLabeled"
            chip = "Spain_1"
            label = np.array([[-1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int16)
            radar = np.full((2, 4, 4), -12, dtype=np.float32)
            radar[:, 0, 0] = np.nan
            optical = np.full((13, 4, 4), 2000, dtype=np.int16)
            optical[:, 1, 1] = 0
            _tiff(base / "LabelHand" / f"{chip}_LabelHand.tif", label, ("",))
            _tiff(base / "S1Hand" / f"{chip}_S1Hand.tif", radar, S1_BANDS, np.nan)
            _tiff(base / "S2Hand" / f"{chip}_S2Hand.tif", optical, S2_BANDS, 0)
            tile = {"tile_id": chip, "event_id": "spain_20190917", "analysis_split": "case_study_exploratory",
                    "official_split": "official_train", "s1_date": "2019-09-17", "s2_date": "2019-09-18",
                    "sensor_offset_days": 1}
            output = root / "canonical"
            metadata = canonicalize_tile(root, tile, output)
            with rasterio.open(output / "label.tif") as labels, \
                 rasterio.open(output / "validity.tif") as validity, \
                 rasterio.open(output / "optical_toa.tif") as toa, \
                 rasterio.open(output / "cloud_status.tif") as cloud:
                self.assertEqual(labels.nodata, -1)
                self.assertEqual(labels.read(1)[0, 0], -1)
                self.assertEqual(validity.read(1)[0, 0], 0)
                self.assertEqual(validity.read(2)[0, 0], 0)
                self.assertEqual(validity.read(3)[1, 1], 0)
                self.assertEqual(validity.read(5)[1, 1], 0)
                self.assertEqual(validity.read(5)[2, 2], 1)
                self.assertTrue(np.isnan(toa.read(1)[1, 1]))
                self.assertAlmostEqual(float(toa.read(1)[2, 2]), 0.2)
                self.assertTrue(np.all(cloud.read(1) == 255))
            self.assertEqual(metadata["quality_counts"]["cloud_known"], 0)

    def test_wrong_band_order_fails_before_writing_canonical_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "data/flood_events/HandLabeled"
            _tiff(base / "LabelHand/X_LabelHand.tif", np.zeros((2, 2), dtype=np.int16), ("",))
            _tiff(base / "S1Hand/X_S1Hand.tif", np.ones((2, 2, 2), dtype=np.float32), ("VH", "VV"))
            _tiff(base / "S2Hand/X_S2Hand.tif", np.ones((13, 2, 2), dtype=np.int16), S2_BANDS)
            with self.assertRaisesRegex(ValueError, "band order"):
                canonicalize_tile(root, {"tile_id": "X"}, root / "out")

    def test_geographically_unpaired_sensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "data/flood_events/HandLabeled"
            reference = from_origin(-1.0, 38.1, 0.001, 0.001)
            far_away = from_origin(10.0, 38.1, 0.001, 0.001)
            _tiff(base / "LabelHand/X_LabelHand.tif", np.zeros((4, 4), dtype=np.int16), ("",), transform=reference)
            _tiff(base / "S1Hand/X_S1Hand.tif", np.ones((2, 4, 4), dtype=np.float32), S1_BANDS, transform=far_away)
            _tiff(base / "S2Hand/X_S2Hand.tif", np.ones((13, 4, 4), dtype=np.int16), S2_BANDS, transform=reference)
            with rasterio.open(base / "LabelHand/X_LabelHand.tif") as label, \
                 rasterio.open(base / "S1Hand/X_S1Hand.tif") as radar:
                self.assertEqual(footprint_coverage(radar, label), 0)
            with self.assertRaisesRegex(ValueError, "footprint"):
                canonicalize_tile(root, {"tile_id": "X"}, root / "out")


class SplitAndVectorTests(unittest.TestCase):
    def test_overlap_cannot_cross_an_analysis_split(self):
        items = [
            {"tile_id": "A_1", "event_location": "A", "event_id": "a", "bbox_epsg4326": [0, 0, 1, 1]},
            {"tile_id": "B_1", "event_location": "B", "event_id": "b", "bbox_epsg4326": [0.5, 0.5, 1.5, 1.5]},
        ]
        with self.assertRaisesRegex(ValueError, "Overlapping chips"):
            assign_event_split(items, {"event_groups": {"train": ["A"], "final_test": ["B"]}})
        self.assertFalse(boxes_overlap([0, 0, 1, 1], [1, 0, 2, 1]))

    def test_vector_validator_rejects_duplicate_ids_and_degree_mistakes(self):
        feature = {"type": "Feature", "id": "hospital_1", "geometry": mapping(Point(-0.7, 38.1)), "properties": {}}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_features([feature, feature], "point")
        projected = {**feature, "geometry": mapping(Point(700000, 4200000))}
        with self.assertRaisesRegex(ValueError, "not EPSG:4326"):
            validate_features([projected], "point")
        line = {"type": "LineString", "coordinates": [[-0.76, 38.1], [-0.75, 38.1]]}
        projected_line = _project(line, "EPSG:4326", "EPSG:25830")
        self.assertGreater(projected_line.length, 800)
        self.assertLess(projected_line.length, 1000)


if __name__ == "__main__":
    unittest.main()
