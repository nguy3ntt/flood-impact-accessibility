"""Canonical, source-preserving raster conversion for the Sen1Floods11 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds


S1_BANDS = ("VV", "VH")
S2_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")
VALIDITY_BANDS = ("label_valid", "radar_valid", "optical_nodata_valid", "paired_sensor_valid", "supervised_paired_valid")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_grid(source: rasterio.io.DatasetReader, reference: rasterio.io.DatasetReader) -> bool:
    return (
        source.crs == reference.crs
        and source.width == reference.width
        and source.height == reference.height
        and np.allclose(tuple(source.transform)[:6], tuple(reference.transform)[:6], rtol=0, atol=1e-10)
    )


def footprint_coverage(source: rasterio.io.DatasetReader, reference: rasterio.io.DatasetReader) -> float:
    """Fraction of the label grid covered by a sensor footprint in label CRS."""
    if source.crs is None or reference.crs is None:
        raise ValueError("Raster CRS is required")
    left, bottom, right, top = transform_bounds(source.crs, reference.crs, *source.bounds, densify_pts=21)
    ref = reference.bounds
    width = max(0.0, min(right, ref.right) - max(left, ref.left))
    height = max(0.0, min(top, ref.top) - max(bottom, ref.bottom))
    area = (ref.right - ref.left) * (ref.top - ref.bottom)
    return width * height / area if area > 0 else 0.0


def _continuous_on_grid(source: rasterio.io.DatasetReader, reference: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray, str]:
    if source.crs is None or reference.crs is None:
        raise ValueError("Raster CRS is required")
    data = np.empty((source.count, reference.height, reference.width), dtype=np.float32)
    valid_bands = np.empty_like(data, dtype=bool)
    aligned = same_grid(source, reference)
    for band in range(1, source.count + 1):
        raw = source.read(band).astype(np.float32)
        source_valid = (source.read_masks(band) > 0) & np.isfinite(raw)
        if aligned:
            values = raw.copy()
            valid = source_valid
        else:
            values = np.full((reference.height, reference.width), np.nan, dtype=np.float32)
            reproject(
                source=raw, destination=values,
                src_transform=source.transform, src_crs=source.crs,
                src_nodata=source.nodata, dst_transform=reference.transform,
                dst_crs=reference.crs, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            valid_u8 = np.zeros(values.shape, dtype=np.uint8)
            reproject(
                source=source_valid.astype(np.uint8), destination=valid_u8,
                src_transform=source.transform, src_crs=source.crs,
                src_nodata=0, dst_transform=reference.transform,
                dst_crs=reference.crs, dst_nodata=0,
                resampling=Resampling.nearest,
            )
            valid = (valid_u8 == 1) & np.isfinite(values)
        values[~valid] = np.nan
        data[band - 1] = values
        valid_bands[band - 1] = valid
    return data, np.all(valid_bands, axis=0), "identity" if aligned else "bilinear_data_nearest_validity"


def _write_raster(path: Path, data: np.ndarray, reference: rasterio.io.DatasetReader,
                  descriptions: tuple[str, ...], nodata: int | float | None) -> None:
    if data.ndim == 2:
        data = data[np.newaxis]
    profile = {
        "driver": "GTiff", "width": reference.width, "height": reference.height,
        "count": data.shape[0], "dtype": str(data.dtype), "crs": reference.crs,
        "transform": reference.transform, "nodata": nodata,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "compress": "deflate", "predictor": 3 if np.issubdtype(data.dtype, np.floating) else 2,
    }
    with rasterio.open(path, "w", **profile) as sink:
        sink.write(data)
        sink.descriptions = descriptions


def canonicalize_tile(raw_root: Path, tile: dict, output_dir: Path) -> dict:
    """Write one trio on the hand-label grid, with explicit unknown/valid masks."""
    chip = tile["tile_id"]
    base = raw_root / "data/flood_events/HandLabeled"
    source_paths = {
        layer: base / layer / f"{chip}_{layer}.tif"
        for layer in ("LabelHand", "S1Hand", "S2Hand")
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_paths["LabelHand"]) as label_source, \
         rasterio.open(source_paths["S1Hand"]) as radar_source, \
         rasterio.open(source_paths["S2Hand"]) as optical_source:
        if label_source.count != 1 or str(label_source.crs) != "EPSG:4326":
            raise ValueError(f"Invalid label grid for {chip}")
        label = label_source.read(1)
        if not np.isin(label, (-1, 0, 1)).all():
            raise ValueError(f"Unexpected label code for {chip}")
        if radar_source.descriptions != S1_BANDS or optical_source.descriptions != S2_BANDS:
            raise ValueError(f"Unexpected sensor band order for {chip}")
        coverage = {"radar": footprint_coverage(radar_source, label_source),
                    "optical": footprint_coverage(optical_source, label_source)}
        if min(coverage.values()) < 0.98:
            raise ValueError(f"Paired source footprint covers less than 98% of label grid for {chip}: {coverage}")
        radar, radar_valid, radar_method = _continuous_on_grid(radar_source, label_source)
        optical, optical_valid, optical_method = _continuous_on_grid(optical_source, label_source)
        optical /= np.float32(10000.0)
        label_valid = label != -1
        paired_valid = radar_valid & optical_valid
        supervised_paired_valid = label_valid & paired_valid
        validity = np.stack((label_valid, radar_valid, optical_valid, paired_valid,
                             supervised_paired_valid)).astype(np.uint8)
        cloud_status = np.full(label.shape, 255, dtype=np.uint8)
        _write_raster(output_dir / "label.tif", label.astype(np.int8), label_source, ("observed_water_label",), -1)
        _write_raster(output_dir / "radar_db.tif", radar, label_source, S1_BANDS, np.nan)
        _write_raster(output_dir / "optical_toa.tif", optical, label_source, S2_BANDS, np.nan)
        _write_raster(output_dir / "validity.tif", validity, label_source, VALIDITY_BANDS, None)
        _write_raster(output_dir / "cloud_status.tif", cloud_status, label_source, ("cloud_status_unknown_255",), 255)
        metadata = {
            "schema_version": "m2_tile_v1",
            "tile_id": chip,
            "event_id": tile["event_id"],
            "analysis_split": tile["analysis_split"],
            "official_split": tile["official_split"],
            "source_dates": {"radar": tile["s1_date"], "optical": tile["s2_date"]},
            "date_precision": "day_only; acquisition hours unverified",
            "sensor_offset_days": tile["sensor_offset_days"],
            "grid": {"crs": str(label_source.crs), "transform": list(label_source.transform)[:6],
                     "bounds": list(label_source.bounds), "width": label_source.width, "height": label_source.height},
            "bands": {"radar": list(S1_BANDS), "optical": list(S2_BANDS)},
            "units": {"radar": "dB", "optical": "TOA reflectance fraction = source int16 / 10000"},
            "alignment": {"radar": radar_method, "optical": optical_method,
                          "categorical_resampling": "label unchanged on its native grid; validity nearest if required"},
            "source_footprint_coverage": coverage,
            "label_semantics": {"-1": "unknown", "0": "observed non-water", "1": "observed water"},
            "cloud_status_codes": {"0": "clear", "1": "cloud", "255": "unknown; no source QA band"},
            "quality_counts": {
                "label_unknown": int((~label_valid).sum()),
                "radar_valid": int(radar_valid.sum()),
                "optical_nodata_valid": int(optical_valid.sum()),
                "paired_sensor_valid": int(paired_valid.sum()),
                "label_and_paired_valid": int(supervised_paired_valid.sum()),
                "cloud_known": 0,
            },
            "sources": {layer: {"path": path.as_posix(), "sha256": sha256(path)}
                        for layer, path in source_paths.items()},
            "interpretation": "Optical nodata-valid is not cloud-clear; water labels do not establish event-specific inundation.",
        }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata
