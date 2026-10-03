"""Verify M5 provenance, grouped metrics and every published development raster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import rasterio

from flood_access.evidence import (calibrated_probability, calibration_points,
                                   fit_monotone_logistic, quality_flags,
                                   retained_mask, spatial_scenarios)
from flood_access.mapping import load_tile
from flood_access.raster_pipeline import sha256
from run_m5_evidence import (CATALOG_PATH, CODE_FILES, CONFIG_PATH, QA_TILE_IDS,
                             _event_summary, _reference, _sensor_counts, _tile_metrics)


def _same(actual, expected) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _same(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and bool(
            np.isclose(actual, expected, rtol=0, atol=1e-10))
    return actual == expected


def _read_raster(path: Path, tile, bands: int, descriptions: list[str], nodata,
                 expected: np.ndarray) -> None:
    with rasterio.open(path) as source:
        if source.count != bands or source.shape != tile.label.shape or source.crs != tile.crs or \
           source.transform != tile.transform or source.descriptions != tuple(descriptions) or \
           not ((np.isnan(source.nodata) and np.isnan(nodata)) if isinstance(nodata, float) and np.isnan(nodata)
                else source.nodata == nodata):
            raise ValueError(f"M5 raster grid/band contract differs: {path}")
        observed = source.read()
    target = expected[None] if expected.ndim == 2 else expected
    if observed.dtype != target.dtype or not np.array_equal(observed, target, equal_nan=True):
        raise ValueError(f"M5 raster values differ: {path}")


def verify(run_id: str) -> dict:
    if re.fullmatch(r"M5-[0-9a-f]{12}", run_id) is None:
        raise ValueError("Invalid M5 run ID")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    m3, m4, catalog, base = _reference(config)
    code_hashes = {path.as_posix(): sha256(path) for path in CODE_FILES}
    fingerprint = {"m3_manifest_sha256": sha256(Path("runs") / m3["run_id"] / "run_manifest.json"),
                   "m4_control_manifest_sha256": sha256(Path("runs") / m4["run_id"] / "run_manifest.json"),
                   "config": config, "code_sha256": code_hashes}
    expected_id = "M5-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    if run_id != expected_id:
        raise ValueError("M5 fingerprint differs from code/config/references")
    root, report_root = Path("runs") / run_id, Path("reports/m5") / run_id
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id or manifest["status"] != "complete" or \
       status["status"] != "complete" or status["manifest_sha256"] != sha256(manifest_path) or \
       manifest["fingerprint"] != fingerprint or manifest["config_sha256"] != sha256(CONFIG_PATH) or \
       manifest["catalog_sha256"] != sha256(CATALOG_PATH) or manifest["code_sha256"] != code_hashes:
        raise ValueError("M5 manifest/status/provenance differs")
    outputs = manifest["output_sha256"]
    actual = {path.as_posix() for folder in (root, report_root) for path in folder.rglob("*")
              if path.is_file() and path not in (root / "status.json", manifest_path)}
    if set(outputs) != actual:
        raise ValueError("M5 output file set differs")
    for name, digest in outputs.items():
        path = Path(name)
        if ".." in path.parts or not (path.is_relative_to(root) or path.is_relative_to(report_root)) or \
           sha256(path) != digest:
            raise ValueError(f"M5 output changed: {name}")
    sample_sets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    calibration_ids = [tile_id for tile_id in m3["train_tile_ids"]
                       if catalog[tile_id]["event_id"] in config["calibration_events"]]
    internal_ids = [tile_id for tile_id in m3["train_tile_ids"]
                    if catalog[tile_id]["event_id"] in config["internal_check_events"]]
    if manifest["calibration_tile_ids"] != calibration_ids or manifest["internal_tile_ids"] != internal_ids or \
       manifest["development_tile_ids"] != m3["validation_tile_ids"] or \
       manifest["calibration_event_ids"] != config["calibration_events"] or \
       manifest["internal_check_event_ids"] != config["internal_check_events"] or \
       manifest["development_event_ids"] != config["development_evaluation_events"] or \
       manifest["source_sha256"] != m3["source_sha256"]:
        raise ValueError("M5 event/tile/source split differs")
    for tile_id in calibration_ids:
        tile = load_tile(base / tile_id, catalog[tile_id])
        sample_sets.setdefault(tile.event_id, []).append(
            calibration_points(tile, config["calibration"]["sample_block_pixels"]))
    samples = {event: (np.concatenate([part[0] for part in parts]),
                       np.concatenate([part[1] for part in parts]))
               for event, parts in sample_sets.items()}
    fit = fit_monotone_logistic(samples, steps=config["calibration"]["steps"],
                                learning_rate=config["calibration"]["learning_rate"],
                                slope_l2=config["calibration"]["slope_l2"])
    stored_fit = json.loads((root / "calibrator.json").read_text(encoding="utf-8"))
    if not _same(stored_fit, fit):
        raise ValueError("M5 calibrator differs on replay")
    stored_rows = json.loads((root / "per_tile.json").read_text(encoding="utf-8"))
    evaluation_ids = [*internal_ids, *m3["validation_tile_ids"]]
    if [row["tile_id"] for row in stored_rows] != evaluation_ids:
        raise ValueError("M5 evaluation tile list differs")
    rows = []
    for index, tile_id in enumerate(evaluation_ids):
        tile = load_tile(base / tile_id, catalog[tile_id])
        p = calibrated_probability(tile, fit)
        flags = quality_flags(tile,
                              weak_denominator_below=config["quality"]["weak_index_denominator_below"],
                              radiometric_extreme_above=config["quality"]["radiometric_extreme_above_toa"])
        cases, names = spatial_scenarios(
            p, tile.eligible, tile.tile_id, block_pixels=config["scenarios"]["block_pixels"],
            thresholds=config["scenarios"]["fixed_probability_thresholds"],
            offset_seeds=config["scenarios"]["structured_offset_seeds"],
            maximum_offset=config["scenarios"]["maximum_block_probability_offset"])
        row = _tile_metrics(tile, p, flags, cases, names, fit, config,
                            _sensor_counts(base / tile_id, tile))
        if not _same(row, stored_rows[index]):
            raise ValueError(f"M5 tile metrics differ: {tile_id}")
        rows.append(row)
        if tile.split == "validation":
            raster_root = root / "tiles" / tile_id
            _read_raster(raster_root / "probability.tif", tile, 1,
                         ["empirical_observed_water_score"], float("nan"), p)
            _read_raster(raster_root / "quality_flags.tif", tile, 1,
                         ["bit1_weak_denominator_bit2_bright_toa_255_unknown"], 255, flags)
            retained = retained_mask(tile, p, flags, ambiguity_margin=0.15, require_quality=True)
            state = np.full(tile.label.shape, 255, dtype=np.uint8)
            state[tile.eligible] = 0
            state[retained] = 1
            _read_raster(raster_root / "retained.tif", tile, 1,
                         ["1_retained_0_abstained_255_source_unknown"], 255, state)
            _read_raster(raster_root / "scenarios.tif", tile, len(names), names, 255, cases)
    event_rows = _event_summary(rows, config)
    if not _same(event_rows, json.loads((root / "per_event.json").read_text(encoding="utf-8"))):
        raise ValueError("M5 per-event aggregates differ")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    for scope, groups in (("internal_calibrator_holdout", config["internal_check_events"]),
                          ("development_validation", config["development_evaluation_events"])):
        entry = summary["scopes"][scope]
        if entry["event_ids"] != groups:
            raise ValueError("M5 summary event scope differs")
        for field, key in (("event_macro_calibrated_brier", "calibrated"),
                           ("event_macro_unit_scaled_brier", "unit_scaled_ndwi"),
                           ("event_macro_constant_brier", "calibration_prevalence")):
            value = float(np.mean([event_rows[event]["reliability"][key]["brier"] for event in groups]))
            if not _same(entry[field], value):
                raise ValueError(f"M5 summary metric differs: {field}")
        ece = float(np.mean([event_rows[event]["reliability"]["calibrated"]["ece"] for event in groups]))
        if not _same(entry["event_macro_calibrated_ece"], ece):
            raise ValueError("M5 summary ECE differs")
    if summary["run_id"] != run_id or summary["calibration_events"] != config["calibration_events"] or \
       summary["internal_check_events"] != config["internal_check_events"] or \
       summary["development_events"] != config["development_evaluation_events"]:
        raise ValueError("M5 summary group identity differs")
    if any(not (report_root / f"qa_{tile_id}.png").is_file() for tile_id in QA_TILE_IDS):
        raise ValueError("M5 QA panels missing")
    return {"run_id": run_id, "status": "verified", "calibration_events": len(samples),
            "internal_tiles": len(internal_ids), "development_rasters_replayed": 4 * len(m3["validation_tile_ids"]),
            "output_files": len(outputs), "manifest_sha256": sha256(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id), indent=2))


if __name__ == "__main__":
    main()
