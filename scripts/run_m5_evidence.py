"""Fit and audit M5 observed-water calibration without opening final-test labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import rasterio

from flood_access.evidence import (calibrated_probability, calibration_points,
                                   fit_monotone_logistic, hashed_event_rank,
                                   quality_flags, reliability, retained_mask,
                                   scenario_summary, spatial_scenarios,
                                   unit_scaled_ndwi)
from flood_access.mapping import load_tile
from flood_access.raster_pipeline import sha256
from verify_m3_run import verify as verify_m3


CONFIG_PATH = Path("configs/m5_evidence_v1.json")
CATALOG_PATH = Path("data/processed/m2_runs/M2-151ccd74b14c/catalog.json")
CODE_FILES = (Path("scripts/run_m5_evidence.py"), Path("src/flood_access/evidence.py"),
              Path("src/flood_access/mapping.py"), Path("src/flood_access/raster_pipeline.py"))
QA_TILE_IDS = ("Mekong_52610", "Ghana_866994", "Mekong_1111068")


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _reference(config: dict) -> tuple[dict, dict, dict, Path]:
    m3_verified = verify_m3(config["m3_reference_run_id"])
    m3 = json.loads((Path("runs") / config["m3_reference_run_id"] / "run_manifest.json").read_text(encoding="utf-8"))
    if m3["plan_id"] != config["m3_plan_id"] or m3_verified["manifest_sha256"] != \
       sha256(Path("runs") / config["m3_reference_run_id"] / "run_manifest.json"):
        raise ValueError("M5 canonical reference differs")
    m4_id = config["m4_control_run_id"]
    if re.fullmatch(r"M4B-[0-9a-f]{12}", m4_id) is None:
        raise ValueError("Invalid controlled M4 reference ID")
    m4_root = Path("runs") / m4_id
    m4_path = m4_root / "run_manifest.json"
    m4 = json.loads(m4_path.read_text(encoding="utf-8"))
    m4_status = json.loads((m4_root / "status.json").read_text(encoding="utf-8"))
    if m4["status"] != "complete" or m4["m3_reference_run_id"] != m3["run_id"] or \
       m4_status["status"] != "complete" or m4_status["manifest_sha256"] != sha256(m4_path):
        raise ValueError("M5 controlled M4 reference differs")
    catalog = {row["tile_id"]: row for row in json.loads(CATALOG_PATH.read_text(encoding="utf-8"))}
    training_events = set(m3["train_event_ids"])
    rank = hashed_event_rank(m3["train_event_ids"], config["calibration_event_rank_salt"])
    chosen = config["calibration_events"]
    if chosen != rank[:len(chosen)] or config["internal_check_events"] != rank[len(chosen):] or \
       len(chosen) < 2 or set(chosen) | set(config["internal_check_events"]) != training_events or \
       set(chosen) & set(config["internal_check_events"]) or \
       set(config["development_evaluation_events"]) != set(m3["validation_event_ids"]):
        raise ValueError("M5 calibration and development groups differ from frozen split/rank")
    for split, ids in (("train", m3["train_tile_ids"]), ("validation", m3["validation_tile_ids"])):
        if any(catalog[tile_id]["analysis_split"] != split for tile_id in ids):
            raise ValueError("M5 tile crosses frozen split")
    base = Path("data/processed/m3_development") / config["m3_plan_id"] / "tiles"
    return m3, m4, catalog, base


def _raster(path: Path, data: np.ndarray, tile, descriptions: list[str], nodata) -> None:
    array = data[None] if data.ndim == 2 else data
    if array.shape[1:] != tile.label.shape or len(array) != len(descriptions):
        raise ValueError("M5 raster shape or band contract differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=tile.label.shape[0], width=tile.label.shape[1],
                       count=len(array), dtype=array.dtype, crs=tile.crs, transform=tile.transform,
                       nodata=nodata, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", predictor=3 if np.issubdtype(array.dtype, np.floating) else 2) as sink:
        sink.write(array)
        sink.descriptions = tuple(descriptions)


def _sensor_counts(tile_dir: Path, tile) -> dict:
    with rasterio.open(tile_dir / "validity.tif") as source:
        masks = source.read() == 1
    labelled = tile.label != -1
    return {"labelled_pixels": int(np.count_nonzero(labelled)),
            "radar_source_valid_labelled": int(np.count_nonzero(labelled & masks[1])),
            "optical_source_valid_labelled": int(np.count_nonzero(labelled & masks[2])),
            "paired_source_valid_labelled": int(np.count_nonzero(labelled & masks[3])),
            "common_eligible_labelled": int(np.count_nonzero(tile.eligible)),
            "cloud_status_known_pixels": 0}


def aggregate_reliability(rows: list[dict], bins: int) -> dict:
    count = sum(row["pixels"] for row in rows)
    if not count:
        return {"pixels": 0, "prevalence": None, "brier": None, "log_loss": None,
                "ece": None, "bins": []}
    result_bins = []
    for index in range(bins):
        parts = [row["bins"][index] for row in rows if row["pixels"]]
        n = sum(part["pixels"] for part in parts)
        mean_p = sum(part["pixels"] * part["mean_probability"] for part in parts if part["pixels"]) / n if n else None
        observed = sum(part["pixels"] * part["observed_fraction"] for part in parts if part["pixels"]) / n if n else None
        result_bins.append({"bin": index, "lower": index / bins, "upper": (index + 1) / bins,
                            "pixels": n, "mean_probability": mean_p, "observed_fraction": observed})
    return {"pixels": count,
            "prevalence": sum(row["pixels"] * row["prevalence"] for row in rows if row["pixels"]) / count,
            "brier": sum(row["pixels"] * row["brier"] for row in rows if row["pixels"]) / count,
            "log_loss": sum(row["pixels"] * row["log_loss"] for row in rows if row["pixels"]) / count,
            "ece": sum(part["pixels"] / count * abs(part["mean_probability"] - part["observed_fraction"])
                       for part in result_bins if part["pixels"]),
            "bins": result_bins}


def _tile_metrics(tile, probability: np.ndarray, flags: np.ndarray, scenarios: np.ndarray,
                  names: list[str], fit: dict, config: dict, sensor: dict) -> dict:
    bins = config["calibration"]["reliability_bins"]
    support = tile.eligible
    reference = unit_scaled_ndwi(tile)
    constant = np.full(tile.label.shape, np.nan, dtype=np.float32)
    constant[support] = fit["event_equal_sample_prevalence"]
    reliability_rows = {"calibrated": reliability(tile.label, probability, support, bins),
                        "unit_scaled_ndwi": reliability(tile.label, reference, support, bins),
                        "calibration_prevalence": reliability(tile.label, constant, support, bins)}
    quality_rows = {}
    for name, mask in (("ordinary", support & (flags == 0)),
                       ("weak_denominator", support & ((flags & 1) != 0)),
                       ("bright_toa_extreme", support & ((flags & 2) != 0))):
        quality_rows[name] = reliability(tile.label, probability, mask, bins)
    abstention = []
    for margin in config["quality"]["ambiguity_margins"]:
        for quality_gate in (False, True):
            kept = retained_mask(tile, probability, flags, ambiguity_margin=margin,
                                 require_quality=quality_gate)
            count = int(np.count_nonzero(kept))
            errors = int(np.count_nonzero(kept & ((probability >= 0.5) != (tile.label == 1))))
            withheld_water = int(np.count_nonzero(support & (tile.label == 1) & ~kept))
            abstention.append({"ambiguity_margin": margin, "quality_gate": quality_gate,
                               "retained_pixels": count,
                               "retained_fraction_common": count / sensor["common_eligible_labelled"]
                               if sensor["common_eligible_labelled"] else None,
                               "retained_fraction_labelled": count / sensor["labelled_pixels"]
                               if sensor["labelled_pixels"] else None,
                               "conditional_error_rate": errors / count if count else None,
                               "errors_on_retained": errors,
                               "withheld_labelled_water": withheld_water,
                               "retained_brier": reliability(tile.label, probability, kept, bins)["brier"]})
    return {"tile_id": tile.tile_id, "event_id": tile.event_id, "analysis_split": tile.split,
            "sensor": sensor, "reliability": reliability_rows, "quality_strata": quality_rows,
            "abstention": abstention,
            "scenarios": scenario_summary(scenarios, support, names)}


def _event_summary(rows: list[dict], config: dict) -> dict:
    bins = config["calibration"]["reliability_bins"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["event_id"]].append(row)
    result = {}
    for event, values in sorted(grouped.items()):
        sensor = {key: sum(row["sensor"][key] for row in values) for key in values[0]["sensor"]}
        references = {name: aggregate_reliability([row["reliability"][name] for row in values], bins)
                      for name in values[0]["reliability"]}
        quality = {name: aggregate_reliability([row["quality_strata"][name] for row in values], bins)
                   for name in values[0]["quality_strata"]}
        abstention = []
        for index in range(len(values[0]["abstention"])):
            parts = [row["abstention"][index] for row in values]
            kept = sum(part["retained_pixels"] for part in parts)
            errors = sum(part["errors_on_retained"] for part in parts)
            abstention.append({"ambiguity_margin": parts[0]["ambiguity_margin"],
                               "quality_gate": parts[0]["quality_gate"],
                               "retained_pixels": kept,
                               "retained_fraction_common": kept / sensor["common_eligible_labelled"]
                               if sensor["common_eligible_labelled"] else None,
                               "retained_fraction_labelled": kept / sensor["labelled_pixels"]
                               if sensor["labelled_pixels"] else None,
                               "conditional_error_rate": errors / kept if kept else None,
                               "errors_on_retained": errors,
                               "withheld_labelled_water": sum(part["withheld_labelled_water"] for part in parts),
                               "retained_brier": sum(part["retained_pixels"] * part["retained_brier"]
                                                     for part in parts if part["retained_pixels"]) / kept if kept else None})
        scenario_names = list(values[0]["scenarios"]["water_pixels_by_scenario"])
        scenario_water = {name: sum(row["scenarios"]["water_pixels_by_scenario"][name] for row in values)
                          for name in scenario_names}
        common = sensor["common_eligible_labelled"]
        scenario = {"eligible_pixels": common,
                    "water_pixels_by_scenario": scenario_water,
                    "water_fraction_by_scenario": {name: count / common if common else None
                                                   for name, count in scenario_water.items()},
                    "unanimous_fraction": sum(row["scenarios"]["eligible_pixels"] *
                                              row["scenarios"]["unanimous_fraction"] for row in values
                                              if row["scenarios"]["eligible_pixels"]) / common if common else None,
                    "frequency_interpretation": "Designed block-offset mask fraction, not a probability."}
        result[event] = {"tiles": len(values), "analysis_split": values[0]["analysis_split"],
                         "sensor": sensor, "reliability": references,
                         "quality_strata": quality, "abstention": abstention, "scenarios": scenario}
    return result


def _qa_panel(path: Path, tile, probability: np.ndarray, flags: np.ndarray,
              scenarios: np.ndarray, retained: np.ndarray) -> None:
    scale = 400
    image = Image.new("RGB", (5 * scale, scale + 65), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    label = np.full((*tile.label.shape, 3), 125, dtype=np.uint8)
    label[tile.eligible & (tile.label == 0)] = (45, 48, 50)
    label[tile.eligible & (tile.label == 1)] = (42, 175, 93)
    probability_rgb = np.full((*tile.label.shape, 3), 125, dtype=np.uint8)
    gray = np.clip((probability[tile.eligible] * 255), 0, 255).astype(np.uint8)
    probability_rgb[tile.eligible] = np.column_stack((gray, 60 + gray // 2, 255 - gray))
    central = np.full((*tile.label.shape, 3), 125, dtype=np.uint8)
    central[tile.eligible & (scenarios[1] == 0)] = (45, 48, 50)
    central[tile.eligible & (scenarios[1] == 1)] = (42, 175, 93)
    spread = np.full((*tile.label.shape, 3), 125, dtype=np.uint8)
    spread[tile.eligible & (scenarios[0] == 0) & (scenarios[2] == 0)] = (45, 48, 50)
    spread[tile.eligible & (scenarios[0] == 1) & (scenarios[2] == 1)] = (42, 175, 93)
    spread[tile.eligible & (scenarios[0] != scenarios[2])] = (235, 172, 45)
    keep = np.full((*tile.label.shape, 3), 125, dtype=np.uint8)
    keep[tile.eligible & ~retained] = (235, 172, 45)
    keep[retained & (tile.label == 0)] = (45, 48, 50)
    keep[retained & (tile.label == 1)] = (42, 175, 93)
    for index, (title, pixels) in enumerate((("source label", label), ("empirical score", probability_rgb),
                                              ("central threshold case", central), ("fixed-case disagreement", spread),
                                              ("retained evidence", keep))):
        image.paste(Image.fromarray(pixels).resize((scale, scale), Image.Resampling.NEAREST),
                    (index * scale, 28))
        draw.text((index * scale + 5, 7), title, fill=(0, 0, 0))
    draw.text((5, scale + 35), f"{tile.tile_id} | gray unknown; orange designed-case disagreement/abstention; no road or closure claim",
              fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    m3, m4, catalog, base = _reference(config)
    code_hashes = {path.as_posix(): sha256(path) for path in CODE_FILES}
    fingerprint = {"m3_manifest_sha256": sha256(Path("runs") / m3["run_id"] / "run_manifest.json"),
                   "m4_control_manifest_sha256": sha256(Path("runs") / m4["run_id"] / "run_manifest.json"),
                   "config": config, "code_sha256": code_hashes}
    run_id = "M5-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    root, report_root = Path("runs") / run_id, Path("reports/m5") / run_id
    status_path = root / "status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete":
            raise FileExistsError(f"Completed M5 run exists: {run_id}; verify it instead")
        raise FileExistsError(f"Incomplete M5 run exists: {run_id}; preserve it and revise code/config to retry")
    write_json(status_path, {"run_id": run_id, "status": "running",
                             "started_at_utc": datetime.now(timezone.utc).isoformat()})
    start = time.perf_counter()
    try:
        sample_sets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        for tile_id in m3["train_tile_ids"]:
            expected = catalog[tile_id]
            if expected["event_id"] in config["calibration_events"]:
                tile = load_tile(base / tile_id, expected)
                sample_sets[tile.event_id].append(calibration_points(tile, config["calibration"]["sample_block_pixels"]))
        samples = {event: (np.concatenate([part[0] for part in parts]),
                           np.concatenate([part[1] for part in parts]))
                   for event, parts in sample_sets.items()}
        if set(samples) != set(config["calibration_events"]):
            raise ValueError("Missing M5 calibration event")
        fit = fit_monotone_logistic(samples, steps=config["calibration"]["steps"],
                                    learning_rate=config["calibration"]["learning_rate"],
                                    slope_l2=config["calibration"]["slope_l2"])
        write_json(root / "calibrator.json", fit)
        rows = []
        internal_ids = [tile_id for tile_id in m3["train_tile_ids"]
                        if catalog[tile_id]["event_id"] in config["internal_check_events"]]
        evaluation_ids = [*internal_ids, *m3["validation_tile_ids"]]
        for index, tile_id in enumerate(evaluation_ids):
            expected = catalog[tile_id]
            tile_dir = base / tile_id
            tile = load_tile(tile_dir, expected)
            probability = calibrated_probability(tile, fit)
            flags = quality_flags(tile,
                                  weak_denominator_below=config["quality"]["weak_index_denominator_below"],
                                  radiometric_extreme_above=config["quality"]["radiometric_extreme_above_toa"])
            scenarios, names = spatial_scenarios(
                probability, tile.eligible, tile.tile_id,
                block_pixels=config["scenarios"]["block_pixels"],
                thresholds=config["scenarios"]["fixed_probability_thresholds"],
                offset_seeds=config["scenarios"]["structured_offset_seeds"],
                maximum_offset=config["scenarios"]["maximum_block_probability_offset"])
            sensor = _sensor_counts(tile_dir, tile)
            rows.append(_tile_metrics(tile, probability, flags, scenarios, names, fit, config, sensor))
            if tile.split == "validation":
                raster_root = root / "tiles" / tile_id
                _raster(raster_root / "probability.tif", probability, tile,
                        ["empirical_observed_water_score"], np.nan)
                _raster(raster_root / "quality_flags.tif", flags, tile,
                        ["bit1_weak_denominator_bit2_bright_toa_255_unknown"], 255)
                retained = retained_mask(tile, probability, flags, ambiguity_margin=0.15,
                                         require_quality=True)
                state = np.full(tile.label.shape, 255, dtype=np.uint8)
                state[tile.eligible] = 0
                state[retained] = 1
                _raster(raster_root / "retained.tif", state, tile,
                        ["1_retained_0_abstained_255_source_unknown"], 255)
                _raster(raster_root / "scenarios.tif", scenarios, tile, names, 255)
                if tile_id in QA_TILE_IDS:
                    _qa_panel(report_root / f"qa_{tile_id}.png", tile, probability, flags, scenarios, retained)
            if (index + 1) % 15 == 0:
                print(json.dumps({"run_id": run_id, "evaluated_tiles": index + 1,
                                  "total_tiles": len(evaluation_ids)}), flush=True)
        write_json(root / "per_tile.json", rows)
        events = _event_summary(rows, config)
        write_json(root / "per_event.json", events)
        split_summary = {}
        for scope, group in (("internal_calibrator_holdout", config["internal_check_events"]),
                             ("development_validation", config["development_evaluation_events"])):
            split_summary[scope] = {"event_ids": group,
                                    "event_macro_calibrated_brier": float(np.mean([
                                        events[event]["reliability"]["calibrated"]["brier"] for event in group])),
                                    "event_macro_unit_scaled_brier": float(np.mean([
                                        events[event]["reliability"]["unit_scaled_ndwi"]["brier"] for event in group])),
                                    "event_macro_constant_brier": float(np.mean([
                                        events[event]["reliability"]["calibration_prevalence"]["brier"] for event in group])),
                                    "event_macro_calibrated_ece": float(np.mean([
                                        events[event]["reliability"]["calibrated"]["ece"] for event in group]))}
        write_json(root / "summary.json", {"schema_version": "m5_evidence_summary_v1", "run_id": run_id,
                                           "calibration_events": config["calibration_events"],
                                           "internal_check_events": config["internal_check_events"],
                                           "development_events": config["development_evaluation_events"],
                                           "scopes": split_summary,
                                           "interpretation": ["Empirical observed-water score only; no unseen-event calibration guarantee.",
                                                              "Cloud status is unknown; brightness and denominator flags are proxies, not cloud labels.",
                                                              "Spatial scenario members are designed sensitivity cases, not posterior draws.",
                                                              "Final Nigeria/Somalia test labels remain unopened."]})
        outputs = sorted(path for folder in (root, report_root) for path in folder.rglob("*")
                         if path.is_file() and path not in (status_path, root / "run_manifest.json"))
        manifest = {"schema_version": "m5_evidence_run_v1", "run_id": run_id, "status": "complete",
                    "fingerprint": fingerprint, "config_sha256": sha256(CONFIG_PATH),
                    "code_sha256": code_hashes, "catalog_sha256": sha256(CATALOG_PATH),
                    "m3_reference_run_id": m3["run_id"], "m4_control_run_id": m4["run_id"],
                    "calibration_event_ids": config["calibration_events"],
                    "internal_check_event_ids": config["internal_check_events"],
                    "development_event_ids": config["development_evaluation_events"],
                    "calibration_tile_ids": [t for t in m3["train_tile_ids"]
                                             if catalog[t]["event_id"] in config["calibration_events"]],
                    "internal_tile_ids": internal_ids, "development_tile_ids": m3["validation_tile_ids"],
                    "source_sha256": m3["source_sha256"],
                    "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                    "rasterio": rasterio.__version__},
                    "output_sha256": {path.as_posix(): sha256(path) for path in outputs},
                    "wall_seconds": time.perf_counter() - start}
        write_json(root / "run_manifest.json", manifest)
        write_json(status_path, {"run_id": run_id, "status": "complete",
                                 "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "manifest_sha256": sha256(root / "run_manifest.json")})
        return {"run_id": run_id, "status": "complete", "calibration_events": len(samples),
                "internal_tiles": len(internal_ids), "development_tiles": len(m3["validation_tile_ids"]),
                "output_files": len(outputs), "wall_seconds": manifest["wall_seconds"],
                "development_event_macro_calibrated_brier":
                split_summary["development_validation"]["event_macro_calibrated_brier"]}
    except Exception as error:
        write_json(status_path, {"run_id": run_id, "status": "failed",
                                 "error": f"{type(error).__name__}: {error}",
                                 "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
