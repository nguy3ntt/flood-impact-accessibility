"""Train and evaluate M3 development baselines without opening final-test labels.

Input is a completed, source-verified M3 staging/preparation plan. All results
stay under ignored runs/reports roots because source redistribution is unresolved.
"""

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
import torch

from flood_access.mapping import (confusion, fit_logistic, load_tile, metrics,
                                  optical_index, predict_logistic,
                                  select_threshold, stratified_training_pixels,
                                  sum_confusions)
from flood_access.mapping_train import predict_unet, train_unet
from flood_access.raster_pipeline import sha256


CONFIG_PATH = Path("configs/m3_baselines_v1.json")
CATALOG_PATH = Path("data/processed/m2_runs/M2-151ccd74b14c/catalog.json")
CODE_FILES = (Path("scripts/run_m3_baselines.py"), Path("src/flood_access/mapping.py"),
              Path("src/flood_access/mapping_train.py"), Path("src/flood_access/raster_pipeline.py"))
METHODS = ("radar_vh", "ndwi", "mndwi", "logistic", "radar_unet", "optical_unet")


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _score_raster(path: Path, score: np.ndarray, tile, description: str) -> None:
    if score.shape != tile.label.shape or np.any(~tile.eligible & np.isfinite(score)) or \
       not np.isfinite(score[tile.eligible]).all():
        raise ValueError(f"Prediction does not preserve eligible/unknown pixels: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=score.shape[1], height=score.shape[0],
                       count=1, dtype="float32", crs=tile.crs, transform=tile.transform,
                       nodata=np.nan, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", predictor=3) as sink:
        sink.write(score.astype(np.float32), 1)
        sink.set_band_description(1, description)


def _mask_raster(path: Path, tile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=tile.label.shape[1], height=tile.label.shape[0],
                       count=1, dtype="uint8", crs=tile.crs, transform=tile.transform,
                       compress="deflate", nodata=None) as sink:
        sink.write(tile.eligible.astype(np.uint8), 1)
        sink.set_band_description(1, "common_eligible_label_and_both_sensors_and_indices")


def _render_errors(path: Path, tile, predictions: dict[str, np.ndarray],
                   thresholds: dict[str, float]) -> None:
    scale = 256
    width = scale * (len(METHODS) + 1)
    image = Image.new("RGB", (width, scale + 68), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    reference = np.zeros((*tile.label.shape, 3), dtype=np.uint8)
    reference[:] = (130, 130, 130)
    reference[tile.eligible & (tile.label == 0)] = (56, 52, 48)
    reference[tile.eligible & (tile.label == 1)] = (40, 110, 210)
    panels = [("Observed label", reference)]
    for method in METHODS:
        prediction = predictions[method] >= thresholds[method]
        rgb = np.zeros((*tile.label.shape, 3), dtype=np.uint8)
        rgb[:] = (130, 130, 130)
        rgb[tile.eligible & (tile.label == 0) & ~prediction] = (56, 52, 48)
        rgb[tile.eligible & (tile.label == 1) & prediction] = (42, 170, 92)
        rgb[tile.eligible & (tile.label == 0) & prediction] = (226, 66, 55)
        rgb[tile.eligible & (tile.label == 1) & ~prediction] = (45, 116, 230)
        panels.append((method, rgb))
    for index, (title, rgb) in enumerate(panels):
        image.paste(Image.fromarray(rgb).resize((scale, scale), Image.Resampling.NEAREST),
                    (index * scale, 28))
        draw.text((index * scale + 5, 7), title, fill=(0, 0, 0))
    draw.text((5, scale + 35),
              f"{tile.tile_id} | gray unsupported, dark correct dry, green correct water, red false water, blue missed water",
              fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _render_curves(path: Path, threshold_curves: dict, profiles: dict) -> None:
    image = Image.new("RGB", (1240, 780), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    colours = {"radar_vh": (198, 72, 50), "ndwi": (35, 112, 210),
               "mndwi": (75, 153, 84), "radar": (198, 72, 50), "optical": (35, 112, 210)}
    boxes = {"radar_vh": (65, 45, 545, 330), "ndwi": (685, 45, 1165, 330),
             "mndwi": (65, 430, 545, 715), "unet": (685, 430, 1165, 715)}
    for name, box in boxes.items():
        draw.rectangle(box, outline=(70, 70, 70), width=2)
        draw.text((box[0], box[1] - 20),
                  f"{name}: train-event macro IoU by threshold" if name != "unet" else "Masked U-Net training loss by epoch",
                  fill=(0, 0, 0))
    for method, curve in threshold_curves.items():
        left, top, right, bottom = boxes[method]
        points = [(row["threshold"], row["event_macro_iou"]) for row in curve
                  if row["event_macro_iou"] is not None]
        if len(points) >= 2:
            xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
            screen = [(int(left + 5 + (right - left - 10) * (x - xmin) / (xmax - xmin)),
                       int(bottom - 5 - (bottom - top - 10) * y)) for x, y in points]
            draw.line(screen, fill=colours[method], width=3)
            draw.text((left, bottom + 6), f"threshold {xmin:g}", fill=(0, 0, 0))
            draw.text((right - 90, bottom + 6), f"{xmax:g}", fill=(0, 0, 0))
        draw.text((left - 30, top), "1.0", fill=(0, 0, 0))
        draw.text((left - 24, bottom - 10), "0", fill=(0, 0, 0))
    all_losses = [entry["train_loss"] for profile in profiles.values() for entry in profile["curve"]]
    ymin, ymax = min(all_losses), max(all_losses)
    span = max(ymax - ymin, 1e-6)
    left, top, right, bottom = boxes["unet"]
    for modality, profile in profiles.items():
        curve = profile["curve"]
        if len(curve) == 1:
            points = [(left + 5, int(bottom - 5 - (bottom - top - 10) * (curve[0]["train_loss"] - ymin) / span))]
            draw.ellipse((points[0][0] - 4, points[0][1] - 4,
                          points[0][0] + 4, points[0][1] + 4), fill=colours[modality])
        else:
            points = [(int(left + 5 + (right - left - 10) * i / (len(curve) - 1)),
                       int(bottom - 5 - (bottom - top - 10) * (row["train_loss"] - ymin) / span))
                      for i, row in enumerate(curve)]
            draw.line(points, fill=colours[modality], width=3)
        draw.text((left + list(profiles).index(modality) * 120, bottom + 24), modality,
                  fill=colours[modality])
    draw.text((left - 45, top), f"{ymax:.2f}", fill=(0, 0, 0))
    draw.text((left - 45, bottom - 10), f"{ymin:.2f}", fill=(0, 0, 0))
    draw.text((left, bottom + 6), "epoch 1", fill=(0, 0, 0))
    draw.text((right - 70, bottom + 6), f"{max(len(p['curve']) for p in profiles.values())}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _load_plan(plan_id: str) -> tuple[dict, dict, list, Path]:
    if re.fullmatch(r"M3S-[0-9a-f]{12}", plan_id) is None:
        raise ValueError("Invalid M3 staging plan ID")
    staging = Path("runs/m3_staging") / plan_id
    plan_path = staging / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    status = json.loads((staging / "preparation_status.json").read_text(encoding="utf-8"))
    base = Path("data/processed/m3_development") / plan_id
    prep_path = base / "preparation_manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if plan["plan_id"] != plan_id or status["status"] != "complete" or \
       status["manifest_sha256"] != sha256(prep_path) or prep["plan_sha256"] != sha256(plan_path) or \
       prep["catalog_sha256"] != sha256(CATALOG_PATH):
        raise ValueError("M3 preparation or frozen catalog integrity differs")
    for name, expected in prep["output_sha256"].items():
        path = Path(name)
        if ".." in path.parts or not path.is_relative_to(base) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Prepared tile changed: {name}")
    catalog = {row["tile_id"]: row for row in json.loads(CATALOG_PATH.read_text(encoding="utf-8"))}
    rows = []
    for item in plan["tiles"]:
        row = catalog[item["tile_id"]]
        if row["analysis_split"] not in ("train", "validation") or \
           row["analysis_split"] != item["analysis_split"] or row["event_id"] != item["event_id"]:
            raise ValueError("Final-test or mismatched tile in M3 development plan")
        rows.append(row)
    return plan, prep, rows, base


def run(plan_id: str, epochs_override: int | None = None) -> dict:
    plan, prep, rows, base = _load_plan(plan_id)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if epochs_override is not None:
        if epochs_override <= 0:
            raise ValueError("Epoch override must be positive")
        config["unet"]["epochs"] = epochs_override
    code_hashes = {path.as_posix(): sha256(path) for path in CODE_FILES}
    fingerprint = {"plan_sha256": sha256(Path("runs/m3_staging") / plan_id / "plan.json"),
                   "preparation_sha256": sha256(base / "preparation_manifest.json"),
                   "config": config, "code_sha256": code_hashes}
    run_id = "M3-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    root = Path("runs") / run_id
    report_root = Path("reports/m3") / run_id
    status_path = root / "status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete":
            raise FileExistsError(f"Completed run already exists: {run_id}; verify instead")
    _write_json(status_path, {"run_id": run_id, "status": "running", "plan_id": plan_id,
                              "started_at_utc": datetime.now(timezone.utc).isoformat()})
    start = time.perf_counter()
    try:
        row_by_id = {row["tile_id"]: row for row in rows}
        tiles = [load_tile(base / "tiles" / row["tile_id"], row) for row in rows]
        training = [tile for tile in tiles if tile.split == "train"]
        validation = [tile for tile in tiles if tile.split == "validation"]
        if not training or not validation or \
           len({tile.event_id for tile in training}) != 6 or len({tile.event_id for tile in validation}) != 2:
            raise ValueError("M3 development run needs all frozen train/validation event groups")
        threshold_curves = {}
        thresholds = {}
        for method in ("radar_vh", "ndwi", "mndwi"):
            grid = config["radar_vh_thresholds"] if method == "radar_vh" else config["optical_index_thresholds"]
            thresholds[method], threshold_curves[method] = select_threshold(training, method, grid)
        x, y, sample_counts = stratified_training_pixels(
            training, config["logistic"]["max_per_class_per_event"], config["seed"])
        logistic_start = time.perf_counter()
        weights, logistic_curve = fit_logistic(x, y, steps=config["logistic"]["steps"],
                                               learning_rate=config["logistic"]["learning_rate"],
                                               l2=config["logistic"]["l2"])
        logistic_seconds = time.perf_counter() - logistic_start
        thresholds.update({"logistic": 0.5, "radar_unet": 0.5, "optical_unet": 0.5})
        _write_json(root / "model/logistic.json", {"feature_order": ["VV", "VH", "B3", "B4", "B8", "B11", "B12", "NDWI", "MNDWI"],
                                                      "coefficients_and_intercept": weights.tolist(),
                                                      "training_sample_counts": sample_counts})
        _write_json(root / "curves.json", {"thresholds": threshold_curves,
                                            "logistic_train_loss": logistic_curve})
        unet_models = {}
        profiles = {}
        for modality in ("radar", "optical"):
            params = config["unet"]
            model, profile = train_unet(training, modality, root / "model" / f"{modality}_unet.pt",
                                        seed=config["seed"] + (0 if modality == "radar" else 1),
                                        epochs=params["epochs"], patches_per_tile=params["patches_per_tile"],
                                        patch_size=params["patch_size"],
                                        learning_rate=params["learning_rate"],
                                        base_channels=params["base_channels"])
            unet_models[modality] = model
            profiles[modality] = profile
        _write_json(root / "training_profile.json", {"unets": profiles,
                                                       "logistic_seconds": logistic_seconds,
                                                       "logistic_pixels": len(y)})
        _render_curves(report_root / "training_curves.png", threshold_curves, profiles)
        event_counts = {method: defaultdict(list) for method in METHODS}
        tile_metrics = []
        tile_predictions = {}
        inference_seconds = {method: 0.0 for method in METHODS}
        for tile in validation:
            scores = {}
            for method in METHODS:
                method_start = time.perf_counter()
                if method == "radar_vh":
                    score = -tile.radar[1].copy()
                    score[~tile.eligible] = np.nan
                elif method in ("ndwi", "mndwi"):
                    score = optical_index(tile, method)
                elif method == "logistic":
                    score = predict_logistic(tile, weights)
                else:
                    score = predict_unet(unet_models[method.removesuffix("_unet")], tile,
                                         method.removesuffix("_unet"))
                inference_seconds[method] += time.perf_counter() - method_start
                _score_raster(root / "predictions" / method / f"{tile.tile_id}.tif", score, tile,
                              f"uncalibrated_{method}_score")
                scores[method] = score
                label_valid_pixels = int(np.count_nonzero(tile.label != -1))
                row = {"tile_id": tile.tile_id, "event_id": tile.event_id, "method": method,
                       "s1_date": row_by_id[tile.tile_id]["s1_date"],
                       "s2_date": row_by_id[tile.tile_id]["s2_date"],
                       "sensor_offset_days": row_by_id[tile.tile_id]["sensor_offset_days"],
                       **metrics(confusion(tile.label, score >= thresholds[method], tile.eligible)),
                       "label_valid_pixels": label_valid_pixels,
                       "common_support_fraction": int(np.count_nonzero(tile.eligible)) / label_valid_pixels
                       if label_valid_pixels else None}
                tile_metrics.append(row)
                event_counts[method][tile.event_id].append(row)
            _mask_raster(root / "eligible" / f"{tile.tile_id}.tif", tile)
            tile_predictions[tile.tile_id] = scores
        event_metrics = []
        summary = {}
        for method in METHODS:
            for event, values in sorted(event_counts[method].items()):
                event_label_valid = sum(row["label_valid_pixels"] for row in values)
                counts = sum_confusions(values)
                event_metrics.append({"event_id": event, "method": method,
                                      **metrics(counts), "label_valid_pixels": event_label_valid,
                                      "common_support_fraction": counts["eligible_pixels"] / event_label_valid
                                      if event_label_valid else None})
            defined = [row["iou"] for row in event_metrics if row["method"] == method and row["iou"] is not None]
            defined_f1 = [row["f1"] for row in event_metrics if row["method"] == method and row["f1"] is not None]
            coverages = [row["common_support_fraction"] for row in event_metrics
                         if row["method"] == method and row["common_support_fraction"] is not None]
            summary[method] = {"event_macro_iou": float(np.mean(defined)) if defined else None,
                               "event_macro_f1": float(np.mean(defined_f1)) if defined_f1 else None,
                               "defined_events": len(defined), "inference_seconds": inference_seconds[method],
                               "event_macro_common_support_fraction": float(np.mean(coverages)) if coverages else None}
        _write_json(root / "metrics.json", {"schema_version": "m3_metrics_v1", "run_id": run_id,
                                              "selection_scope": "development validation only; final test unopened",
                                              "thresholds": thresholds, "per_event": event_metrics,
                                              "per_tile": tile_metrics, "summary": summary,
                                              "train_tiles": len(training), "validation_tiles": len(validation)})
        for event in sorted({tile.event_id for tile in validation}):
            for target_method in ("ndwi", "optical_unet"):
                candidates = [row for row in tile_metrics if row["event_id"] == event
                              and row["method"] == target_method]
                worst = max(candidates, key=lambda row: (row["fp"] + row["fn"], row["tile_id"]))
                tile = next(tile for tile in validation if tile.tile_id == worst["tile_id"])
                _render_errors(report_root / f"errors_{event}_{target_method}_{tile.tile_id}.png",
                               tile, tile_predictions[tile.tile_id], thresholds)
        output_files = sorted(path for directory in (root, report_root) for path in directory.rglob("*")
                              if path.is_file() and path not in (status_path, root / "run_manifest.json"))
        manifest = {
            "schema_version": "m3_run_v1", "run_id": run_id, "status": "complete",
            "plan_id": plan_id, "fingerprint": fingerprint,
            "config_sha256": sha256(CONFIG_PATH), "catalog_sha256": sha256(CATALOG_PATH),
            "preparation_manifest_sha256": sha256(base / "preparation_manifest.json"),
            "source_sha256": prep["source_sha256"], "code_sha256": code_hashes,
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "rasterio": rasterio.__version__, "torch": torch.__version__,
                            "cuda_available": torch.cuda.is_available()},
            "train_event_ids": sorted({tile.event_id for tile in training}),
            "validation_event_ids": sorted({tile.event_id for tile in validation}),
            "train_tile_ids": [tile.tile_id for tile in training],
            "validation_tile_ids": [tile.tile_id for tile in validation],
            "output_sha256": {path.as_posix(): sha256(path) for path in output_files},
            "wall_seconds": time.perf_counter() - start,
            "interpretation": ["Water label means observed surface water, not event-specific inundation.",
                               "Cloud status is unknown; optical index results are not cloud-screened.",
                               "Validation events are held out from fitting, but these are development results, not final-test scores.",
                               "Pixel scores are uncalibrated and never road-closure probabilities."],
        }
        _write_json(root / "run_manifest.json", manifest)
        _write_json(status_path, {"run_id": run_id, "status": "complete", "plan_id": plan_id,
                                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "manifest_sha256": sha256(root / "run_manifest.json")})
        return {"run_id": run_id, "status": "complete", "train_tiles": len(training),
                "validation_tiles": len(validation), "event_macro_iou":
                {method: row["event_macro_iou"] for method, row in summary.items()},
                "wall_seconds": manifest["wall_seconds"], "output_files": len(output_files)}
    except Exception as error:
        _write_json(status_path, {"run_id": run_id, "status": "failed", "plan_id": plan_id,
                                  "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "error": f"{type(error).__name__}: {error}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_id")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Recorded pilot override; omit for configured full run")
    args = parser.parse_args()
    print(json.dumps(run(args.plan_id, args.epochs), indent=2))


if __name__ == "__main__":
    main()
