"""Verify M3 source/output hashes, split boundary and recorded validation scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from flood_access.mapping import confusion, load_tile, metrics, sum_confusions
from flood_access.raster_pipeline import sha256


METHODS = ("radar_vh", "ndwi", "mndwi", "logistic", "radar_unet", "optical_unet")


def verify(run_id: str) -> dict:
    if not run_id.startswith("M3-") or any(c in run_id for c in "/\\."):
        raise ValueError("Invalid M3 run ID")
    root = Path("runs") / run_id
    report_root = Path("reports/m3") / run_id
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id or manifest["status"] != "complete" or \
       status["status"] != "complete" or status["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("M3 run is incomplete or its manifest changed")
    plan_id = manifest["plan_id"]
    base = Path("data/processed/m3_development") / plan_id
    prep_path = base / "preparation_manifest.json"
    if sha256(prep_path) != manifest["preparation_manifest_sha256"]:
        raise ValueError("Prepared input manifest changed")
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    for name, expected in prep["output_sha256"].items():
        path = Path(name)
        if ".." in path.parts or not path.is_relative_to(base) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Prepared canonical tile changed: {name}")
    for name, expected in manifest["source_sha256"].items():
        path = Path(name)
        if ".." in path.parts or not path.is_relative_to("data/raw") or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Source missing or changed: {name}")
    for name, expected in manifest["output_sha256"].items():
        path = Path(name)
        if ".." in path.parts or not (path.is_relative_to(root) or path.is_relative_to(report_root)) or \
           not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Output missing or changed: {name}")
    actual = {path.as_posix() for directory in (root, report_root) for path in directory.rglob("*")
              if path.is_file() and path not in (manifest_path, root / "status.json")}
    if actual != set(manifest["output_sha256"]):
        raise ValueError("Output file set differs from manifest")
    catalog = {row["tile_id"]: row for row in json.loads(Path(
        "data/processed/m2_runs/M2-151ccd74b14c/catalog.json").read_text(encoding="utf-8"))}
    if set(manifest["train_tile_ids"]) & set(manifest["validation_tile_ids"]):
        raise ValueError("Training and validation tiles overlap")
    for split, key in (("train", "train_tile_ids"), ("validation", "validation_tile_ids")):
        if any(catalog[tile_id]["analysis_split"] != split for tile_id in manifest[key]):
            raise ValueError("Run crosses the frozen event split")
    metric_file = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    recorded = {(row["tile_id"], row["method"]): row for row in metric_file["per_tile"]}
    if len(recorded) != len(manifest["validation_tile_ids"]) * len(METHODS):
        raise ValueError("Incomplete or duplicate per-tile metrics")
    event_rows: dict[tuple[str, str], list[dict]] = {}
    for tile_id in manifest["validation_tile_ids"]:
        tile = load_tile(base / "tiles" / tile_id, catalog[tile_id])
        with rasterio.open(root / "eligible" / f"{tile_id}.tif") as source:
            if source.crs != tile.crs or source.transform != tile.transform or \
               not np.array_equal(source.read(1) == 1, tile.eligible):
                raise ValueError(f"Common eligible mask differs: {tile_id}")
        for method in METHODS:
            with rasterio.open(root / "predictions" / method / f"{tile_id}.tif") as source:
                if source.crs != tile.crs or source.transform != tile.transform:
                    raise ValueError(f"Prediction grid differs: {method}/{tile_id}")
                score = source.read(1)
            if not np.array_equal(np.isfinite(score), tile.eligible):
                raise ValueError(f"Unknown pixels in prediction differ: {method}/{tile_id}")
            counts = confusion(tile.label, score >= metric_file["thresholds"][method], tile.eligible)
            row = recorded[(tile_id, method)]
            if any(row[key] != value for key, value in counts.items()):
                raise ValueError(f"Stored confusion counts differ: {method}/{tile_id}")
            label_valid_pixels = int(np.count_nonzero(tile.label != -1))
            if "label_valid_pixels" in row:
                expected_coverage = counts["eligible_pixels"] / label_valid_pixels if label_valid_pixels else None
                if row["label_valid_pixels"] != label_valid_pixels or \
                   (expected_coverage is None) != (row["common_support_fraction"] is None) or \
                   (expected_coverage is not None and not np.isclose(row["common_support_fraction"],
                                                                     expected_coverage, atol=1e-12)):
                    raise ValueError(f"Stored common-support coverage differs: {method}/{tile_id}")
            recomputed = metrics(counts)
            for name in ("iou", "f1", "precision", "recall"):
                if recomputed[name] is None:
                    if row[name] is not None:
                        raise ValueError(f"Stored undefined metric differs: {method}/{tile_id}")
                elif row[name] is None or not np.isclose(row[name], recomputed[name], atol=1e-12):
                    raise ValueError(f"Stored tile metric differs: {method}/{tile_id}/{name}")
            event_rows.setdefault((tile.event_id, method), []).append(counts)
    event_metrics = {(row["event_id"], row["method"]): row for row in metric_file["per_event"]}
    if set(event_rows) != set(event_metrics):
        raise ValueError("Per-event metric set differs from prediction set")
    for key, rows in event_rows.items():
        counts = sum_confusions(rows)
        if any(event_metrics[key][name] != value for name, value in counts.items()):
            raise ValueError(f"Per-event metrics differ: {key}")
        if "label_valid_pixels" in event_metrics[key]:
            label_valid_pixels = sum(recorded[(tile_id, key[1])]["label_valid_pixels"]
                                     for tile_id in manifest["validation_tile_ids"]
                                     if catalog[tile_id]["event_id"] == key[0])
            expected_coverage = counts["eligible_pixels"] / label_valid_pixels if label_valid_pixels else None
            if event_metrics[key]["label_valid_pixels"] != label_valid_pixels or \
               (expected_coverage is None) != (event_metrics[key]["common_support_fraction"] is None) or \
               (expected_coverage is not None and not np.isclose(event_metrics[key]["common_support_fraction"],
                                                                  expected_coverage, atol=1e-12)):
                raise ValueError(f"Per-event common-support coverage differs: {key}")
        recalculated = metrics(counts)
        for name in ("iou", "f1", "precision", "recall"):
            old, new = event_metrics[key][name], recalculated[name]
            if (old is None) != (new is None) or (old is not None and not np.isclose(old, new, atol=1e-12)):
                raise ValueError(f"Per-event metric differs: {key}/{name}")
    for method in METHODS:
        values = [event_metrics[key]["iou"] for key in event_metrics if key[1] == method
                  and event_metrics[key]["iou"] is not None]
        reported = metric_file["summary"][method]["event_macro_iou"]
        if not values or reported is None or not np.isclose(reported, np.mean(values), atol=1e-12):
            raise ValueError(f"Macro IoU differs: {method}")
        if "event_macro_common_support_fraction" in metric_file["summary"][method]:
            coverages = [event_metrics[key]["common_support_fraction"] for key in event_metrics
                         if key[1] == method and event_metrics[key]["common_support_fraction"] is not None]
            reported_coverage = metric_file["summary"][method]["event_macro_common_support_fraction"]
            if not coverages or not np.isclose(reported_coverage, np.mean(coverages), atol=1e-12):
                raise ValueError(f"Macro common-support coverage differs: {method}")
    return {"run_id": run_id, "status": "verified", "source_objects": len(manifest["source_sha256"]),
            "output_files": len(manifest["output_sha256"]),
            "validation_tiles": len(manifest["validation_tile_ids"]),
            "prediction_rasters": len(recorded), "manifest_sha256": sha256(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id), indent=2))


if __name__ == "__main__":
    main()
