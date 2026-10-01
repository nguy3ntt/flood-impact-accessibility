"""Build and verify the deterministic M2 pilot from immutable local source files.

Run from the repository root after installing requirements/m2-pipeline.lock.
All derived outputs stay under ignored data/processed, reports and runs roots.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import PIL
import rasterio
import shapely

from flood_access.catalog import assign_event_split, build_catalog
from flood_access.case_vectors import build_spain_vectors, validate_features
from flood_access.qa import render_tile_panel
from flood_access.raster_pipeline import S1_BANDS, S2_BANDS, canonicalize_tile, sha256


ROOT = Path(".")
RAW = ROOT / "data/raw/sen1floods11/v1.1"
OUTPUT_BASE = ROOT / "data/processed/m2_runs"
REPORTS_BASE = ROOT / "reports/m2"
SPLIT_CONFIG = ROOT / "configs/event_split_v1.json"
SOURCE_MANIFEST = ROOT / ".project/m1_source_inventory.json"
PILOT_CHIPS = ("Bolivia_103757", "Ghana_103272", "Spain_7370579")
CODE_FILES = (
    Path("scripts/build_m2_pilot.py"), Path("src/flood_access/catalog.py"),
    Path("src/flood_access/raster_pipeline.py"), Path("src/flood_access/case_vectors.py"),
    Path("src/flood_access/qa.py"),
)


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _verify_sources() -> list[dict]:
    records = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    seen = set()
    for record in records:
        path = Path(record["local_path"])
        if path in seen or not path.is_relative_to("data/raw"):
            raise ValueError(f"Duplicate or non-raw manifest path: {path}")
        seen.add(path)
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing or changed source object: {path}")
    return records


def _used_sources(records: list[dict]) -> list[dict]:
    needed = {
        RAW / "Sen1Floods11_Metadata.geojson", RAW / "catalog.zip",
        Path("data/raw/cems/EMSR388/EMSR388_AOI07_GRA_PRODUCT_r1_RTP01_v1_vector.zip"),
        Path("data/raw/gva/ca_hospitales_20260805.geojson"),
        Path("data/raw/ine/secciones_2019_spain_pilot.geojson"),
    }
    needed.update(RAW / "splits/flood_handlabeled" / filename for filename in
                  ("flood_train_data.csv", "flood_valid_data.csv", "flood_test_data.csv", "flood_bolivia_data.csv"))
    needed.update(RAW / "data/flood_events/HandLabeled" / layer / f"{chip}_{layer}.tif"
                  for chip in PILOT_CHIPS for layer in ("LabelHand", "S1Hand", "S2Hand"))
    selected = [record for record in records if Path(record["local_path"]) in needed]
    if {Path(record["local_path"]) for record in selected} != needed:
        raise ValueError("Required source is absent from the verified inventory")
    return selected


def _validate_tile(tile_dir: Path) -> None:
    with rasterio.open(tile_dir / "label.tif") as label, \
         rasterio.open(tile_dir / "radar_db.tif") as radar, \
         rasterio.open(tile_dir / "optical_toa.tif") as optical, \
         rasterio.open(tile_dir / "validity.tif") as validity, \
         rasterio.open(tile_dir / "cloud_status.tif") as cloud:
        datasets = (label, radar, optical, validity, cloud)
        if any(ds.crs != label.crs or ds.transform != label.transform or ds.shape != label.shape for ds in datasets):
            raise ValueError(f"Canonical grids differ in {tile_dir}")
        if (label.count, radar.count, optical.count, validity.count, cloud.count) != (1, 2, 13, 5, 1):
            raise ValueError(f"Canonical band count differs in {tile_dir}")
        if radar.descriptions != S1_BANDS or optical.descriptions != S2_BANDS:
            raise ValueError(f"Canonical band order differs in {tile_dir}")
        labels = label.read(1)
        masks = validity.read()
        if not np.isin(labels, (-1, 0, 1)).all() or not np.array_equal(masks[0] == 1, labels != -1):
            raise ValueError(f"Canonical label/validity mismatch in {tile_dir}")
        if not np.array_equal(masks[3] == 1, (masks[1] == 1) & (masks[2] == 1)):
            raise ValueError(f"Canonical sensor-validity mismatch in {tile_dir}")
        if not np.array_equal(masks[4] == 1, (masks[0] == 1) & (masks[3] == 1)):
            raise ValueError(f"Canonical supervised-validity mismatch in {tile_dir}")
        if not np.all(cloud.read(1) == 255):
            raise ValueError(f"Cloud status is incorrectly inferred in {tile_dir}")


def _validate_vectors(vector_dir: Path) -> None:
    for filename, geometry_type in (("roads.geojson", "line"), ("flood_traces.geojson", "polygon"),
                                    ("hospitals.geojson", "point"), ("origins.geojson", "point")):
        collection = json.loads((vector_dir / filename).read_text(encoding="utf-8"))
        if collection.get("type") != "FeatureCollection":
            raise ValueError(f"Invalid output vector table: {filename}")
        validate_features(collection["features"], geometry_type)


def main() -> None:
    if not (ROOT / "pyproject.toml").is_file():
        raise SystemExit("Run from the repository root")
    records = _used_sources(_verify_sources())
    config = json.loads(SPLIT_CONFIG.read_text(encoding="utf-8"))
    code_hashes = {p.as_posix(): sha256(p) for p in CODE_FILES}
    revision = hashlib.sha256(json.dumps({"code": code_hashes, "sources": {r["source_id"]: r["sha256"] for r in records},
                                         "split": sha256(SPLIT_CONFIG)}, sort_keys=True).encode()).hexdigest()[:12]
    run_id = f"M2-{revision}"
    output_root = OUTPUT_BASE / run_id
    report_root = REPORTS_BASE / run_id
    status_path = ROOT / "runs" / run_id / "status.json"
    _write_json(status_path, {"run_id": run_id, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                              "output_root": output_root.as_posix()})
    try:
        catalog = build_catalog(RAW)
        catalog, overlap = assign_event_split(catalog, config)
        _write_json(output_root / "catalog.json", catalog)
        _write_json(output_root / "split_manifest.json", {
            "schema_version": "m2_split_v1", "configuration": config,
            "configuration_sha256": sha256(SPLIT_CONFIG),
            "chips": [{"tile_id": row["tile_id"], "event_id": row["event_id"],
                       "analysis_split": row["analysis_split"], "official_split": row["official_split"]} for row in catalog],
        })
        _write_json(output_root / "overlap_audit.json", overlap)
        indexed = {row["tile_id"]: row for row in catalog}
        tile_summaries = {}
        for chip in PILOT_CHIPS:
            tile_dir = output_root / "tiles" / chip
            tile_summaries[chip] = canonicalize_tile(RAW, indexed[chip], tile_dir)
            _validate_tile(tile_dir)
        vector_dir = output_root / "vectors/spain_7370579"
        vector_summary = build_spain_vectors(
            Path("data/raw/cems/EMSR388/EMSR388_AOI07_GRA_PRODUCT_r1_RTP01_v1_vector.zip"),
            Path("data/raw/gva/ca_hospitales_20260805.geojson"),
            Path("data/raw/ine/secciones_2019_spain_pilot.geojson"),
            RAW / "data/flood_events/HandLabeled/LabelHand/Spain_7370579_LabelHand.tif", vector_dir,
        )
        _write_json(vector_dir / "audit.json", vector_summary)
        _validate_vectors(vector_dir)
        for chip in PILOT_CHIPS:
            render_tile_panel(output_root / "tiles" / chip, report_root / f"qa_{chip}.png",
                              vector_dir if chip == "Spain_7370579" else None)
        outputs = sorted(path for directory in (output_root, report_root) for path in directory.rglob("*")
                         if path.is_file() and path != output_root / "run_manifest.json")
        manifest = {
            "schema_version": "m2_run_v1", "run_id": run_id, "status": "complete",
            "output_root": output_root.as_posix(), "qa_root": report_root.as_posix(),
            "source_objects": {r["source_id"]: {
                "source_url": r["source_url"], "local_path": r["local_path"],
                "retrieved_at_utc": r["retrieved_at_utc"], "bytes": r["bytes"],
                "sha256": r["sha256"], "generation": r.get("generation"),
                "licence_status": r["licence_status"], "redistribution_status": r["redistribution_status"],
            } for r in records},
            "code_sha256": code_hashes, "split_config_sha256": sha256(SPLIT_CONFIG),
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "rasterio": rasterio.__version__, "shapely": shapely.__version__,
                            "pillow": PIL.__version__},
            "pilot_tiles": list(PILOT_CHIPS),
            "split_counts": overlap["split_chip_counts"],
            "overlap_counts": {"pairs": len(overlap["overlap_pairs"]),
                               "cross_event": overlap["cross_event_overlap_count"],
                               "cross_split": overlap["cross_split_overlap_count"]},
            "tile_quality_counts": {chip: value["quality_counts"] for chip, value in tile_summaries.items()},
            "vector_summary": vector_summary,
            "output_sha256": {path.as_posix(): sha256(path) for path in outputs},
            "limitations": ["No optical cloud QA; all cloud states unknown.",
                            "Spain road lines are exposure context, not a navigable graph or closure labels.",
                            "Spain case imagery and 2026 hospitals have a temporal mismatch."],
        }
        _write_json(output_root / "run_manifest.json", manifest)
        _write_json(status_path, {"run_id": run_id, "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "manifest_sha256": sha256(output_root / "run_manifest.json"), "output_root": output_root.as_posix()})
        print(json.dumps({"run_id": run_id, "status": "complete", "output_root": output_root.as_posix(),
                          "qa_root": report_root.as_posix(), "chips": len(PILOT_CHIPS),
                          "catalog_chips": len(catalog), "overlap_pairs": len(overlap["overlap_pairs"]),
                          "vectors": vector_summary["selected_counts"], "output_files": len(outputs)}, indent=2))
    except Exception as error:
        _write_json(status_path, {"run_id": run_id, "status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "error": f"{type(error).__name__}: {error}", "output_root": output_root.as_posix()})
        raise


if __name__ == "__main__":
    main()
