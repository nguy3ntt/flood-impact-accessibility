"""Validate staged development sources and materialise M2-format canonical tiles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
import rasterio

from flood_access.mapping import load_tile
from flood_access.raster_pipeline import canonicalize_tile, sha256


CATALOG = Path("data/processed/m2_runs/M2-151ccd74b14c/catalog.json")
RAW = Path("data/raw/sen1floods11/v1.1")
INVENTORY = Path(".project/m1_source_inventory.json")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(plan_id: str) -> dict:
    if re.fullmatch(r"M3S-[0-9a-f]{12}", plan_id) is None:
        raise ValueError("Invalid M3 staging plan ID")
    plan_path = Path("runs/m3_staging") / plan_id / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["plan_id"] != plan_id or plan["catalog_sha256"] != sha256(CATALOG):
        raise ValueError("Staging plan differs from frozen M2 catalog")
    catalog = {row["tile_id"]: row for row in json.loads(CATALOG.read_text(encoding="utf-8"))}
    inventory = {row["local_path"]: row for row in json.loads(INVENTORY.read_text(encoding="utf-8"))}
    if len(inventory) != len(json.loads(INVENTORY.read_text(encoding="utf-8"))):
        raise ValueError("Source inventory has duplicate paths")
    selected = plan["tiles"]
    if not selected or len(selected) != len({row["tile_id"] for row in selected}):
        raise ValueError("Staging selection is empty or duplicated")
    base = Path("data/processed/m3_development") / plan_id
    status_path = Path("runs/m3_staging") / plan_id / "preparation_status.json"
    source_hashes = {}
    for planned in selected:
        row = catalog[planned["tile_id"]]
        if row["analysis_split"] not in ("train", "validation") or \
           planned["analysis_split"] != row["analysis_split"] or planned["event_id"] != row["event_id"]:
            raise ValueError("Staging plan crosses the frozen development boundary")
        for layer in ("LabelHand", "S1Hand", "S2Hand"):
            path = RAW / "data/flood_events/HandLabeled" / layer / f"{row['tile_id']}_{layer}.tif"
            logical = path.as_posix()
            record = inventory.get(logical)
            if record is None or not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError(f"Missing or changed inventory source: {logical}")
            source_hashes[logical] = record["sha256"]
    _write_json(status_path, {"plan_id": plan_id, "status": "running",
                              "started_at_utc": datetime.now(timezone.utc).isoformat()})
    done = 0
    try:
        for planned in selected:
            row = catalog[planned["tile_id"]]
            tile_dir = base / "tiles" / row["tile_id"]
            if not (tile_dir / "metadata.json").exists():
                canonicalize_tile(RAW, row, tile_dir)
            tile = load_tile(tile_dir, row)
            metadata = json.loads((tile_dir / "metadata.json").read_text(encoding="utf-8"))
            for source in metadata["sources"].values():
                if source_hashes.get(source["path"]) != source["sha256"]:
                    raise ValueError(f"Cached canonical tile uses changed source: {row['tile_id']}")
            if tile.eligible.shape != (512, 512):
                raise ValueError("Unexpected benchmark chip shape")
            done += 1
            if done % 10 == 0:
                print(f"Validated {done}/{len(selected)} canonical tiles", flush=True)
        files = sorted(path for path in (base / "tiles").rglob("*") if path.is_file())
        manifest = {"schema_version": "m3_preparation_v1", "plan_id": plan_id,
                    "plan_sha256": sha256(plan_path), "catalog_sha256": sha256(CATALOG),
                    "source_sha256": source_hashes,
                    "output_sha256": {path.as_posix(): sha256(path) for path in files},
                    "tile_count": done, "environment": {"python": sys.version.split()[0],
                    "numpy": np.__version__, "rasterio": rasterio.__version__}}
        _write_json(base / "preparation_manifest.json", manifest)
        _write_json(status_path, {"plan_id": plan_id, "status": "complete",
                                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "tile_count": done, "manifest_sha256": sha256(base / "preparation_manifest.json")})
    except Exception as error:
        _write_json(status_path, {"plan_id": plan_id, "status": "failed",
                                  "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "completed_tiles": done, "error": f"{type(error).__name__}: {error}"})
        raise
    return {"plan_id": plan_id, "status": "complete", "tiles": done,
            "canonical_files": len(files), "eligible_pixels": "recorded per tile"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_id")
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan_id), indent=2))


if __name__ == "__main__":
    main()
