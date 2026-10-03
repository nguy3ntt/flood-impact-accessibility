"""Validate frozen final source trios and build canonical evaluation tiles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from flood_access.mapping import load_tile
from flood_access.raster_pipeline import canonicalize_tile, sha256
from stage_m8_final import CATALOG, CONFIG, MANIFEST, RAW_ROOT, STAGE_ROOT, frozen_rows, write_json


def prepare(plan_id: str) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan_path = STAGE_ROOT / plan_id / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    stage_status = json.loads((STAGE_ROOT / plan_id / "status.json").read_text(encoding="utf-8"))
    if (plan["plan_id"] != plan_id or plan["config_sha256"] != sha256(CONFIG)
            or stage_status["status"] != "complete" or sha256(CATALOG) != config["catalog_sha256"]):
        raise ValueError("Final source stage or frozen config differs")
    rows = frozen_rows(json.loads(CATALOG.read_text(encoding="utf-8")), config)
    if [row["tile_id"] for row in rows] != [row["tile_id"] for row in plan["tiles"]]:
        raise ValueError("Final stage tile order differs")
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inventory = {row["local_path"]: row for row in records}
    if len(records) != len(inventory):
        raise ValueError("Duplicate source inventory paths")
    source_hashes = {}
    for row in rows:
        for layer in ("LabelHand", "S1Hand", "S2Hand"):
            path = RAW_ROOT / "v1.1/data/flood_events/HandLabeled" / layer / f"{row['tile_id']}_{layer}.tif"
            record = inventory.get(path.as_posix())
            if (record is None or not path.is_file() or path.stat().st_size != record["bytes"]
                    or sha256(path) != record["sha256"]):
                raise ValueError(f"Missing or changed inventoried source: {path}")
            source_hashes[path.as_posix()] = record["sha256"]
    base = Path("data/processed/m8_final") / plan_id
    status_path = STAGE_ROOT / plan_id / "preparation_status.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8"))["status"] == "complete":
        raise FileExistsError("Final tiles already prepared")
    write_json(status_path, {"plan_id": plan_id, "status": "running",
                             "started_at_utc": datetime.now(timezone.utc).isoformat()})
    done = 0
    try:
        for row in rows:
            folder = base / "tiles" / row["tile_id"]
            if not (folder / "metadata.json").exists():
                canonicalize_tile(RAW_ROOT / "v1.1", row, folder)
            tile = load_tile(folder, row)
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if tile.eligible.shape != (512, 512) or any(
                    source_hashes.get(source["path"]) != source["sha256"]
                    for source in metadata["sources"].values()):
                raise ValueError(f"Canonical source or shape differs: {row['tile_id']}")
            done += 1
            if done % 10 == 0:
                print(f"Validated {done}/{len(rows)} final tiles", flush=True)
        outputs = {p.as_posix(): sha256(p) for p in sorted((base / "tiles").rglob("*")) if p.is_file()}
        manifest = {"schema_version": "m8_final_preparation_v1", "plan_id": plan_id,
                    "plan_sha256": sha256(plan_path), "config_sha256": sha256(CONFIG),
                    "source_sha256": source_hashes, "output_sha256": outputs,
                    "tile_count": done}
        manifest_path = base / "preparation_manifest.json"
        write_json(manifest_path, manifest)
        write_json(status_path, {"plan_id": plan_id, "status": "complete",
                                 "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "manifest_sha256": sha256(manifest_path), "tiles": done})
    except Exception as error:
        write_json(status_path, {"plan_id": plan_id, "status": "failed",
                                 "completed_tiles": done, "error": f"{type(error).__name__}: {error}"})
        raise
    return {"plan_id": plan_id, "status": "complete", "tiles": done, "canonical_files": len(outputs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_id")
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan_id), indent=2))
