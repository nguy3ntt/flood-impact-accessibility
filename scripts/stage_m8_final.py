"""Preflight and acquire only the frozen Nigeria/Somalia final-test trios.

Planning reads catalog metadata and remote object sizes, never label pixels.
All source files and inventory records stay under ignored local paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from acquire_sen1floods11 import MANIFEST, RAW_ROOT, acquire, object_info


CONFIG = Path("configs/m8_final_v1.json")
CATALOG = Path("data/processed/m2_runs/M2-151ccd74b14c/catalog.json")
SPLIT = Path("configs/event_split_v1.json")
STAGE_ROOT = Path("runs/m8_final_stage")
LAYERS = ("LabelHand", "S1Hand", "S2Hand")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    part.replace(path)


def freeze_plan(path: Path, proposed: dict, remaining_remote: dict) -> dict:
    """Never rewrite a preflight plan after acquisition has begun."""
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        for field in ("schema_version", "plan_id", "config_sha256", "catalog_sha256",
                      "tiles", "keys", "max_new_bytes"):
            if frozen.get(field) != proposed[field]:
                raise ValueError(f"Existing frozen M8 staging plan changed: {field}")
        for key, info in remaining_remote.items():
            if frozen["new_objects"].get(key) != info:
                raise ValueError(f"Remote object changed since frozen preflight: {key}")
        return frozen
    write_json(path, proposed)
    return proposed


def frozen_rows(catalog: list[dict], config: dict) -> list[dict]:
    """Reject any selection that differs from the frozen final-event groups."""
    rows = sorted((row for row in catalog if row["analysis_split"] == "final_test"),
                  key=lambda row: row["tile_id"])
    counts = {event: sum(row["event_id"] == event for row in rows)
              for event in config["final_event_chip_counts"]}
    if (not rows or counts != config["final_event_chip_counts"] or
        len(rows) != sum(counts.values()) or
        len({row["tile_id"] for row in rows}) != len(rows)):
        raise ValueError("Catalog final-test selection differs from frozen M8 config")
    return rows


def source_keys(rows: list[dict]) -> list[str]:
    return sorted(
        f"v1.1/data/flood_events/HandLabeled/{layer}/{row['tile_id']}_{layer}.tif"
        for row in rows for layer in LAYERS
    )


def stage(*, plan_only: bool) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (config["schema_version"] != "m8_final_v1" or
        sha256(CATALOG) != config["catalog_sha256"] or
        sha256(SPLIT) != config["event_split_sha256"]):
        raise ValueError("Frozen M8 config, source catalog or event split changed")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    rows = frozen_rows(catalog, config)
    keys = source_keys(rows)
    plan_id = "M8S-" + hashlib.sha256(
        (sha256(CONFIG) + "\n" + "\n".join(keys)).encode()
    ).hexdigest()[:12]
    root = STAGE_ROOT / plan_id
    inventory = json.loads(MANIFEST.read_text(encoding="utf-8"))
    indexed = {record["local_path"]: record for record in inventory}
    if len(indexed) != len(inventory):
        raise ValueError("Source inventory has duplicate local paths")
    remote = {}
    new_bytes = 0
    for key in keys:
        path = RAW_ROOT.joinpath(*key.split("/"))
        logical = path.as_posix()
        if logical in indexed:
            record = indexed[logical]
            if (not path.is_file() or path.stat().st_size != record["bytes"] or
                sha256(path) != record["sha256"]):
                raise ValueError(f"Existing source differs from inventory: {logical}")
        elif path.exists():
            raise ValueError(f"Uninventoried source exists: {logical}")
        else:
            info = object_info(key)
            size = int(info["size"])
            if size <= 0 or not info.get("generation"):
                raise ValueError(f"Invalid remote size/generation: {key}")
            remote[key] = {"bytes": size, "generation": info["generation"]}
            new_bytes += size
    proposed_plan = {
        "schema_version": "m8_final_stage_v1",
        "plan_id": plan_id,
        "config_sha256": sha256(CONFIG),
        "catalog_sha256": sha256(CATALOG),
        "tiles": [{"tile_id": row["tile_id"], "event_id": row["event_id"],
                   "analysis_split": row["analysis_split"]} for row in rows],
        "keys": keys,
        "new_objects": remote,
        "new_bytes_estimate": new_bytes,
        "max_new_bytes": config["max_new_source_bytes"],
        "rights": "Local research only; source redistribution unresolved."
    }
    plan_path = root / "plan.json"
    plan = freeze_plan(plan_path, proposed_plan, remote)
    if new_bytes > config["max_new_source_bytes"]:
        raise ValueError(f"Preflight needs {new_bytes} bytes above cap; inspect {root / 'plan.json'}")
    if plan_only:
        return {"plan_id": plan_id, "status": "planned", "tiles": len(rows),
                "new_objects": len(remote), "new_bytes": new_bytes}
    status_path = root / "status.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8"))["status"] == "complete":
        raise FileExistsError(f"Completed final source stage exists: {plan_id}")
    write_json(status_path, {"plan_id": plan_id, "status": "running",
                             "started_at_utc": datetime.now(timezone.utc).isoformat()})
    downloaded = 0
    remaining = config["max_new_source_bytes"]
    try:
        for key in keys:
            if key not in remote:
                continue
            record = acquire(key, remaining)
            if (record["bytes"] != remote[key]["bytes"] or
                record["generation"] != remote[key]["generation"]):
                raise ValueError(f"Remote generation changed during acquisition: {key}")
            inventory.append(record)
            write_json(MANIFEST, inventory)
            downloaded += 1
            remaining -= record["bytes"]
            if downloaded % 10 == 0:
                print(f"Acquired {downloaded}/{len(remote)} final-test source objects", flush=True)
        write_json(status_path, {"plan_id": plan_id, "status": "complete",
                                 "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "downloaded_objects": downloaded,
                                 "downloaded_bytes": config["max_new_source_bytes"] - remaining})
    except Exception as error:
        write_json(status_path, {"plan_id": plan_id, "status": "failed",
                                 "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "downloaded_objects": downloaded,
                                 "error": f"{type(error).__name__}: {error}"})
        raise
    return {"plan_id": plan_id, "status": "complete", "tiles": len(rows),
            "downloaded_objects": downloaded,
            "downloaded_bytes": config["max_new_source_bytes"] - remaining}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true",
                        help="Preflight remote sizes without acquiring final-test objects")
    args = parser.parse_args()
    print(json.dumps(stage(plan_only=args.plan_only), indent=2))
