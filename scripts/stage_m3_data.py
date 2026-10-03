"""Stage explicitly selected M3 development chips; never acquire final-test labels.

Run from the repository root. The plan and acquisition status are ignored local
records. Source objects retain their exact bytes and existing inventory format.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from acquire_sen1floods11 import MANIFEST, RAW_ROOT, acquire, object_info


CATALOG = Path("data/processed/m2_runs/M2-151ccd74b14c/catalog.json")
PLAN_ROOT = Path("runs/m3_staging")
LAYERS = ("LabelHand", "S1Hand", "S2Hand")
SELECTION_SALT = "m3_event_development_v1"


def selected_chips(catalog: list[dict], train_per_event: int, validation_per_event: int) -> list[dict]:
    """Stable nested sample by ID only; validation 0 means every tile."""
    if train_per_event < 0 or validation_per_event < 0:
        raise ValueError("Per-event limits cannot be negative")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in catalog:
        if row["analysis_split"] in ("train", "validation"):
            grouped.setdefault((row["analysis_split"], row["event_location"]), []).append(row)
    if not grouped or not any(key[0] == "validation" for key in grouped):
        raise ValueError("Catalog lacks the frozen development event groups")
    selected = []
    for (split, _), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: hashlib.sha256(
            f"{SELECTION_SALT}:{row['tile_id']}".encode()).hexdigest())
        limit = train_per_event if split == "train" else validation_per_event
        selected.extend(rows[:limit or None])
    if len({row["tile_id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate tile in staged selection")
    return sorted(selected, key=lambda row: row["tile_id"])


def object_keys(rows: list[dict]) -> list[str]:
    keys = [f"v1.1/data/flood_events/HandLabeled/{layer}/{row['tile_id']}_{layer}.tif"
            for row in rows for layer in LAYERS]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source object in plan")
    return sorted(keys)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(train_per_event: int, validation_per_event: int, max_new_bytes: int,
          plan_only: bool) -> dict:
    if max_new_bytes <= 0:
        raise ValueError("Download byte cap must be positive")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    rows = selected_chips(catalog, train_per_event, validation_per_event)
    keys = object_keys(rows)
    plan_id = "M3S-" + hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:12]
    plan_path = PLAN_ROOT / plan_id / "plan.json"
    status_path = PLAN_ROOT / plan_id / "status.json"
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
            if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
                raise ValueError(f"Existing source differs from inventory: {logical}")
        elif path.exists():
            raise ValueError(f"Uninventoried source exists; inspect before resuming: {logical}")
        else:
            info = object_info(key)
            size = int(info["size"])
            if size <= 0:
                raise ValueError(f"Invalid remote source size: {key}")
            remote[key] = {"bytes": size, "generation": info["generation"]}
            new_bytes += size
    plan = {
        "schema_version": "m3_stage_v1", "plan_id": plan_id,
        "catalog_sha256": _sha256(CATALOG), "selection_salt": SELECTION_SALT,
        "train_per_event": train_per_event, "validation_per_event": validation_per_event,
        "tiles": [{"tile_id": r["tile_id"], "event_id": r["event_id"],
                   "analysis_split": r["analysis_split"]} for r in rows],
        "keys": keys, "new_objects": remote, "new_bytes_estimate": new_bytes,
        "max_new_bytes": max_new_bytes,
        "rights": "Local research only; source redistribution unresolved.",
    }
    _write_json(plan_path, plan)
    if new_bytes > max_new_bytes:
        raise ValueError(f"Plan needs {new_bytes} new bytes, above cap {max_new_bytes}; see {plan_path}")
    if plan_only:
        return {"plan_id": plan_id, "status": "planned", "tiles": len(rows),
                "new_objects": len(remote), "new_bytes": new_bytes}
    _write_json(status_path, {"plan_id": plan_id, "status": "running",
                              "started_at_utc": datetime.now(timezone.utc).isoformat()})
    downloaded = 0
    try:
        remaining = max_new_bytes
        for key in keys:
            if key not in remote:
                continue
            record = acquire(key, remaining)
            if record["bytes"] != remote[key]["bytes"] or record["generation"] != remote[key]["generation"]:
                raise ValueError(f"Remote generation changed during acquisition: {key}")
            inventory.append(record)
            _write_json(MANIFEST, inventory)
            downloaded += 1
            remaining -= record["bytes"]
            if downloaded % 10 == 0:
                print(f"Acquired {downloaded}/{len(remote)} new objects", flush=True)
        _write_json(status_path, {"plan_id": plan_id, "status": "complete",
                                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "downloaded_objects": downloaded, "downloaded_bytes": max_new_bytes - remaining})
    except Exception as error:
        _write_json(status_path, {"plan_id": plan_id, "status": "failed",
                                  "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "downloaded_objects": downloaded,
                                  "error": f"{type(error).__name__}: {error}"})
        raise
    return {"plan_id": plan_id, "status": "complete", "tiles": len(rows),
            "downloaded_objects": downloaded, "downloaded_bytes": max_new_bytes - remaining}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-event", type=int, default=2,
                        help="0 selects all six training events")
    parser.add_argument("--validation-per-event", type=int, default=2,
                        help="0 selects all two validation events")
    parser.add_argument("--max-new-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(stage(args.train_per_event, args.validation_per_event,
                           args.max_new_bytes, args.plan_only), indent=2))


if __name__ == "__main__":
    main()
