"""Verify frozen M8 outputs and independently rebuild event aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flood_access.raster_pipeline import sha256
from run_m8_final import _aggregate
from stage_m8_final import CONFIG, STAGE_ROOT


def verify(run_id: str) -> dict:
    root = Path("runs") / run_id
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if (manifest["schema_version"] != "m8_final_run_v1" or manifest["run_id"] != run_id
            or status["status"] != "complete" or status["manifest_sha256"] != sha256(manifest_path)
            or manifest["config_sha256"] != sha256(CONFIG)):
        raise ValueError("M8 final manifest, status or config differs")
    for path, digest in manifest["code_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Evaluation code changed: {path}")
    for name, digest in manifest["output_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Final output changed: {name}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan_id = manifest["plan_id"]
    preparation = Path("data/processed/m8_final") / plan_id / "preparation_manifest.json"
    preparation_status = json.loads((STAGE_ROOT / plan_id / "preparation_status.json").read_text(encoding="utf-8"))
    if (preparation_status["status"] != "complete"
            or preparation_status["manifest_sha256"] != sha256(preparation)
            or manifest["preparation_manifest_sha256"] != sha256(preparation)):
        raise ValueError("Final source preparation changed")
    prepared = json.loads(preparation.read_text(encoding="utf-8"))
    for name, digest in {**prepared["source_sha256"], **prepared["output_sha256"]}.items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Final source or canonical output changed: {name}")
    reference_manifests = {}
    for milestone in ("m3", "m4", "m5"):
        refs = config["references"]
        path = Path("runs") / refs[f"{milestone}_run_id"] / "run_manifest.json"
        if sha256(path) != refs[f"{milestone}_manifest_sha256"]:
            raise ValueError(f"Frozen {milestone} manifest changed")
        reference_manifests[milestone] = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_paths = {}
    m3_root = Path("runs") / refs["m3_run_id"] / "model"
    for modality in ("radar", "optical"):
        checkpoint_paths[f"m3_{modality}"] = m3_root / f"{modality}_unet.pt"
    checkpoint_paths["m3_logistic"] = m3_root / "logistic.json"
    checkpoint_paths["m5_calibrator"] = Path("runs") / refs["m5_run_id"] / "calibrator.json"
    for seed in config["m4_seeds"]:
        for method in ("radar", "optical", "early", "robust_early"):
            checkpoint_paths[f"m4_{seed}_{method}"] = (Path("runs") / refs["m4_run_id"] /
                "experiments" / f"budget_{config['m4_budget_per_event']}" / f"seed_{seed}" / method / "model.pt")
    if set(checkpoint_paths) != set(manifest["reference_checkpoint_sha256"]):
        raise ValueError("Final checkpoint set differs from frozen methods")
    for key, path in checkpoint_paths.items():
        milestone = key[:2]
        digest = manifest["reference_checkpoint_sha256"][key]
        if reference_manifests[milestone]["output_sha256"].get(path.as_posix()) != digest or sha256(path) != digest:
            raise ValueError(f"Frozen source checkpoint differs: {key}")
    rows = json.loads((root / "per_tile.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if len(rows) != manifest["chips"] or len({row["tile_id"] for row in rows}) != len(rows):
        raise ValueError("Final rows missing or duplicated")
    if {event: sum(row["event_id"] == event for row in rows)
            for event in config["final_event_chip_counts"]} != config["final_event_chip_counts"]:
        raise ValueError("Final event counts differ")
    for row in rows:
        if row["water_pixels"] > row["eligible_pixels"] or row["retained"]["pixels"] > row["eligible_pixels"]:
            raise ValueError("Invalid per-chip pixel counts")
        for method in row["methods"].values():
            if sum(method[key] for key in ("tp", "fp", "fn", "tn")) != row["eligible_pixels"]:
                raise ValueError("Method support differs from common eligible pixels")
    if _aggregate(rows, config) != summary:
        raise ValueError("Event aggregates do not reproduce from per-chip counts")
    return {"run_id": run_id, "status": "verified", "chips": len(rows),
            "events": sorted(summary["events"]), "manifest_sha256": sha256(manifest_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id), indent=2))
