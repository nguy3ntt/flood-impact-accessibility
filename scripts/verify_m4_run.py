"""Verify an M4 fusion run, its split, metrics and sampled score replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from flood_access.fusion import METHODS, CONDITIONS, nested_train_subset
from flood_access.mapping import confusion, load_tile, metrics, sum_confusions
from flood_access.raster_pipeline import sha256
from run_m4_fusion import (CATALOG_PATH, CODE_FILES, CONFIG_PATH, _checkpoint_model,
                           _experiment_dir, _reference, _score)


def _equal_metric(left, right) -> bool:
    return (left is None and right is None) or (left is not None and right is not None and
                                                bool(np.isclose(left, right, atol=1e-12, rtol=0)))


def verify(run_id: str, *, smoke: bool = False) -> dict:
    if re.fullmatch(r"M4-[0-9a-f]{12}", run_id) is None:
        raise ValueError("Invalid M4 run ID")
    root, report_root = Path("runs") / run_id, Path("reports/m4") / run_id
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if smoke:
        config["train_tiles_per_event"] = [1]
        config["seeds"] = [4107]
        config["unet"]["epochs"] = 1
        config["unet"]["patches_per_tile"] = 1
        config["stress"]["evaluation_budget"] = 1
        config["evaluation"]["smoke"] = True
    m3, catalog, reference, base = _reference(config)
    code_hashes = {path.as_posix(): sha256(path) for path in CODE_FILES}
    fingerprint = {"m3_manifest_sha256": reference["manifest_sha256"],
                   "config": config, "code_sha256": code_hashes}
    expected_id = "M4-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    if run_id != expected_id:
        raise ValueError("Run ID differs from current inputs/code/config")
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id or manifest["status"] != "complete" or \
       status["status"] != "complete" or status["manifest_sha256"] != sha256(manifest_path) or \
       manifest["fingerprint"] != fingerprint or manifest["catalog_sha256"] != sha256(CATALOG_PATH):
        raise ValueError("M4 manifest/status/input integrity differs")
    for name, expected in manifest["output_sha256"].items():
        path = Path(name)
        if ".." in path.parts or not (path.is_relative_to(root) or path.is_relative_to(report_root)) or \
           not path.is_file() or sha256(path) != expected:
            raise ValueError(f"M4 output changed: {name}")
    actual = {path.as_posix() for folder in (root, report_root) for path in folder.rglob("*")
              if path.is_file() and path not in (manifest_path, root / "status.json")}
    if actual != set(manifest["output_sha256"]):
        raise ValueError("M4 output file set differs")
    if manifest["validation_tile_ids"] != m3["validation_tile_ids"] or \
       any(catalog[tile_id]["analysis_split"] != "validation" for tile_id in manifest["validation_tile_ids"]):
        raise ValueError("M4 validation group differs")
    train_tiles = [load_tile(base / tile_id, catalog[tile_id]) for tile_id in m3["train_tile_ids"]]
    for budget in config["train_tiles_per_event"]:
        expected_ids = [tile.tile_id for tile in nested_train_subset(train_tiles, budget)]
        if manifest["train_tile_ids_by_budget"][str(budget)] != expected_ids:
            raise ValueError("M4 training subset differs")
    del train_tiles
    validation_ids = manifest["validation_tile_ids"]
    if smoke:
        validation_ids = [row["tile_id"] for event in config["evaluation"]["heldout_events"]
                          for row in [catalog[t] for t in validation_ids if catalog[t]["event_id"] == event][:2]]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    recorded_summary = {(row["budget_per_event"], row["seed"], row["method"], row["condition"]): row
                        for row in summary["results"]}
    expected_summary_keys = set()
    task_count, replay_count = 0, 0
    replay_ids = {row["tile_id"] for event in config["evaluation"]["heldout_events"]
                  for row in [catalog[t] for t in validation_ids if catalog[t]["event_id"] == event][:1]}
    for budget in config["train_tiles_per_event"]:
        for seed in config["seeds"]:
            for method in METHODS:
                folder = _experiment_dir(root, budget, seed, method)
                result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                conditions = CONDITIONS if (budget == config["stress"]["evaluation_budget"] and
                                            seed == config["stress"]["evaluation_seed"]) else ("clean",)
                if result["budget_per_event"] != budget or result["seed"] != seed or result["method"] != method or \
                   result["conditions"] != list(conditions) or result["validation_tile_ids"] != validation_ids or \
                   result["train_tile_ids"] != manifest["train_tile_ids_by_budget"][str(budget)]:
                    raise ValueError("Experiment grouping or split differs")
                if method == "late":
                    sources = {name: sha256(_experiment_dir(root, budget, seed, name) / "model.pt")
                               for name in ("radar", "optical")}
                    if result["checkpoint_sha256"] is not None or \
                       result["profile"]["source_checkpoint_sha256"] != sources:
                        raise ValueError("Late-fusion source models differ")
                    models = {name: _checkpoint_model(_experiment_dir(root, budget, seed, name) / "model.pt", name)
                              for name in ("radar", "optical")}
                else:
                    checkpoint = folder / "model.pt"
                    if sha256(checkpoint) != result["checkpoint_sha256"]:
                        raise ValueError("M4 checkpoint changed")
                    models = {method: _checkpoint_model(checkpoint, method)}
                tile_rows = {(row["tile_id"], row["condition"]): row for row in result["per_tile"]}
                if len(tile_rows) != len(validation_ids) * len(conditions) or \
                   set(tile_rows) != {(tile_id, condition) for tile_id in validation_ids for condition in conditions}:
                    raise ValueError("Missing or duplicate M4 tile metric")
                for (tile_id, condition), row in tile_rows.items():
                    if row["method"] != method or row["event_id"] != catalog[tile_id]["event_id"] or \
                       sum(row[key] for key in ("tp", "fp", "fn", "tn")) != row["eligible_pixels"]:
                        raise ValueError("M4 tile confusion or event differs")
                    for key in ("iou", "f1", "precision", "recall"):
                        if not _equal_metric(row[key], metrics(row)[key]):
                            raise ValueError("M4 tile derived metric differs")
                    if tile_id in replay_ids:
                        tile = load_tile(base / tile_id, catalog[tile_id])
                        score, present = _score(models, tile, method, condition, config)
                        support = tile.eligible & present if method == "optical" else tile.eligible
                        counts = confusion(tile.label, score >= config["decision_threshold"], support)
                        if any(row[key] != counts[key] for key in counts) or \
                           row["original_common_pixels"] != int(np.count_nonzero(tile.eligible)) or \
                           row["optical_present_common_pixels"] != int(np.count_nonzero(tile.eligible & present)):
                            raise ValueError(f"M4 replay differs: {method}/{condition}/{tile_id}")
                        replay_count += 1
                event_rows = {(row["condition"], row["event_id"]): row for row in result["per_event"]}
                for condition in conditions:
                    grouped: dict[str, list[dict]] = {}
                    for row in result["per_tile"]:
                        if row["condition"] == condition:
                            grouped.setdefault(row["event_id"], []).append(row)
                    if set(grouped) != set(config["evaluation"]["heldout_events"]):
                        raise ValueError("M4 held-out events differ")
                    for event, rows in grouped.items():
                        counts = sum_confusions(rows)
                        actual_event = event_rows[(condition, event)]
                        if any(actual_event[key] != counts[key] for key in counts):
                            raise ValueError("M4 event confusion differs")
                        for key in ("iou", "f1", "precision", "recall"):
                            if not _equal_metric(actual_event[key], metrics(counts)[key]):
                                raise ValueError("M4 event derived metric differs")
                    event_iou = [event_rows[(condition, event)]["iou"] for event in grouped]
                    defined = [value for value in event_iou if value is not None]
                    expected_macro = float(np.mean(defined)) if defined else None
                    if not _equal_metric(expected_macro, result["macro"][condition]["event_macro_iou"]):
                        raise ValueError("M4 event macro differs")
                    summary_key = (budget, seed, method, condition)
                    expected_summary_keys.add(summary_key)
                    if not _equal_metric(recorded_summary[summary_key]["event_macro_iou"], expected_macro):
                        raise ValueError("M4 summary differs")
                task_count += 1
    if set(recorded_summary) != expected_summary_keys:
        raise ValueError("M4 summary row set differs")
    return {"run_id": run_id, "status": "verified", "tasks": task_count,
            "validation_tiles_per_task": len(validation_ids), "replayed_scores": replay_count,
            "output_files": len(actual), "manifest_sha256": sha256(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
