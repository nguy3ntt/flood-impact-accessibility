"""Verify fixed-update M4 budget results and replay held-out predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from flood_access.fusion import METHODS, nested_train_subset
from flood_access.mapping import confusion, load_tile, metrics, sum_confusions
from flood_access.raster_pipeline import sha256
from run_m4_budget_control import _fingerprint, _settings
from run_m4_fusion import _checkpoint_model, _experiment_dir, _reference, _score


def _same(a, b) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and
                                         bool(np.isclose(a, b, atol=1e-12, rtol=0)))


def verify(run_id: str) -> dict:
    if re.fullmatch(r"M4B-[0-9a-f]{12}", run_id) is None:
        raise ValueError("Invalid budget-control run ID")
    control, config = _settings()
    m3, catalog, reference, tile_root = _reference(config)
    fingerprint = _fingerprint(control, config, reference["manifest_sha256"])
    expected_id = "M4B-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    if run_id != expected_id:
        raise ValueError("Budget-control run fingerprint differs")
    root, report_root = Path("runs") / run_id, Path("reports/m4") / run_id
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id or manifest["status"] != "complete" or \
       status["status"] != "complete" or status["manifest_sha256"] != sha256(manifest_path) or \
       manifest["fingerprint"] != fingerprint:
        raise ValueError("Budget-control run manifest/status differs")
    for name, expected in manifest["output_sha256"].items():
        path = Path(name)
        if ".." in path.parts or not (path.is_relative_to(root) or path.is_relative_to(report_root)) or \
           not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Budget-control output changed: {name}")
    actual = {path.as_posix() for folder in (root, report_root) for path in folder.rglob("*")
              if path.is_file() and path not in (root / "status.json", manifest_path)}
    if actual != set(manifest["output_sha256"]):
        raise ValueError("Budget-control output file set differs")
    if manifest["validation_tile_ids"] != m3["validation_tile_ids"] or \
       any(catalog[tile_id]["analysis_split"] != "validation" for tile_id in manifest["validation_tile_ids"]):
        raise ValueError("Budget-control validation differs")
    source_tiles = [load_tile(tile_root / tile_id, catalog[tile_id]) for tile_id in m3["train_tile_ids"]]
    for budget in control["budgets_per_event"]:
        ids = [tile.tile_id for tile in nested_train_subset(source_tiles, budget)]
        if manifest["train_tile_ids_by_budget"][str(budget)] != ids:
            raise ValueError("Budget-control nested split differs")
    del source_tiles
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary_rows = {(row["budget_per_event"], row["seed"], row["method"]): row for row in summary["results"]}
    expected_keys = set()
    replay_ids = {next(tile_id for tile_id in m3["validation_tile_ids"]
                       if catalog[tile_id]["event_id"] == event)
                  for event in config["evaluation"]["heldout_events"]}
    replayed = 0
    observed_steps: dict[tuple[int, int, str], int] = {}
    for budget in control["budgets_per_event"]:
        budget_config = json.loads(json.dumps(config))
        budget_config["unet"]["patches_per_tile"] = control["patches_per_tile_by_budget"][str(budget)]
        expected_steps = len(manifest["train_tile_ids_by_budget"][str(budget)]) * \
            control["patches_per_tile_by_budget"][str(budget)] * control["epochs"]
        for seed in control["seeds"]:
            for method in METHODS:
                key = budget, seed, method
                expected_keys.add(key)
                folder = _experiment_dir(root, budget, seed, method)
                result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                if result["budget_per_event"] != budget or result["seed"] != seed or result["method"] != method or \
                   result["conditions"] != ["clean"] or result["validation_tile_ids"] != m3["validation_tile_ids"] or \
                   result["train_tile_ids"] != manifest["train_tile_ids_by_budget"][str(budget)]:
                    raise ValueError("Budget-control task/split differs")
                if method == "late":
                    sources = {name: sha256(_experiment_dir(root, budget, seed, name) / "model.pt")
                               for name in ("radar", "optical")}
                    if result["profile"]["source_checkpoint_sha256"] != sources:
                        raise ValueError("Late model sources differ")
                    models = {name: _checkpoint_model(_experiment_dir(root, budget, seed, name) / "model.pt", name)
                              for name in ("radar", "optical")}
                else:
                    checkpoint = folder / "model.pt"
                    performed_steps = sum(row["steps"] for row in result["profile"]["curve"])
                    if sha256(checkpoint) != result["checkpoint_sha256"] or \
                       not expected_steps * (1 - control["maximum_training_step_difference_fraction"]) <= performed_steps <= expected_steps:
                        raise ValueError("Model checkpoint or optimizer steps differ")
                    observed_steps[key] = performed_steps
                    models = {method: _checkpoint_model(checkpoint, method)}
                tiles = {row["tile_id"]: row for row in result["per_tile"]}
                if set(tiles) != set(m3["validation_tile_ids"]) or len(result["per_tile"]) != len(tiles):
                    raise ValueError("Budget-control per-tile results incomplete")
                grouped: dict[str, list[dict]] = {}
                for tile_id, row in tiles.items():
                    if row["method"] != method or row["condition"] != "clean" or \
                       row["event_id"] != catalog[tile_id]["event_id"] or \
                       sum(row[name] for name in ("tp", "fp", "fn", "tn")) != row["eligible_pixels"] or \
                       row["eligible_pixels"] != row["original_common_pixels"]:
                        raise ValueError("Budget-control tile counts/support differ")
                    if any(not _same(row[name], metrics(row)[name]) for name in ("iou", "f1", "precision", "recall")):
                        raise ValueError("Budget-control tile metric differs")
                    grouped.setdefault(row["event_id"], []).append(row)
                    if tile_id in replay_ids:
                        tile = load_tile(tile_root / tile_id, catalog[tile_id])
                        score, _ = _score(models, tile, method, "clean", budget_config)
                        counts = confusion(tile.label, score >= budget_config["decision_threshold"], tile.eligible)
                        if any(counts[name] != row[name] for name in counts):
                            raise ValueError(f"Budget-control replay differs: {key}/{tile_id}")
                        replayed += 1
                events = {row["event_id"]: row for row in result["per_event"]}
                if set(events) != set(config["evaluation"]["heldout_events"]):
                    raise ValueError("Budget-control event rows differ")
                for event, rows in grouped.items():
                    counts = sum_confusions(rows)
                    if any(counts[name] != events[event][name] for name in counts) or \
                       any(not _same(events[event][name], metrics(counts)[name])
                           for name in ("iou", "f1", "precision", "recall")):
                        raise ValueError("Budget-control event metrics differ")
                macro = float(np.mean([events[event]["iou"] for event in events]))
                if not _same(macro, result["macro"]["clean"]["event_macro_iou"]) or \
                   not _same(macro, summary_rows[key]["event_macro_iou"]):
                    raise ValueError("Budget-control macro summary differs")
                ndwi = summary["m3_ndwi_event_macro_iou"]
                gain = macro - ndwi
                worst_loss = max(summary["m3_ndwi_per_event_iou"][event] - events[event]["iou"]
                                 for event in events)
                rule = control["candidate_rule"]
                passed = gain >= rule["minimum_macro_iou_gain_over_ndwi"] and \
                    worst_loss <= rule["maximum_single_event_iou_loss"]
                if not _same(summary_rows[key]["gain_over_m3_ndwi"], gain) or \
                   not _same(summary_rows[key]["worst_event_loss_vs_m3_ndwi"], worst_loss) or \
                   summary_rows[key]["passes_clean_promotion_rule"] != passed:
                    raise ValueError("Budget-control decision fields differ")
                if summary_rows[key]["optimizer_steps"] != observed_steps.get(key, 0):
                    raise ValueError("Budget-control observed update count differs")
    if set(summary_rows) != expected_keys:
        raise ValueError("Budget-control summary row set differs")
    for seed in control["seeds"]:
        for method in METHODS[:-1]:
            steps = [observed_steps[(budget, seed, method)] for budget in control["budgets_per_event"]]
            if (max(steps) - min(steps)) / max(steps) > control["maximum_training_step_difference_fraction"]:
                raise ValueError("Realised training updates are not matched across budgets")
    for method in METHODS:
        expected = all(summary_rows[(max(control["budgets_per_event"]), seed, method)]["passes_clean_promotion_rule"]
                       for seed in control["seeds"])
        if summary["promoted_on_clean_rule_all_seeds"][method] != expected:
            raise ValueError("Budget-control promotion decision differs")
    return {"run_id": run_id, "status": "verified", "tasks": len(expected_keys),
            "validation_tiles_per_task": len(m3["validation_tile_ids"]),
            "replayed_scores": replayed, "output_files": len(actual),
            "manifest_sha256": sha256(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id), indent=2))


if __name__ == "__main__":
    main()
