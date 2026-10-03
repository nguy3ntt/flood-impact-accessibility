"""Match training updates across M4 label budgets and repeat clean event transfer.

This controlled follow-up retains the first M4 run as exploratory evidence. It
reuses its fixed architecture, split and evaluation code without new downloads.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from flood_access.fusion import METHODS, nested_train_subset
from flood_access.mapping import load_tile, optical_index
from flood_access.raster_pipeline import sha256
from run_m4_fusion import (CATALOG_PATH, CODE_FILES, QA_TILE_IDS, _checkpoint_model,
                           _experiment_dir, _panel, _reference, _result, write_json,
                           _score)


CONTROL_PATH = Path("configs/m4_budget_control_v1.json")
BASE_CONFIG_PATH = Path("configs/m4_fusion_v1.json")


def _settings() -> tuple[dict, dict]:
    control = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    if control["budgets_per_event"] != base["train_tiles_per_event"] or control["seeds"] != base["seeds"] or \
       control["evaluation_conditions"] != ["clean"] or control["epochs"] != base["unet"]["epochs"]:
        raise ValueError("Controlled experiment must preserve M4 split, seeds and epochs")
    # A zero budget is absent from the experiment grid, so _result evaluates clean only.
    base["stress"]["evaluation_budget"] = 0
    base["stress"]["evaluation_seed"] = 0
    return control, base


def _fingerprint(control: dict, base: dict, m3_sha: str) -> dict:
    files = (*CODE_FILES, Path("scripts/run_m4_budget_control.py"))
    return {"m3_manifest_sha256": m3_sha, "control": control, "base_evaluation_config": base,
            "code_sha256": {path.as_posix(): sha256(path) for path in files}}


def _qa(root: Path, report_root: Path, base: Path, catalog: dict, config: dict) -> None:
    budget, seed = max(config["train_tiles_per_event"]), config["seeds"][0]
    models = {method: _checkpoint_model(_experiment_dir(root, budget, seed, method) / "model.pt", method)
              for method in METHODS if method != "late"}
    for tile_id in QA_TILE_IDS:
        tile = load_tile(base / tile_id, catalog[tile_id])
        scores = {"ndwi": optical_index(tile, "ndwi")}
        for method in METHODS:
            score, _ = _score(models, tile, method, "clean", config)
            scores[method] = score
        _panel(report_root / f"qa_{tile_id}_clean.png", tile, scores,
               "clean fixed-update comparison", config["decision_threshold"], None)


def run(*, resume: bool = False) -> dict:
    control, base_config = _settings()
    m3, catalog, reference, tile_root = _reference(base_config)
    fingerprint = _fingerprint(control, base_config, reference["manifest_sha256"])
    run_id = "M4B-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    root, report_root = Path("runs") / run_id, Path("reports/m4") / run_id
    status_path = root / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous["status"] == "complete":
            raise FileExistsError(f"Completed budget-control run exists: {run_id}; verify it instead")
        if not resume:
            raise FileExistsError(f"Incomplete budget-control run exists: {run_id}; use --resume")
    source_tiles = [load_tile(tile_root / tile_id, catalog[tile_id]) for tile_id in m3["train_tile_ids"]]
    selected = {budget: [tile.tile_id for tile in nested_train_subset(source_tiles, budget)]
                for budget in control["budgets_per_event"]}
    del source_tiles
    step_counts = {budget: len(selected[budget]) * control["patches_per_tile_by_budget"][str(budget)]
                   for budget in control["budgets_per_event"]}
    if (max(step_counts.values()) - min(step_counts.values())) / max(step_counts.values()) > \
       control["maximum_training_step_difference_fraction"]:
        raise ValueError("Training-update counts are insufficiently matched")
    validation_rows = [catalog[tile_id] for tile_id in m3["validation_tile_ids"]]
    if {row["event_id"] for row in validation_rows} != set(base_config["evaluation"]["heldout_events"]):
        raise ValueError("Development validation groups differ")
    write_json(status_path, {"run_id": run_id, "status": "running", "completed_tasks": 0,
                             "started_at_utc": datetime.now(timezone.utc).isoformat()})
    start = time.perf_counter()
    results = []
    try:
        for budget in control["budgets_per_event"]:
            config = json.loads(json.dumps(base_config))
            config["unet"]["patches_per_tile"] = control["patches_per_tile_by_budget"][str(budget)]
            for seed in control["seeds"]:
                for method in METHODS:
                    result = _result(root, budget, seed, method, selected[budget], validation_rows,
                                     tile_root, catalog, config, resume)
                    results.append(result)
                    write_json(status_path, {"run_id": run_id, "status": "running",
                                             "completed_tasks": len(results), "latest_task": [budget, seed, method],
                                             "resume_command": "python scripts/run_m4_budget_control.py --resume"})
                    print(json.dumps({"task": [budget, seed, method], "event_macro_iou":
                                      result["macro"]["clean"]["event_macro_iou"],
                                      "tasks_complete": len(results)}), flush=True)
        _qa(root, report_root, tile_root, catalog, base_config)
        reference_metrics = json.loads((Path("runs") / m3["run_id"] / "metrics.json").read_text(encoding="utf-8"))
        ndwi = reference_metrics["summary"]["ndwi"]["event_macro_iou"]
        ndwi_events = {row["event_id"]: row["iou"] for row in reference_metrics["per_event"]
                       if row["method"] == "ndwi"}
        summary = []
        for result in results:
            events = {row["event_id"]: row["iou"] for row in result["per_event"]}
            gain = result["macro"]["clean"]["event_macro_iou"] - ndwi
            worst_loss = max(ndwi_events[event] - events[event] for event in ndwi_events)
            rule = control["candidate_rule"]
            summary.append({"budget_per_event": result["budget_per_event"], "seed": result["seed"],
                            "method": result["method"], "event_macro_iou": result["macro"]["clean"]["event_macro_iou"],
                            "event_macro_f1": result["macro"]["clean"]["event_macro_f1"],
                            "per_event": result["per_event"], "gain_over_m3_ndwi": gain,
                            "worst_event_loss_vs_m3_ndwi": worst_loss,
                            "passes_clean_promotion_rule": gain >= rule["minimum_macro_iou_gain_over_ndwi"] and
                                                           worst_loss <= rule["maximum_single_event_iou_loss"],
                            "training_seconds": result["profile"]["training_seconds"],
                            "inference_seconds": result["profile"]["inference_seconds"],
                            "optimizer_steps": sum(row["steps"] for row in result["profile"].get("curve", []))})
        full_budget = max(control["budgets_per_event"])
        promotion = {method: all(row["passes_clean_promotion_rule"] for row in summary
                                 if row["budget_per_event"] == full_budget and row["method"] == method)
                     for method in METHODS}
        if any(sum(row["budget_per_event"] == full_budget and row["method"] == method for row in summary)
               != len(control["seeds"]) for method in METHODS):
            raise ValueError("Missing full-budget seed for candidate rule")
        write_json(root / "summary.json", {"schema_version": "m4_budget_control_summary_v1",
                                           "run_id": run_id, "m3_ndwi_event_macro_iou": ndwi,
                                           "m3_ndwi_per_event_iou": ndwi_events,
                                           "optimizer_steps_per_epoch": step_counts,
                                           "results": summary, "promoted_on_clean_rule_all_seeds": promotion,
                                           "interpretation": "Two validation events; this is a development decision, not final transfer evidence."})
        outputs = sorted(path for folder in (root, report_root) for path in folder.rglob("*")
                         if path.is_file() and path not in (status_path, root / "run_manifest.json"))
        manifest = {"schema_version": "m4_budget_control_run_v1", "run_id": run_id,
                    "status": "complete", "fingerprint": fingerprint, "m3_reference_run_id": m3["run_id"],
                    "m3_manifest_sha256": reference["manifest_sha256"],
                    "train_tile_ids_by_budget": selected, "validation_tile_ids": m3["validation_tile_ids"],
                    "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                    "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
                    "output_sha256": {path.as_posix(): sha256(path) for path in outputs},
                    "wall_seconds": time.perf_counter() - start}
        write_json(root / "run_manifest.json", manifest)
        write_json(status_path, {"run_id": run_id, "status": "complete",
                                 "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "manifest_sha256": sha256(root / "run_manifest.json")})
        return {"run_id": run_id, "tasks": len(results), "outputs": len(outputs),
                "wall_seconds": manifest["wall_seconds"], "promotion": promotion}
    except Exception as error:
        write_json(status_path, {"run_id": run_id, "status": "failed", "completed_tasks": len(results),
                                 "error": f"{type(error).__name__}: {error}",
                                 "resume_command": "python scripts/run_m4_budget_control.py --resume"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
