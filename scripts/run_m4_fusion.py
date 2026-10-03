"""Run bounded M4 fusion, label-budget and injected-quality comparisons.

Uses only the completed M3 development preparation. No final-test objects are
opened, no external weights are fetched, and all outputs stay under ignored roots.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from flood_access.fusion import (METHODS, CONDITIONS, late_fusion, nested_train_subset,
                                 optical_observation, predict_model, stress_block, train_model)
from flood_access.mapping import confusion, load_tile, metrics, optical_index, sum_confusions
from flood_access.mapping_train import CompactUNet
from flood_access.raster_pipeline import sha256
from verify_m3_run import verify as verify_m3


CONFIG_PATH = Path("configs/m4_fusion_v1.json")
CATALOG_PATH = Path("data/processed/m2_runs/M2-151ccd74b14c/catalog.json")
CODE_FILES = (Path("scripts/run_m4_fusion.py"), Path("src/flood_access/fusion.py"),
              Path("src/flood_access/mapping.py"), Path("src/flood_access/mapping_train.py"))
QA_TILE_IDS = ("Mekong_1111068", "Ghana_141910")


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checkpoint_model(path: Path, method: str) -> CompactUNet:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["method"] != method:
        raise ValueError("Checkpoint method differs")
    model = CompactUNet(saved["input_channels"], saved["base_channels"])
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def _experiment_dir(root: Path, budget: int, seed: int, method: str) -> Path:
    return root / "experiments" / f"budget_{budget}" / f"seed_{seed}" / method


def _reference(config: dict) -> tuple[dict, dict, dict, Path]:
    reference = verify_m3(config["m3_reference_run_id"])
    m3_root = Path("runs") / config["m3_reference_run_id"]
    manifest_path = m3_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["plan_id"] != config["m3_plan_id"] or manifest["run_id"] != reference["run_id"]:
        raise ValueError("M4 reference and prepared plan differ")
    catalog = {row["tile_id"]: row for row in json.loads(CATALOG_PATH.read_text(encoding="utf-8"))}
    if set(manifest["train_tile_ids"]) & set(manifest["validation_tile_ids"]) or \
       any(catalog[t]["analysis_split"] != "train" for t in manifest["train_tile_ids"]) or \
       any(catalog[t]["analysis_split"] != "validation" for t in manifest["validation_tile_ids"]):
        raise ValueError("M4 reference crosses frozen split")
    base = Path("data/processed/m3_development") / config["m3_plan_id"] / "tiles"
    return manifest, catalog, reference, base


def _score(models: dict[str, CompactUNet], tile, method: str, condition: str,
           config: dict) -> tuple[np.ndarray, np.ndarray]:
    stress = config["stress"]
    kwargs = {"block_fraction": stress["block_fraction"],
              "brightening_mix": stress["brightening_mix"]}
    if method == "late":
        radar, _ = predict_model(models["radar"], tile, "radar", condition, **kwargs)
        optical, present = predict_model(models["optical"], tile, "optical", condition, **kwargs)
        return late_fusion(radar, optical, present, radar_weight=config["late_fusion"]["radar_weight"]), present
    return predict_model(models[method], tile, method, condition, **kwargs)


def _evaluate(models: dict[str, CompactUNet], method: str, validation_rows: list[dict],
              base: Path, config: dict, conditions: tuple[str, ...]) -> tuple[list[dict], float]:
    rows = []
    start = time.perf_counter()
    threshold = config["decision_threshold"]
    for expected in validation_rows:
        tile = load_tile(base / expected["tile_id"], expected)
        labelled = int(np.count_nonzero(tile.label != -1))
        original = int(np.count_nonzero(tile.eligible))
        for condition in conditions:
            score, present = _score(models, tile, method, condition, config)
            # Optical-only inference cannot report a map where optical is absent.
            support = tile.eligible & present if method == "optical" else tile.eligible
            counts = confusion(tile.label, score >= threshold, support)
            if int(np.count_nonzero(support)) != counts["eligible_pixels"]:
                raise ValueError("Evaluated support differs")
            rows.append({"tile_id": tile.tile_id, "event_id": tile.event_id,
                         "condition": condition, "method": method,
                         "label_valid_pixels": labelled, "original_common_pixels": original,
                         "optical_present_common_pixels": int(np.count_nonzero(tile.eligible & present)),
                         "support_fraction_original": counts["eligible_pixels"] / original if original else None,
                         **metrics(counts)})
    return rows, time.perf_counter() - start


def _summarise(rows: list[dict], conditions: tuple[str, ...]) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["event_id"])].append(row)
    per_event = []
    macro = {}
    for condition in conditions:
        event_rows = []
        for (name, event), values in sorted(grouped.items()):
            if name != condition:
                continue
            counts = sum_confusions(values)
            original = sum(row["original_common_pixels"] for row in values)
            event_rows.append({"condition": condition, "event_id": event,
                               "original_common_pixels": original,
                               "support_fraction_original": counts["eligible_pixels"] / original if original else None,
                               **metrics(counts)})
        per_event.extend(event_rows)
        defined = [row["iou"] for row in event_rows if row["iou"] is not None]
        f1 = [row["f1"] for row in event_rows if row["f1"] is not None]
        macro[condition] = {"event_macro_iou": float(np.mean(defined)) if defined else None,
                            "event_macro_f1": float(np.mean(f1)) if f1 else None,
                            "defined_events": len(defined),
                            "mean_event_support_fraction": float(np.mean([row["support_fraction_original"] for row in event_rows]))}
    return per_event, macro


def _result(root: Path, budget: int, seed: int, method: str, train_ids: list[str],
            validation_rows: list[dict], base: Path, catalog: dict[str, dict],
            config: dict, resume: bool) -> dict:
    folder = _experiment_dir(root, budget, seed, method)
    result_path = folder / "result.json"
    conditions = CONDITIONS if (budget == config["stress"]["evaluation_budget"] and
                                seed == config["stress"]["evaluation_seed"]) else ("clean",)
    if resume and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous["train_tile_ids"] == train_ids and previous["conditions"] == list(conditions):
            checkpoint = folder / "model.pt"
            late_sources = {source: sha256(_experiment_dir(root, budget, seed, source) / "model.pt")
                            for source in ("radar", "optical")} if method == "late" else None
            if (method == "late" and late_sources == previous["profile"]["source_checkpoint_sha256"]) or \
               (checkpoint.is_file() and sha256(checkpoint) == previous["checkpoint_sha256"]):
                return previous
    started = time.perf_counter()
    if method == "late":
        radar_path = _experiment_dir(root, budget, seed, "radar") / "model.pt"
        optical_path = _experiment_dir(root, budget, seed, "optical") / "model.pt"
        models = {"radar": _checkpoint_model(radar_path, "radar"),
                  "optical": _checkpoint_model(optical_path, "optical")}
        profile = {"training_seconds": 0.0, "parameters": 0,
                   "source_checkpoint_sha256": {"radar": sha256(radar_path), "optical": sha256(optical_path)}}
        checkpoint_sha256 = None
    else:
        training = [load_tile(base / tile_id, catalog[tile_id]) for tile_id in train_ids]
        params = config["unet"]
        model, profile = train_model(training, method, folder / "model.pt", seed=seed,
                                     epochs=params["epochs"], patches_per_tile=params["patches_per_tile"],
                                     patch_size=params["patch_size"], base_channels=params["base_channels"],
                                     learning_rate=params["learning_rate"],
                                     full_dropout_probability=params["robust_full_dropout_probability"],
                                     block_dropout_probability=params["robust_block_dropout_probability"],
                                     block_fraction=config["stress"]["block_fraction"])
        del training
        models = {method: model}
        checkpoint_sha256 = sha256(folder / "model.pt")
    tile_rows, inference_seconds = _evaluate(models, method, validation_rows, base, config, conditions)
    events, macro = _summarise(tile_rows, conditions)
    result = {"budget_per_event": budget, "seed": seed, "method": method,
              "train_tile_ids": train_ids, "validation_tile_ids": [row["tile_id"] for row in validation_rows],
              "conditions": list(conditions), "checkpoint_sha256": checkpoint_sha256,
              "profile": {**profile, "inference_seconds": inference_seconds,
                          "total_task_seconds": time.perf_counter() - started},
              "per_tile": tile_rows, "per_event": events, "macro": macro}
    write_json(result_path, result)
    return result


def _panel(path: Path, tile, scores: dict[str, np.ndarray], condition: str,
           threshold: float, block: np.ndarray | None) -> None:
    scale = 400
    methods = list(scores)
    image = Image.new("RGB", (len(methods) * scale, scale + 75), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for index, method in enumerate(methods):
        score = scores[method]
        valid = tile.eligible & np.isfinite(score)
        prediction = score >= (-0.05 if method == "ndwi" else threshold)
        rgb = np.full((*tile.label.shape, 3), (130, 130, 130), dtype=np.uint8)
        rgb[valid & (tile.label == 0) & ~prediction] = (56, 52, 48)
        rgb[valid & (tile.label == 1) & prediction] = (42, 170, 92)
        rgb[valid & (tile.label == 0) & prediction] = (226, 66, 55)
        rgb[valid & (tile.label == 1) & ~prediction] = (45, 116, 230)
        image.paste(Image.fromarray(rgb).resize((scale, scale), Image.Resampling.NEAREST),
                    (index * scale, 28))
        draw.text((index * scale + 6, 7), method, fill=(0, 0, 0))
        if block is not None:
            ys, xs = np.where(block)
            if len(ys):
                box = (index * scale + int(xs.min() * scale / tile.label.shape[1]),
                       28 + int(ys.min() * scale / tile.label.shape[0]),
                       index * scale + int((xs.max() + 1) * scale / tile.label.shape[1]),
                       28 + int((ys.max() + 1) * scale / tile.label.shape[0]))
                draw.rectangle(box, outline=(255, 220, 0), width=3)
    draw.text((5, scale + 38), f"{tile.tile_id} | {condition} | yellow: injected block; gray: unknown/unavailable; red: false water; blue: missed water", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _qa(root: Path, report_root: Path, base: Path, catalog: dict[str, dict], config: dict) -> None:
    budget = config["stress"]["evaluation_budget"]
    seed = config["stress"]["evaluation_seed"]
    models = {method: _checkpoint_model(_experiment_dir(root, budget, seed, method) / "model.pt", method)
              for method in METHODS if method != "late"}
    for tile_id in QA_TILE_IDS:
        tile = load_tile(base / tile_id, catalog[tile_id])
        for condition in ("clean", "block_missing"):
            scores = {}
            ndwi = optical_index(tile, "ndwi")
            if condition == "block_missing":
                ndwi[stress_block(tile_id, tile.label.shape, config["stress"]["block_fraction"])] = np.nan
            scores["ndwi"] = ndwi
            for method in METHODS:
                score, present = _score(models, tile, method, condition, config)
                if method == "optical":
                    score[~present] = np.nan
                scores[method] = score
            block = stress_block(tile_id, tile.label.shape, config["stress"]["block_fraction"]) \
                if condition == "block_missing" else None
            _panel(report_root / f"qa_{tile_id}_{condition}.png", tile, scores,
                   condition, config["decision_threshold"], block)


def run(*, resume: bool = False, smoke: bool = False) -> dict:
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
    run_id = "M4-" + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    root, report_root = Path("runs") / run_id, Path("reports/m4") / run_id
    status_path = root / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] == "complete":
            raise FileExistsError(f"Completed M4 run exists: {run_id}; verify it instead")
        if not resume:
            raise FileExistsError(f"Incomplete M4 run exists: {run_id}; rerun with --resume")
    train_tiles = [load_tile(base / tile_id, catalog[tile_id]) for tile_id in m3["train_tile_ids"]]
    selected = {budget: [tile.tile_id for tile in nested_train_subset(train_tiles, budget)]
                for budget in config["train_tiles_per_event"]}
    del train_tiles
    validation_rows = [catalog[tile_id] for tile_id in m3["validation_tile_ids"]]
    if smoke:
        validation_rows = [row for event in config["evaluation"]["heldout_events"]
                           for row in [item for item in validation_rows if item["event_id"] == event][:2]]
    if {row["event_id"] for row in validation_rows} != set(config["evaluation"]["heldout_events"]):
        raise ValueError("Held-out development events differ")
    write_json(status_path, {"run_id": run_id, "status": "running", "completed_tasks": 0,
                             "started_at_utc": datetime.now(timezone.utc).isoformat()})
    start = time.perf_counter()
    try:
        results = []
        for budget in config["train_tiles_per_event"]:
            for seed in config["seeds"]:
                for method in METHODS:
                    result = _result(root, budget, seed, method, selected[budget],
                                     validation_rows, base, catalog, config, resume)
                    results.append(result)
                    write_json(status_path, {"run_id": run_id, "status": "running",
                                             "completed_tasks": len(results), "latest_task": [budget, seed, method],
                                             "resume_command": "python scripts/run_m4_fusion.py --resume"})
                    print(json.dumps({"task": [budget, seed, method],
                                      "clean_event_macro_iou": result["macro"]["clean"]["event_macro_iou"],
                                      "tasks_complete": len(results)}), flush=True)
        if not smoke:
            _qa(root, report_root, base, catalog, config)
        summary = []
        for result in results:
            for condition, row in result["macro"].items():
                summary.append({"budget_per_event": result["budget_per_event"], "seed": result["seed"],
                                "method": result["method"], "condition": condition, **row,
                                "per_event": [event for event in result["per_event"] if event["condition"] == condition],
                                "training_seconds": result["profile"]["training_seconds"],
                                "inference_seconds": result["profile"]["inference_seconds"]})
        write_json(root / "summary.json", {"schema_version": "m4_summary_v1", "run_id": run_id,
                                           "m3_reference_run_id": m3["run_id"], "results": summary,
                                           "interpretation": ["Validation events support development comparison only.",
                                                              "Cloud and missing-optical conditions are injected stress tests.",
                                                              "No predicted score is calibrated or a road-closure probability."]})
        output_files = sorted(path for directory in (root, report_root) for path in directory.rglob("*")
                              if path.is_file() and path not in (status_path, root / "run_manifest.json"))
        manifest = {"schema_version": "m4_run_v1", "run_id": run_id, "status": "complete",
                    "fingerprint": fingerprint, "m3_reference_run_id": m3["run_id"],
                    "m3_manifest_sha256": reference["manifest_sha256"],
                    "preparation_manifest_sha256": m3["preparation_manifest_sha256"],
                    "catalog_sha256": sha256(CATALOG_PATH), "config_sha256": sha256(CONFIG_PATH),
                    "code_sha256": code_hashes, "train_tile_ids_by_budget": selected,
                    "validation_tile_ids": m3["validation_tile_ids"],
                    "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                    "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
                    "output_sha256": {path.as_posix(): sha256(path) for path in output_files},
                    "wall_seconds": time.perf_counter() - start,
                    "foundation_branch": {"status": "deferred_input_and_capacity_gate",
                                          "reason": "Six-band 30 m HLS surface reflectance is not the Sen1 L1C TOA input; CPU-only runtime and insufficient disk headroom for 300M weights."}}
        write_json(root / "run_manifest.json", manifest)
        write_json(status_path, {"run_id": run_id, "status": "complete",
                                 "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "manifest_sha256": sha256(root / "run_manifest.json")})
        return {"run_id": run_id, "tasks": len(results), "outputs": len(output_files),
                "wall_seconds": manifest["wall_seconds"]}
    except Exception as error:
        write_json(status_path, {"run_id": run_id, "status": "failed",
                                 "completed_tasks": len(results) if "results" in locals() else 0,
                                 "error": f"{type(error).__name__}: {error}",
                                 "resume_command": "python scripts/run_m4_fusion.py --resume"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(resume=args.resume, smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
