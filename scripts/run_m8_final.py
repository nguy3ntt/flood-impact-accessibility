"""One-time frozen Nigeria/Somalia observed-water evaluation.

No model fitting or test-driven threshold selection occurs here. Tile-level records
stay ignored locally; only reviewed event aggregates may be made public.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from flood_access.evidence import (calibrated_probability, quality_flags, reliability,
                                   retained_mask, unit_scaled_ndwi)
from flood_access.mapping import (confusion, load_tile, metrics, optical_index,
                                  predict_logistic, sum_confusions)
from flood_access.mapping_train import CompactUNet, predict_unet
from flood_access.raster_pipeline import sha256
from run_m5_evidence import aggregate_reliability
from stage_m8_final import CATALOG, CONFIG, STAGE_ROOT, frozen_rows, write_json
from prepare_m8_final import prepare  # Import validates the frozen preparation interface.
from flood_access.fusion import predict_model, late_fusion


def _unet(path: Path, *, modality: str | None = None, method: str | None = None) -> CompactUNet:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if modality is not None and saved.get("modality") != modality:
        raise ValueError(f"M3 model modality differs: {path}")
    if method is not None and saved.get("method") != method:
        raise ValueError(f"M4 model method differs: {path}")
    model = CompactUNet(saved["input_channels"], saved["base_channels"])
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def _models(config: dict) -> tuple[dict, dict, dict, dict]:
    refs = config["references"]
    for milestone in ("m3", "m4", "m5"):
        path = Path("runs") / refs[f"{milestone}_run_id"] / "run_manifest.json"
        if sha256(path) != refs[f"{milestone}_manifest_sha256"]:
            raise ValueError(f"Frozen {milestone.upper()} reference manifest differs")
    m3 = Path("runs") / refs["m3_run_id"] / "model"
    m4 = Path("runs") / refs["m4_run_id"] / "experiments"
    m5 = Path("runs") / refs["m5_run_id"] / "calibrator.json"
    checkpoints: dict[str, str] = {}
    m3_models = {}
    for modality in ("radar", "optical"):
        path = m3 / f"{modality}_unet.pt"
        checkpoints[f"m3_{modality}"] = sha256(path)
        m3_models[modality] = _unet(path, modality=modality)
    m4_models = {}
    for seed in config["m4_seeds"]:
        m4_models[seed] = {}
        for method in ("radar", "optical", "early", "robust_early"):
            path = m4 / f"budget_{config['m4_budget_per_event']}" / f"seed_{seed}" / method / "model.pt"
            checkpoints[f"m4_{seed}_{method}"] = sha256(path)
            m4_models[seed][method] = _unet(path, method=method)
    logistic_path = m3 / "logistic.json"
    checkpoints["m3_logistic"] = sha256(logistic_path)
    checkpoints["m5_calibrator"] = sha256(m5)
    weights = np.asarray(json.loads(logistic_path.read_text(encoding="utf-8"))["coefficients_and_intercept"],
                         dtype=np.float64)
    fit = json.loads(m5.read_text(encoding="utf-8"))
    return m3_models, m4_models, {"weights": weights, "fit": fit}, checkpoints


def _preparation(plan_id: str, config: dict) -> tuple[list[dict], Path, str]:
    plan_path = STAGE_ROOT / plan_id / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base = Path("data/processed/m8_final") / plan_id
    manifest_path = base / "preparation_manifest.json"
    status = json.loads((STAGE_ROOT / plan_id / "preparation_status.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (status["status"] != "complete" or status["manifest_sha256"] != sha256(manifest_path)
            or manifest["config_sha256"] != sha256(CONFIG)
            or manifest["plan_sha256"] != sha256(plan_path)
            or plan["config_sha256"] != sha256(CONFIG)):
        raise ValueError("Frozen M8 final preparation differs")
    for name, digest in manifest["output_sha256"].items():
        path = Path(name)
        if not path.is_relative_to(base) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Canonical final tile changed: {name}")
    rows = frozen_rows(json.loads(CATALOG.read_text(encoding="utf-8")), config)
    if len(rows) != manifest["tile_count"] or [r["tile_id"] for r in rows] != [r["tile_id"] for r in plan["tiles"]]:
        raise ValueError("Final tile selection differs")
    return rows, base / "tiles", sha256(manifest_path)


def _m4_score(models: dict, tile, method: str) -> np.ndarray:
    if method == "late":
        radar, _ = predict_model(models["radar"], tile, "radar", "clean",
                                 block_fraction=0.25, brightening_mix=0.35)
        optical, present = predict_model(models["optical"], tile, "optical", "clean",
                                         block_fraction=0.25, brightening_mix=0.35)
        return late_fusion(radar, optical, present, radar_weight=0.5)
    return predict_model(models[method], tile, method, "clean",
                         block_fraction=0.25, brightening_mix=0.35)[0]


def _aggregate(rows: list[dict], config: dict) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["event_id"]].append(row)
    events = {}
    bins = config["m5_reliability_bins"]
    for event, parts in sorted(grouped.items()):
        methods = {method: metrics(sum_confusions([part["methods"][method] for part in parts]))
                   for method in parts[0]["methods"]}
        rel = {name: aggregate_reliability([part["reliability"][name] for part in parts], bins)
               for name in parts[0]["reliability"]}
        retained = sum(part["retained"]["pixels"] for part in parts)
        eligible = sum(part["eligible_pixels"] for part in parts)
        water = sum(part["water_pixels"] for part in parts)
        events[event] = {"chips": len(parts), "eligible_pixels": eligible,
                         "water_pixels": water, "water_prevalence": water / eligible if eligible else None,
                         "methods": methods, "reliability": rel,
                         "retained": {"pixels": retained,
                                      "fraction": retained / eligible if eligible else None,
                                      "withheld_water": sum(part["retained"]["withheld_water"] for part in parts),
                                      "reliability": aggregate_reliability(
                                          [part["retained"]["reliability"] for part in parts], bins)}}
    method_names = list(next(iter(events.values()))["methods"])
    macro = {method: {
        "iou": float(np.mean([event["methods"][method]["iou"] for event in events.values()
                              if event["methods"][method]["iou"] is not None])),
        "f1": float(np.mean([event["methods"][method]["f1"] for event in events.values()
                             if event["methods"][method]["f1"] is not None])),
        "defined_events": sum(event["methods"][method]["iou"] is not None for event in events.values())}
        for method in method_names}
    return {"events": events, "event_macro": macro}


def evaluate(plan_id: str) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows, base, preparation_hash = _preparation(plan_id, config)
    m3_models, m4_models, fitted, checkpoint_hashes = _models(config)
    code = [Path("scripts/run_m8_final.py"), Path("scripts/prepare_m8_final.py"),
            Path("src/flood_access/mapping.py"), Path("src/flood_access/mapping_train.py"),
            Path("src/flood_access/fusion.py"), Path("src/flood_access/evidence.py")]
    code_hashes = {p.as_posix(): sha256(p) for p in code}
    run_id = "M8-" + hashlib.sha256((sha256(CONFIG) + preparation_hash +
                                     json.dumps(checkpoint_hashes, sort_keys=True) +
                                     json.dumps(code_hashes, sort_keys=True)).encode()).hexdigest()[:12]
    root = Path("runs") / run_id
    if (root / "status.json").exists():
        raise FileExistsError("Final evaluation already started; inspect status before any rerun")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "status.json", {"run_id": run_id, "status": "running",
                                      "started_at_utc": datetime.now(timezone.utc).isoformat()})
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = []
    try:
        for row in rows:
            tile = load_tile(base / row["tile_id"], row)
            eligible = int(np.count_nonzero(tile.eligible))
            water = int(np.count_nonzero(tile.eligible & (tile.label == 1)))
            scores = {"radar_vh": -tile.radar[1],
                      "ndwi": optical_index(tile, "ndwi"),
                      "mndwi": optical_index(tile, "mndwi"),
                      "logistic": predict_logistic(tile, fitted["weights"]),
                      "radar_unet": predict_unet(m3_models["radar"], tile, "radar"),
                      "optical_unet": predict_unet(m3_models["optical"], tile, "optical")}
            for seed, models in m4_models.items():
                for method in config["m4_clean_methods"]:
                    scores[f"m4_{seed}_{method}"] = _m4_score(models, tile, method)
            method_counts = {}
            for method, score in scores.items():
                if not np.isfinite(score[tile.eligible]).all():
                    raise ValueError(f"Non-finite eligible score: {row['tile_id']} {method}")
                threshold = config["m3_thresholds"].get(method, config["m4_threshold"])
                method_counts[method] = confusion(tile.label, score >= threshold, tile.eligible)
            probability = calibrated_probability(tile, fitted["fit"])
            flags = quality_flags(tile,
                                  weak_denominator_below=config["m5_quality_policy"]["weak_index_denominator_below"],
                                  radiometric_extreme_above=config["m5_quality_policy"]["radiometric_extreme_above_toa"])
            retained = retained_mask(tile, probability, flags,
                                     ambiguity_margin=config["m5_quality_policy"]["ambiguity_margin"],
                                     require_quality=True)
            reference = unit_scaled_ndwi(tile)
            prior = np.full(tile.label.shape, np.nan, dtype=np.float32)
            prior[tile.eligible] = fitted["fit"]["event_equal_sample_prevalence"]
            rel = {"calibrated": reliability(tile.label, probability, tile.eligible, config["m5_reliability_bins"]),
                   "unit_scaled_ndwi": reliability(tile.label, reference, tile.eligible, config["m5_reliability_bins"]),
                   "calibration_prevalence": reliability(tile.label, prior, tile.eligible, config["m5_reliability_bins"])}
            item = {"tile_id": row["tile_id"], "event_id": row["event_id"],
                    "eligible_pixels": eligible, "water_pixels": water,
                    "methods": method_counts, "reliability": rel,
                    "retained": {"pixels": int(np.count_nonzero(retained)),
                                 "withheld_water": int(np.count_nonzero(tile.eligible & (tile.label == 1) & ~retained)),
                                 "reliability": reliability(tile.label, probability, retained, config["m5_reliability_bins"])}}
            output.append(item)
            if len(output) % 5 == 0:
                print(f"Evaluated {len(output)}/{len(rows)} frozen final chips", flush=True)
        summary = _aggregate(output, config)
        write_json(root / "per_tile.json", output)
        write_json(root / "summary.json", summary)
        manifest = {"schema_version": "m8_final_run_v1", "run_id": run_id,
                    "status": "complete", "plan_id": plan_id, "config_sha256": sha256(CONFIG),
                    "preparation_manifest_sha256": preparation_hash,
                    "reference_checkpoint_sha256": checkpoint_hashes, "code_sha256": code_hashes,
                    "output_sha256": {name: sha256(root / name) for name in ("per_tile.json", "summary.json")},
                    "chips": len(output), "events": sorted(summary["events"]),
                    "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                    "torch": torch.__version__, "device": "cpu"},
                    "interpretation": "Observed-water final-test metrics only; two events; no flood or road-closure ground truth."}
        write_json(root / "run_manifest.json", manifest)
        write_json(root / "status.json", {"run_id": run_id, "status": "complete",
                                          "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                          "manifest_sha256": sha256(root / "run_manifest.json")})
    except Exception as error:
        write_json(root / "status.json", {"run_id": run_id, "status": "failed",
                                          "completed_chips": len(output),
                                          "error": f"{type(error).__name__}: {error}"})
        raise
    return {"run_id": run_id, "status": "complete", "chips": len(output),
            "event_macro": summary["event_macro"],
            "manifest_sha256": sha256(root / "run_manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_id")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.plan_id), indent=2))
