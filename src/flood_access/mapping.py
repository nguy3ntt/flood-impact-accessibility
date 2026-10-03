"""Observed-water baselines and event-level evaluation for canonical M2 tiles.

Scores are evidence about the source water label, not flood depth, road closure,
calibrated probability, or safe travel. Every method uses the same eligible pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import rasterio

from .raster_pipeline import S1_BANDS, S2_BANDS, VALIDITY_BANDS


EPSILON = np.float32(1e-6)
OPTICAL_CHANNELS = ("B3", "B4", "B8", "B11", "B12")


@dataclass
class MappingTile:
    tile_id: str
    event_id: str
    split: str
    label: np.ndarray
    radar: np.ndarray
    optical: np.ndarray
    eligible: np.ndarray
    transform: object
    crs: object


def load_tile(tile_dir: Path, expected: dict) -> MappingTile:
    metadata = json.loads((tile_dir / "metadata.json").read_text(encoding="utf-8"))
    for key, value in (("tile_id", expected["tile_id"]), ("event_id", expected["event_id"]),
                       ("analysis_split", expected["analysis_split"])):
        if metadata[key] != value:
            raise ValueError(f"Canonical tile {key} differs from frozen catalog")
    with rasterio.open(tile_dir / "label.tif") as label_source, \
         rasterio.open(tile_dir / "radar_db.tif") as radar_source, \
         rasterio.open(tile_dir / "optical_toa.tif") as optical_source, \
         rasterio.open(tile_dir / "validity.tif") as validity_source, \
         rasterio.open(tile_dir / "cloud_status.tif") as cloud_source:
        sources = (radar_source, optical_source, validity_source, cloud_source)
        if any(source.crs != label_source.crs or source.transform != label_source.transform
               or source.shape != label_source.shape for source in sources):
            raise ValueError(f"Canonical tile grids differ: {tile_dir}")
        if (label_source.count, radar_source.descriptions, optical_source.descriptions,
            validity_source.descriptions, cloud_source.count) != (1, S1_BANDS, S2_BANDS,
                                                                 VALIDITY_BANDS, 1):
            raise ValueError(f"Canonical tile band contract differs: {tile_dir}")
        label = label_source.read(1)
        radar = radar_source.read().astype(np.float32)
        optical = optical_source.read().astype(np.float32)
        validity = validity_source.read() == 1
        cloud = cloud_source.read(1)
        transform, crs = label_source.transform, label_source.crs
    if not np.isin(label, (-1, 0, 1)).all() or not np.all(cloud == 255):
        raise ValueError(f"Unknown label/cloud semantics changed: {tile_dir}")
    if not np.array_equal(validity[0], label != -1):
        raise ValueError(f"Label-valid mask differs from label codes: {tile_dir}")
    sensor_valid = np.isfinite(radar).all(axis=0) & np.isfinite(optical).all(axis=0)
    if not np.array_equal(validity[3], validity[1] & validity[2]) or \
       not np.array_equal(validity[4], validity[0] & validity[3]) or \
       not np.array_equal(validity[3], sensor_valid):
        raise ValueError(f"Canonical validity masks are inconsistent: {tile_dir}")
    green, nir, swir = optical[S2_BANDS.index("B3")], optical[S2_BANDS.index("B8")], optical[S2_BANDS.index("B11")]
    index_defined = (np.abs(green + nir) > EPSILON) & (np.abs(green + swir) > EPSILON)
    eligible = validity[4] & index_defined
    return MappingTile(expected["tile_id"], expected["event_id"], expected["analysis_split"],
                       label, radar, optical, eligible, transform, crs)


def optical_index(tile: MappingTile, name: str) -> np.ndarray:
    if name not in ("ndwi", "mndwi"):
        raise ValueError("Unknown optical index")
    green = tile.optical[S2_BANDS.index("B3")]
    other = tile.optical[S2_BANDS.index("B8" if name == "ndwi" else "B11")]
    result = np.full(tile.label.shape, np.nan, dtype=np.float32)
    np.divide(green - other, green + other, out=result,
              where=tile.eligible)
    return result


def channel_inputs(tile: MappingTile, modality: str) -> np.ndarray:
    if modality == "radar":
        values = (tile.radar + np.float32(20)) / np.float32(10)
    elif modality == "optical":
        indices = [S2_BANDS.index(name) for name in OPTICAL_CHANNELS]
        values = (tile.optical[indices] - np.float32(0.2)) / np.float32(0.2)
    else:
        raise ValueError("Modality must be radar or optical")
    values = np.clip(values, -3, 3)
    values[:, ~tile.eligible] = 0
    if not np.isfinite(values).all():
        raise ValueError("Non-finite model inputs after eligibility masking")
    return values.astype(np.float32, copy=False)


def logistic_features(tile: MappingTile) -> np.ndarray:
    radar = channel_inputs(tile, "radar")
    optical = channel_inputs(tile, "optical")
    ndwi = np.nan_to_num(optical_index(tile, "ndwi"), nan=0.0)
    mndwi = np.nan_to_num(optical_index(tile, "mndwi"), nan=0.0)
    return np.concatenate((radar, optical, ndwi[None], mndwi[None]), axis=0)


def confusion(label: np.ndarray, prediction: np.ndarray, eligible: np.ndarray) -> dict[str, int]:
    if label.shape != prediction.shape or label.shape != eligible.shape or eligible.dtype != bool:
        raise ValueError("Label, prediction and eligibility shapes must match")
    if np.any(eligible & ~np.isin(label, (0, 1))):
        raise ValueError("Unknown label included in evaluation")
    if np.any(eligible & ~np.isin(prediction, (0, 1))):
        raise ValueError("Prediction must be binary on eligible pixels")
    return {
        "tp": int(np.count_nonzero(eligible & (label == 1) & (prediction == 1))),
        "fp": int(np.count_nonzero(eligible & (label == 0) & (prediction == 1))),
        "fn": int(np.count_nonzero(eligible & (label == 1) & (prediction == 0))),
        "tn": int(np.count_nonzero(eligible & (label == 0) & (prediction == 0))),
        "eligible_pixels": int(np.count_nonzero(eligible)),
    }


def metrics(counts: dict[str, int]) -> dict:
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    union = tp + fp + fn
    return {
        **counts,
        "iou": tp / union if union else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def sum_confusions(rows: list[dict[str, int]]) -> dict[str, int]:
    return {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn", "eligible_pixels")}


def select_threshold(tiles: list[MappingTile], score_name: str,
                     thresholds: list[float]) -> tuple[float, list[dict]]:
    """Select on training events by event-macro IoU, then smallest threshold."""
    if not tiles or not thresholds or any(tile.split != "train" for tile in tiles):
        raise ValueError("Threshold selection requires training tiles and candidates")
    curves = []
    for threshold in thresholds:
        per_event: dict[str, list[dict]] = {}
        for tile in tiles:
            score = -tile.radar[1] if score_name == "radar_vh" else optical_index(tile, score_name)
            prediction = score >= threshold
            per_event.setdefault(tile.event_id, []).append(confusion(tile.label, prediction, tile.eligible))
        event_iou = {event: metrics(sum_confusions(rows))["iou"] for event, rows in per_event.items()}
        defined = [value for value in event_iou.values() if value is not None]
        curves.append({"threshold": float(threshold), "event_macro_iou": float(np.mean(defined)) if defined else None,
                       "event_iou": event_iou})
    viable = [row for row in curves if row["event_macro_iou"] is not None]
    if not viable:
        raise ValueError("No defined water IoU on training events")
    best = max(viable, key=lambda row: (row["event_macro_iou"], -row["threshold"]))
    return best["threshold"], curves


def stratified_training_pixels(tiles: list[MappingTile], max_per_class_per_event: int,
                               seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    if any(tile.split != "train" for tile in tiles):
        raise ValueError("Learning pixels must come only from training events")
    if max_per_class_per_event <= 0:
        raise ValueError("Positive sample cap required")
    rng = np.random.default_rng(seed)
    event_rows: dict[str, list[MappingTile]] = {}
    for tile in tiles:
        event_rows.setdefault(tile.event_id, []).append(tile)
    xs, ys, sample_counts = [], [], {}
    for event, event_tiles in sorted(event_rows.items()):
        # Bound memory per event before concatenation; sampling itself uses only train labels.
        positions = {0: [], 1: []}
        per_tile_cap = max(1, math.ceil(max_per_class_per_event / len(event_tiles)) * 2)
        for tile_index, tile in enumerate(event_tiles):
            for cls in (0, 1):
                flat = np.flatnonzero(tile.eligible & (tile.label == cls))
                if flat.size:
                    chosen = rng.choice(flat, min(flat.size, per_tile_cap), replace=False)
                    positions[cls].extend((tile_index, int(index)) for index in chosen)
        sample_counts[event] = {}
        for cls in (0, 1):
            candidates = positions[cls]
            if not candidates:
                sample_counts[event][str(cls)] = 0
                continue
            chosen = rng.choice(len(candidates), min(len(candidates), max_per_class_per_event), replace=False)
            grouped: dict[int, list[int]] = {}
            for index in chosen:
                tile_index, flat = candidates[int(index)]
                grouped.setdefault(tile_index, []).append(flat)
            for tile_index, flat in grouped.items():
                features = logistic_features(event_tiles[tile_index]).reshape(9, -1).T[flat]
                xs.append(features)
                ys.append(np.full(len(flat), cls, dtype=np.float32))
            sample_counts[event][str(cls)] = len(chosen)
    if not xs or not any((part == 1).any() for part in ys) or not any((part == 0).any() for part in ys):
        raise ValueError("Training sample needs both water and non-water")
    return np.concatenate(xs).astype(np.float64), np.concatenate(ys), sample_counts


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 150,
                 learning_rate: float = 0.15, l2: float = 0.01) -> tuple[np.ndarray, list[float]]:
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or \
       not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("Invalid binary logistic training sample")
    weights = np.zeros(x.shape[1] + 1, dtype=np.float64)
    design = np.column_stack((x, np.ones(len(x))))
    curve = []
    for _ in range(steps):
        logits = np.clip(design @ weights, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        penalty = np.r_[weights[:-1], 0.0]
        loss = float(np.mean(np.logaddexp(0, logits) - y * logits) + l2 * np.dot(weights[:-1], weights[:-1]) / 2)
        curve.append(loss)
        weights -= learning_rate * (design.T @ (probabilities - y) / len(y) + l2 * penalty)
    return weights, curve


def predict_logistic(tile: MappingTile, weights: np.ndarray) -> np.ndarray:
    features = logistic_features(tile)
    if weights.shape != (features.shape[0] + 1,):
        raise ValueError("Logistic coefficient count differs from feature contract")
    score = np.full(tile.label.shape, np.nan, dtype=np.float32)
    values = np.einsum("cij,c->ij", features, weights[:-1], optimize=True) + weights[-1]
    score[tile.eligible] = (1 / (1 + np.exp(-np.clip(values[tile.eligible], -30, 30)))).astype(np.float32)
    return score
