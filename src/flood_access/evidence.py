"""M5 empirical calibration, quality abstention and coherent scenario methods.

The target is the Sen1Floods11 observed-water label. Probabilities are empirical
development scores and have no coverage guarantee under event/domain shift.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .mapping import MappingTile, optical_index
from .raster_pipeline import S2_BANDS


def hashed_event_rank(events: list[str], salt: str) -> list[str]:
    if len(set(events)) != len(events) or not salt:
        raise ValueError("Event rank requires unique IDs and a salt")
    return sorted(events, key=lambda event: (hashlib.sha256(f"{salt}:{event}".encode()).digest(), event))


def calibration_points(tile: MappingTile, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample one eligible pixel per spatial block without consulting its label."""
    if block_size <= 0 or tile.split != "train":
        raise ValueError("Calibration sampling requires a training event and positive block size")
    score = optical_index(tile, "ndwi")
    xs, ys = [], []
    height, width = tile.label.shape
    for top in range(0, height, block_size):
        for left in range(0, width, block_size):
            window = tile.eligible[top:top + block_size, left:left + block_size]
            positions = np.flatnonzero(window)
            if not len(positions):
                continue
            digest = hashlib.sha256(f"m5-sample-v1:{tile.tile_id}:{top}:{left}".encode()).digest()
            picked = int(positions[int.from_bytes(digest[:8], "little") % len(positions)])
            local_y, local_x = np.unravel_index(picked, window.shape)
            y, x = top + local_y, left + local_x
            xs.append(float(np.clip(score[y, x], -1, 1)))
            ys.append(int(tile.label[y, x]))
    if not xs:
        raise ValueError("Calibration tile has no eligible block")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def fit_monotone_logistic(samples: dict[str, tuple[np.ndarray, np.ndarray]], *,
                          steps: int, learning_rate: float, slope_l2: float) -> dict:
    """Equalise each event's total loss weight; do not treat pixels as events."""
    if len(samples) < 2 or steps <= 0 or learning_rate <= 0 or slope_l2 < 0:
        raise ValueError("Calibration requires multiple events and valid optimisation settings")
    events = sorted(samples)
    xs, ys, weights = [], [], []
    for event in events:
        x, y = samples[event]
        if not len(x) or x.shape != y.shape or not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
            raise ValueError("Invalid event calibration sample")
        xs.append(x)
        ys.append(y)
        weights.append(np.full(len(x), 1 / (len(events) * len(x)), dtype=np.float64))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration needs both observed-water classes")
    slope, intercept = 2.0, 0.0
    curve = []
    for iteration in range(steps):
        z = np.clip(slope * x + intercept, -30, 30)
        probability = 1 / (1 + np.exp(-z))
        slope_gradient = float(np.dot(w, (probability - y) * x) + slope_l2 * slope)
        intercept_gradient = float(np.dot(w, probability - y))
        slope = max(0.0, slope - learning_rate * slope_gradient)
        intercept -= learning_rate * intercept_gradient
        if iteration == 0 or (iteration + 1) % 50 == 0 or iteration + 1 == steps:
            current = np.clip(slope * x + intercept, -30, 30)
            loss = float(np.dot(w, np.logaddexp(0, current) - y * current)
                         + 0.5 * slope_l2 * slope * slope)
            curve.append({"step": iteration + 1, "weighted_log_loss": loss})
    prevalence = float(np.dot(w, y))
    if not np.isfinite((slope, intercept, prevalence)).all() or not 0 < prevalence < 1:
        raise ValueError("Invalid fitted calibrator")
    return {"slope": slope, "intercept": intercept, "event_equal_sample_prevalence": prevalence,
            "events": events, "samples_per_event": {event: len(samples[event][0]) for event in events},
            "curve": curve, "method": "event_equal_monotone_logistic_on_fixed_ndwi"}


def calibrated_probability(tile: MappingTile, fit: dict) -> np.ndarray:
    score = optical_index(tile, "ndwi")
    result = np.full(tile.label.shape, np.nan, dtype=np.float32)
    value = np.clip(fit["slope"] * np.clip(score[tile.eligible], -1, 1) + fit["intercept"], -30, 30)
    result[tile.eligible] = (1 / (1 + np.exp(-value))).astype(np.float32)
    return result


def unit_scaled_ndwi(tile: MappingTile) -> np.ndarray:
    """Fixed score scaling for comparison, not a fitted probability model."""
    score = optical_index(tile, "ndwi")
    result = np.full(tile.label.shape, np.nan, dtype=np.float32)
    result[tile.eligible] = np.clip((score[tile.eligible] + 1) / 2, 0, 1)
    return result


def quality_flags(tile: MappingTile, *, weak_denominator_below: float,
                  radiometric_extreme_above: float) -> np.ndarray:
    """Bit 1: weak index denominator; bit 2: bright TOA extreme; 255: unknown."""
    if weak_denominator_below <= 0 or radiometric_extreme_above <= 0:
        raise ValueError("Quality thresholds must be positive")
    green = tile.optical[S2_BANDS.index("B3")]
    nir = tile.optical[S2_BANDS.index("B8")]
    swir = tile.optical[S2_BANDS.index("B11")]
    flags = np.full(tile.label.shape, 255, dtype=np.uint8)
    weak = (np.minimum(np.abs(green + nir), np.abs(green + swir)) < weak_denominator_below)
    bright = (np.maximum.reduce((green, nir, swir)) > radiometric_extreme_above)
    flags[tile.eligible] = (weak[tile.eligible].astype(np.uint8)
                            + 2 * bright[tile.eligible].astype(np.uint8))
    return flags


def retained_mask(tile: MappingTile, probability: np.ndarray, flags: np.ndarray, *,
                  ambiguity_margin: float, require_quality: bool) -> np.ndarray:
    if probability.shape != tile.label.shape or flags.shape != tile.label.shape or \
       not 0 <= ambiguity_margin < 0.5:
        raise ValueError("Invalid abstention inputs")
    retained = tile.eligible & np.isfinite(probability) & (np.abs(probability - 0.5) > ambiguity_margin)
    if require_quality:
        retained &= flags == 0
    return retained


def reliability(label: np.ndarray, probability: np.ndarray, support: np.ndarray,
                bins: int) -> dict:
    if label.shape != probability.shape or support.shape != label.shape or support.dtype != bool or bins < 2:
        raise ValueError("Reliability arrays or bins differ")
    if np.any(support & (~np.isin(label, (0, 1)) | ~np.isfinite(probability))) or \
       np.any(support & ((probability < 0) | (probability > 1))):
        raise ValueError("Invalid label or probability in reliability support")
    y = label[support].astype(np.float64)
    p = probability[support].astype(np.float64)
    if not len(y):
        return {"pixels": 0, "prevalence": None, "brier": None, "log_loss": None,
                "ece": None, "bins": []}
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    index = np.minimum((p * bins).astype(int), bins - 1)
    counts = np.bincount(index, minlength=bins)
    sum_p = np.bincount(index, weights=p, minlength=bins)
    sum_y = np.bincount(index, weights=y, minlength=bins)
    rows = [{"bin": i, "lower": i / bins, "upper": (i + 1) / bins,
             "pixels": int(counts[i]),
             "mean_probability": float(sum_p[i] / counts[i]) if counts[i] else None,
             "observed_fraction": float(sum_y[i] / counts[i]) if counts[i] else None}
            for i in range(bins)]
    ece = float(sum(counts[i] / len(y) * abs(rows[i]["mean_probability"] - rows[i]["observed_fraction"])
                    for i in range(bins) if counts[i]))
    return {"pixels": len(y), "prevalence": float(np.mean(y)),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))),
            "ece": ece, "bins": rows}


def spatial_scenarios(probability: np.ndarray, eligible: np.ndarray, tile_id: str, *,
                      block_pixels: int, thresholds: list[float], offset_seeds: list[int],
                      maximum_offset: float) -> tuple[np.ndarray, list[str]]:
    """Pixel masks with spatially shared block offsets, not posterior draws."""
    if probability.shape != eligible.shape or eligible.dtype != bool or block_pixels <= 0 or \
       not thresholds or not all(0 < value < 1 for value in thresholds) or \
       not 0 <= maximum_offset < 0.5 or len(set(offset_seeds)) != len(offset_seeds):
        raise ValueError("Invalid spatial-scenario specification")
    if np.any(eligible & ~np.isfinite(probability)) or np.any(~eligible & np.isfinite(probability)):
        raise ValueError("Scenario support differs from calibrated probability")
    names = [f"threshold_{value:.2f}" for value in thresholds] + [f"offset_seed_{seed}" for seed in offset_seeds]
    bands = np.full((len(names), *eligible.shape), 255, dtype=np.uint8)
    height, width = eligible.shape
    for top in range(0, height, block_pixels):
        for left in range(0, width, block_pixels):
            window = np.s_[top:top + block_pixels, left:left + block_pixels]
            valid = eligible[window]
            if not valid.any():
                continue
            local_probability = probability[window][valid]
            values = [local_probability >= threshold for threshold in thresholds]
            for seed in offset_seeds:
                digest = hashlib.sha256(f"m5-block-v1:{tile_id}:{top}:{left}:{seed}".encode()).digest()
                unit = int.from_bytes(digest[:8], "little") / (2 ** 64 - 1)
                delta = (2 * unit - 1) * maximum_offset
                values.append(np.clip(local_probability + delta, 0, 1) >= 0.5)
            for band, value in enumerate(values):
                band_view = bands[band][window]
                band_view[valid] = value.astype(np.uint8)
    return bands, names


def scenario_summary(bands: np.ndarray, eligible: np.ndarray, names: list[str]) -> dict:
    if bands.ndim != 3 or bands.shape[1:] != eligible.shape or len(names) != len(bands) or \
       np.any(bands[:, ~eligible] != 255) or np.any(~np.isin(bands[:, eligible], (0, 1))):
        raise ValueError("Scenario unknown/support contract differs")
    count = int(np.count_nonzero(eligible))
    water = [int(np.count_nonzero(band[eligible] == 1)) for band in bands]
    min_mask = bands[0, eligible] == 1
    max_mask = bands[min(2, len(bands) - 1), eligible] == 1
    intersection = int(np.count_nonzero(min_mask & max_mask))
    union = int(np.count_nonzero(min_mask | max_mask))
    return {"eligible_pixels": count,
            "water_pixels_by_scenario": dict(zip(names, water)),
            "water_fraction_by_scenario": {name: value / count if count else None
                                           for name, value in zip(names, water)},
            "fixed_extremes_jaccard": intersection / union if union else None,
            "unanimous_fraction": float(np.mean(np.all(bands[:, eligible] == bands[0, eligible], axis=0)))
            if count else None,
            "offset_member_frequency_mean": float(np.mean(bands[3:, eligible] == 1)) if count and len(bands) > 3 else None,
            "frequency_interpretation": "Fraction of designed block-offset masks, not a water probability."}
