"""Compact sensor-fusion experiments on M3's fixed observed-water support.

Injected optical gaps and brightness are stress tests, not observed cloud masks.
Unknown source pixels remain excluded from every comparison.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .mapping import MappingTile, channel_inputs
from .mapping_train import CompactUNet


METHODS = ("radar", "optical", "early", "robust_early", "late")
CONDITIONS = ("clean", "missing_optical", "block_missing", "block_brightened")


def nested_train_subset(tiles: list[MappingTile], per_event: int) -> list[MappingTile]:
    """Rank IDs without consulting label balance, making budgets nested."""
    if per_event <= 0 or not tiles or any(tile.split != "train" for tile in tiles):
        raise ValueError("Training selection requires train tiles and a positive budget")
    events: dict[str, list[MappingTile]] = {}
    for tile in tiles:
        events.setdefault(tile.event_id, []).append(tile)
    selected = []
    for event, event_tiles in sorted(events.items()):
        ranked = sorted(event_tiles, key=lambda tile: (
            hashlib.sha256(f"m4-nested-v1:{event}:{tile.tile_id}".encode()).digest(), tile.tile_id))
        selected.extend(ranked[:per_event])
    return selected


def stress_block(tile_id: str, shape: tuple[int, int], fraction: float) -> np.ndarray:
    """A reproducible contiguous square independent of image content and labels."""
    if not 0 < fraction < 1 or len(shape) != 2 or min(shape) < 2:
        raise ValueError("Invalid stress block shape or fraction")
    side = min(min(shape), max(1, round(math.sqrt(fraction) * min(shape))))
    digest = hashlib.sha256(f"m4-stress-v1:{tile_id}".encode()).digest()
    top = int.from_bytes(digest[:8], "little") % (shape[0] - side + 1)
    left = int.from_bytes(digest[8:16], "little") % (shape[1] - side + 1)
    block = np.zeros(shape, dtype=bool)
    block[top:top + side, left:left + side] = True
    return block


def optical_observation(tile: MappingTile, condition: str, *, block_fraction: float,
                        brightening_mix: float) -> tuple[np.ndarray, np.ndarray]:
    if condition not in CONDITIONS or not 0 <= brightening_mix <= 1:
        raise ValueError("Unknown optical stress condition")
    optical = channel_inputs(tile, "optical").copy()
    present = np.ones(tile.label.shape, dtype=bool)
    if condition == "missing_optical":
        present[:] = False
    elif condition in ("block_missing", "block_brightened"):
        block = stress_block(tile.tile_id, tile.label.shape, block_fraction)
        if condition == "block_missing":
            present[block] = False
        else:
            # A bounded, explicitly synthetic brightness perturbation in the
            # model's fixed normalised optical scale, not a physical cloud model.
            optical[:, block] = np.clip((1 - brightening_mix) * optical[:, block]
                                        + brightening_mix * 2.0, -3, 3)
    optical[:, ~present] = 0
    return optical, present


def model_inputs(tile: MappingTile, method: str, condition: str = "clean", *,
                 block_fraction: float = 0.25, brightening_mix: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    if method not in METHODS[:-1]:
        raise ValueError("Method has no direct model inputs")
    radar = channel_inputs(tile, "radar")
    optical, present = optical_observation(tile, condition, block_fraction=block_fraction,
                                            brightening_mix=brightening_mix)
    if method == "radar":
        values = radar
    elif method == "optical":
        values = optical
    else:
        values = np.concatenate((radar, optical, present[None].astype(np.float32)), axis=0)
    values = values.astype(np.float32, copy=False)
    values[:, ~tile.eligible] = 0
    if not np.isfinite(values).all():
        raise ValueError("Non-finite fusion inputs")
    return values, present


def late_fusion(radar: np.ndarray, optical: np.ndarray, present: np.ndarray,
                *, radar_weight: float) -> np.ndarray:
    if radar.shape != optical.shape or radar.shape != present.shape or not 0 <= radar_weight <= 1:
        raise ValueError("Late-fusion inputs differ")
    result = radar.copy()
    both = present & np.isfinite(radar) & np.isfinite(optical)
    result[both] = radar_weight * radar[both] + (1 - radar_weight) * optical[both]
    return result


def _patch(tile: MappingTile, method: str, rng: np.random.Generator, size: int,
           prefer_water: bool, *, full_dropout_probability: float,
           block_dropout_probability: float, block_fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = tile.label.shape
    if min(height, width) < size:
        raise ValueError("Training patch exceeds tile")
    candidates = np.argwhere(tile.eligible & (tile.label == 1)) if prefer_water else np.empty((0, 2), dtype=int)
    if len(candidates):
        y, x = candidates[int(rng.integers(len(candidates)))]
        top = int(np.clip(y - rng.integers(size), 0, height - size))
        left = int(np.clip(x - rng.integers(size), 0, width - size))
    else:
        top = int(rng.integers(height - size + 1))
        left = int(rng.integers(width - size + 1))
    window = np.s_[top:top + size, left:left + size]
    values, _ = model_inputs(tile, method)
    values = values[:, window[0], window[1]].copy()
    if method == "robust_early":
        draw = rng.random()
        if draw < full_dropout_probability:
            values[2:7] = 0
            values[7] = 0
        elif draw < full_dropout_probability + block_dropout_probability:
            block = stress_block(f"{tile.tile_id}:{top}:{left}", (size, size), block_fraction)
            values[2:7, block] = 0
            values[7, block] = 0
    target = (tile.label[window] == 1).astype(np.float32)
    valid = tile.eligible[window].astype(np.float32)
    return (torch.from_numpy(values[None]), torch.from_numpy(target[None, None]),
            torch.from_numpy(valid[None, None]))


def train_model(tiles: list[MappingTile], method: str, checkpoint: Path, *, seed: int,
                epochs: int, patches_per_tile: int, patch_size: int,
                base_channels: int, learning_rate: float,
                full_dropout_probability: float, block_dropout_probability: float,
                block_fraction: float) -> tuple[CompactUNet, dict]:
    if method not in METHODS[:-1] or not tiles or any(tile.split != "train" for tile in tiles):
        raise ValueError("Fusion training requires only training events")
    if epochs <= 0 or patches_per_tile <= 0 or learning_rate <= 0 or not 0 <= full_dropout_probability <= 1 or \
       not 0 <= block_dropout_probability <= 1 or full_dropout_probability + block_dropout_probability > 1:
        raise ValueError("Invalid fusion training configuration")
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = {"radar": 2, "optical": 5, "early": 8, "robust_early": 8}[method]
    model = CompactUNet(channels, base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positives = sum(int(np.count_nonzero(tile.eligible & (tile.label == 1))) for tile in tiles)
    negatives = sum(int(np.count_nonzero(tile.eligible & (tile.label == 0))) for tile in tiles)
    if positives == 0 or negatives == 0:
        raise ValueError("Training set needs both classes")
    positive_weight = min(10.0, negatives / positives)
    weight = torch.tensor(positive_weight, device=device)
    curve = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for index in rng.permutation(len(tiles)):
            tile = tiles[int(index)]
            for sample in range(patches_per_tile):
                x, y, valid = _patch(tile, method, rng, patch_size, sample % 2 == 0,
                                     full_dropout_probability=full_dropout_probability,
                                     block_dropout_probability=block_dropout_probability,
                                     block_fraction=block_fraction)
                if not valid.any():
                    continue
                x, y, valid = x.to(device), y.to(device), valid.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss_pixels = F.binary_cross_entropy_with_logits(model(x), y, reduction="none", pos_weight=weight)
                loss = (loss_pixels * valid).sum() / valid.sum()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite fusion training loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("No valid fusion training patches")
        curve.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "steps": len(losses)})
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "method": method,
                "input_channels": channels, "base_channels": base_channels, "seed": seed,
                "epochs": epochs, "positive_weight": positive_weight}, checkpoint)
    return model.eval(), {"method": method, "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                          "training_seconds": time.perf_counter() - start, "positive_pixels": positives,
                          "negative_pixels": negatives, "curve": curve,
                          "gpu_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}


@torch.inference_mode()
def predict_model(model: CompactUNet, tile: MappingTile, method: str, condition: str, *,
                  block_fraction: float, brightening_mix: float) -> tuple[np.ndarray, np.ndarray]:
    values, present = model_inputs(tile, method, condition, block_fraction=block_fraction,
                                    brightening_mix=brightening_mix)
    score = torch.sigmoid(model.eval()(torch.from_numpy(values[None]))).squeeze().numpy().astype(np.float32)
    score[~tile.eligible] = np.nan
    if not np.isfinite(score[tile.eligible]).all():
        raise ValueError("Non-finite fusion prediction on eligible pixels")
    return score, present
