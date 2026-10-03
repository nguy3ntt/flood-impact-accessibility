"""Small CPU-capable U-Nets for M3 radar-only and optical-only comparisons."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mapping import MappingTile, channel_inputs


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(4, output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(4, output_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CompactUNet(nn.Module):
    def __init__(self, input_channels: int, base_channels: int = 8) -> None:
        super().__init__()
        if input_channels <= 0 or base_channels < 4 or base_channels % 4:
            raise ValueError("U-Net channel counts are invalid")
        self.enc1 = ConvBlock(input_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.middle = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.enc1(x)
        second = self.enc2(F.max_pool2d(first, 2))
        middle = self.middle(F.max_pool2d(second, 2))
        up2 = F.interpolate(middle, size=second.shape[-2:], mode="bilinear", align_corners=False)
        decoded2 = self.dec2(torch.cat((up2, second), dim=1))
        up1 = F.interpolate(decoded2, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.dec1(torch.cat((up1, first), dim=1)))


def _patch(tile: MappingTile, modality: str, rng: np.random.Generator,
           patch_size: int, prefer_water: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = tile.label.shape
    if height < patch_size or width < patch_size:
        raise ValueError("Tile is smaller than training patch")
    candidates = np.argwhere(tile.eligible & (tile.label == 1)) if prefer_water else np.empty((0, 2), dtype=int)
    if len(candidates):
        center_y, center_x = candidates[int(rng.integers(len(candidates)))]
        top = int(np.clip(center_y - rng.integers(patch_size), 0, height - patch_size))
        left = int(np.clip(center_x - rng.integers(patch_size), 0, width - patch_size))
    else:
        top = int(rng.integers(height - patch_size + 1))
        left = int(rng.integers(width - patch_size + 1))
    window = np.s_[top:top + patch_size, left:left + patch_size]
    x = channel_inputs(tile, modality)[:, window[0], window[1]].copy()
    y = (tile.label[window] == 1).astype(np.float32)
    valid = tile.eligible[window].astype(np.float32)
    return (torch.from_numpy(x[None]), torch.from_numpy(y[None, None]),
            torch.from_numpy(valid[None, None]))


def train_unet(tiles: list[MappingTile], modality: str, checkpoint: Path,
               *, seed: int, epochs: int, patches_per_tile: int,
               patch_size: int = 128, learning_rate: float = 0.001,
               base_channels: int = 8) -> tuple[CompactUNet, dict]:
    if not tiles or any(tile.split != "train" for tile in tiles):
        raise ValueError("U-Net training requires only frozen training events")
    if modality not in ("radar", "optical") or epochs <= 0 or patches_per_tile <= 0:
        raise ValueError("Invalid U-Net experiment settings")
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = 2 if modality == "radar" else 5
    model = CompactUNet(channels, base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive = sum(int(np.count_nonzero(tile.eligible & (tile.label == 1))) for tile in tiles)
    negative = sum(int(np.count_nonzero(tile.eligible & (tile.label == 0))) for tile in tiles)
    if positive == 0 or negative == 0:
        raise ValueError("Training tiles need water and non-water support")
    positive_weight = min(10.0, negative / positive)
    curves = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        order = rng.permutation(len(tiles))
        for tile_index in order:
            tile = tiles[int(tile_index)]
            for sample in range(patches_per_tile):
                x, y, valid = _patch(tile, modality, rng, patch_size, prefer_water=sample % 2 == 0)
                if not valid.any():
                    continue
                x, y, valid = x.to(device), y.to(device), valid.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                pixel_loss = F.binary_cross_entropy_with_logits(
                    logits, y, reduction="none", pos_weight=torch.tensor(positive_weight, device=device))
                loss = (pixel_loss * valid).sum() / valid.sum()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite U-Net loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("No valid training patches")
        curves.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                       "steps": len(losses), "elapsed_seconds": time.perf_counter() - start})
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "modality": modality,
                "input_channels": channels, "base_channels": base_channels,
                "seed": seed, "epochs": epochs, "patch_size": patch_size,
                "positive_weight": positive_weight}, checkpoint)
    profile = {"modality": modality, "device": str(device),
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "training_seconds": time.perf_counter() - start,
               "positive_pixels": positive, "negative_pixels": negative,
               "positive_weight": positive_weight, "curve": curves,
               "gpu_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    return model.eval(), profile


@torch.inference_mode()
def predict_unet(model: CompactUNet, tile: MappingTile, modality: str) -> np.ndarray:
    if model.output.in_channels <= 0:
        raise ValueError("Invalid trained model")
    x = torch.from_numpy(channel_inputs(tile, modality)[None])
    score = torch.sigmoid(model.eval()(x)).squeeze().numpy().astype(np.float32)
    score[~tile.eligible] = np.nan
    return score
