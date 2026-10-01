"""Local visual QA panels; these are diagnostic previews, not public results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
from rasterio.features import rasterize


def _stretch(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = values[valid & np.isfinite(values)]
    if selected.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(selected, (2, 98))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0, 1)
    result = (scaled * 255).astype(np.uint8)
    result[~valid] = 0
    return result


def _rgb(optical: np.ndarray, valid: np.ndarray) -> Image.Image:
    channels = [_stretch(optical[band], valid) for band in (3, 2, 1)]
    return Image.fromarray(np.stack(channels, axis=-1), mode="RGB")


def _label_image(label: np.ndarray) -> Image.Image:
    colors = np.zeros((*label.shape, 3), dtype=np.uint8)
    colors[label == -1] = (105, 105, 105)
    colors[label == 0] = (48, 52, 42)
    colors[label == 1] = (35, 135, 225)
    return Image.fromarray(colors, mode="RGB")


def _case_overlay(base: Image.Image, transform, vector_dir: Path) -> Image.Image:
    result = base.convert("RGBA")
    shape_hw = (base.height, base.width)
    trace = json.loads((vector_dir / "flood_traces.geojson").read_text(encoding="utf-8"))["features"]
    roads = json.loads((vector_dir / "roads.geojson").read_text(encoding="utf-8"))["features"]
    origins = json.loads((vector_dir / "origins.geojson").read_text(encoding="utf-8"))["features"]
    trace_mask = rasterize(((f["geometry"], 1) for f in trace), out_shape=shape_hw, transform=transform, dtype="uint8")
    fill = np.zeros((*shape_hw, 4), dtype=np.uint8)
    fill[trace_mask == 1] = (0, 190, 230, 85)
    result = Image.alpha_composite(result, Image.fromarray(fill, mode="RGBA"))
    for grade, color in (("Possibly damaged", (255, 180, 30, 255)), ("Damaged", (230, 30, 45, 255))):
        selected = [f for f in roads if f["properties"]["damage_grade"] == grade]
        mask = rasterize(((f["geometry"], 1) for f in selected), out_shape=shape_hw,
                         transform=transform, all_touched=True, dtype="uint8")
        ink = np.zeros((*shape_hw, 4), dtype=np.uint8)
        ink[mask == 1] = color
        result = Image.alpha_composite(result, Image.fromarray(ink, mode="RGBA"))
    draw = ImageDraw.Draw(result)
    for feature in origins:
        x, y = feature["geometry"]["coordinates"]
        col, row = ~transform * (x, y)
        if 0 <= col < base.width and 0 <= row < base.height:
            draw.ellipse((col - 3, row - 3, col + 3, row + 3), fill=(255, 245, 50, 255), outline=(0, 0, 0, 255))
    return result.convert("RGB")


def render_tile_panel(tile_dir: Path, output: Path, vector_dir: Path | None = None) -> None:
    with rasterio.open(tile_dir / "label.tif") as source:
        label, transform = source.read(1), source.transform
    with rasterio.open(tile_dir / "radar_db.tif") as source:
        radar = source.read(1)
    with rasterio.open(tile_dir / "optical_toa.tif") as source:
        optical = source.read()
    with rasterio.open(tile_dir / "validity.tif") as source:
        validity = source.read()
    rgb = _rgb(optical, validity[2] == 1)
    radar_gray = _stretch(radar, validity[1] == 1)
    panels = [rgb, Image.fromarray(np.stack((radar_gray,) * 3, axis=-1), mode="RGB"), _label_image(label)]
    titles = ["S2 TOA RGB (visual stretch)", "S1 VV dB (visual stretch)", "Label: water / non-water / unknown"]
    if vector_dir is not None:
        panels.append(_case_overlay(rgb, transform, vector_dir))
        titles.append("CEMS trace + road damage + origins")
    gap, heading, footer = 12, 48, 64
    image = Image.new("RGB", (len(panels) * 512 + (len(panels) + 1) * gap, 512 + heading + footer), (240, 241, 238))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = gap + index * (512 + gap)
        image.paste(panel, (x, heading))
        draw.text((x, 12), title, fill=(20, 28, 32), font=font)
    draw.text((gap, heading + 520), "Gray = unknown label; black = sensor nodata. Optical cloud status is unknown (no source QA).", fill=(25, 35, 40), font=font)
    if vector_dir is not None:
        draw.text((gap, heading + 540), "Cyan = CEMS flood trace; red/orange = interpreted road damage; yellow = origin candidate. None proves closure or safe travel.", fill=(25, 35, 40), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=False)
