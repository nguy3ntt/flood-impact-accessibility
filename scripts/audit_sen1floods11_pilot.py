"""Inspect real Sen1Floods11 pilot bytes without changing source files."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path("data/raw/sen1floods11/v1.1")
REPORT = Path(".project/m1_pilot_audit.json")
LAYERS = ("LabelHand", "S1Hand", "S2Hand")


def raster_summary(path: Path) -> dict:
    with rasterio.open(path) as source:
        data = source.read()
        result = {
            "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "driver": source.driver,
            "width": source.width,
            "height": source.height,
            "bands": source.count,
            "dtypes": list(source.dtypes),
            "band_descriptions": list(source.descriptions),
            "crs": str(source.crs),
            "transform": list(source.transform)[:6],
            "bounds": list(source.bounds),
            "nodata": (
                "NaN" if source.nodata is not None and np.isnan(source.nodata) else source.nodata
            ),
            "mask_valid_fraction": float(np.mean(source.dataset_mask() > 0)),
        }
        if "LabelHand" in path.name:
            values, counts = np.unique(data, return_counts=True)
            result["label_counts"] = {str(int(v)): int(c) for v, c in zip(values, counts)}
            result["label_codes_valid"] = set(result["label_counts"]) <= {"-1", "0", "1"}
        else:
            result["band_stats"] = []
            for band in data:
                finite = band[np.isfinite(band)]
                result["band_stats"].append({
                    "min": float(np.min(finite)) if finite.size else None,
                    "median": float(np.median(finite)) if finite.size else None,
                    "max": float(np.max(finite)) if finite.size else None,
                    "zero_fraction": float(np.mean(band == 0)),
                    "nonfinite_count": int(band.size - finite.size),
                })
        return result


def official_splits() -> dict:
    directory = ROOT / "splits/flood_handlabeled"
    rows = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as source:
            pairs = list(csv.reader(source))
        if not all(len(pair) == 2 for pair in pairs):
            raise ValueError(f"Unexpected split row shape in {path}")
        rows[path.stem] = pairs
    sets = {name: {tuple(pair) for pair in pairs} for name, pairs in rows.items()}
    return {
        "counts": {name: len(pairs) for name, pairs in rows.items()},
        "event_counts": {
            name: dict(sorted(Counter(pair[0].split("_", 1)[0] for pair in pairs).items()))
            for name, pairs in rows.items()
        },
        "duplicate_rows_within_split": {
            name: len(pairs) - len(sets[name]) for name, pairs in rows.items()
        },
        "overlap_rows_between_splits": {
            f"{a}__{b}": len(sets[a] & sets[b])
            for a in sets
            for b in sets
            if a < b
        },
    }


def main() -> None:
    metadata = json.loads((ROOT / "Sen1Floods11_Metadata.geojson").read_text(encoding="utf-8"))
    event_metadata = {feature["properties"]["location"]: feature["properties"] for feature in metadata["features"]}
    base = ROOT / "data/flood_events/HandLabeled"
    chips = {}
    for label in sorted((base / "LabelHand").glob("*_LabelHand.tif")):
        chip = label.name.removesuffix("_LabelHand.tif")
        layers = {layer: raster_summary(base / layer / f"{chip}_{layer}.tif") for layer in LAYERS}
        reference = layers["LabelHand"]
        layers_align = all(
            value["width"] == reference["width"]
            and value["height"] == reference["height"]
            and value["crs"] == reference["crs"]
            and np.allclose(value["transform"], reference["transform"], rtol=0, atol=1e-10)
            for value in layers.values()
        )
        event = chip.split("_", 1)[0]
        event_info = event_metadata.get(event)
        chips[chip] = {
            "event": event,
            "event_metadata": event_info,
            "same_grid": layers_align,
            "layers": layers,
        }
    report = {"dataset_version": "v1.1", "official_splits": official_splits(), "chips": chips}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": REPORT.as_posix(),
        "chips": len(chips),
        "same_grid": {key: value["same_grid"] for key, value in chips.items()},
        "label_counts": {key: value["layers"]["LabelHand"]["label_counts"] for key, value in chips.items()},
        "official_split_counts": report["official_splits"]["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
