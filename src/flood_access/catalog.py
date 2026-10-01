"""Source-only Sen1Floods11 catalog and event-safe split validation."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
from itertools import combinations
import json
import math
from pathlib import Path
import zipfile

from shapely.geometry import Point, shape


EVENT_ALIASES = {"Mekong": "Cambodia"}
SPLIT_FILENAMES = {
    "official_train": "flood_train_data.csv",
    "official_validation": "flood_valid_data.csv",
    "official_test": "flood_test_data.csv",
    "official_bolivia": "flood_bolivia_data.csv",
}


def normalized_date(value: str) -> str:
    return datetime.strptime(value, "%Y/%m/%d").date().isoformat()


def boxes_overlap(a: list[float], b: list[float]) -> bool:
    """Positive-area bbox intersection; touching edges are not overlap."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def read_official_splits(root: Path) -> dict[str, str]:
    membership: dict[str, str] = {}
    for split, filename in SPLIT_FILENAMES.items():
        path = root / "splits/flood_handlabeled" / filename
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.reader(source):
                if len(row) != 2 or not row[0].endswith("_S1Hand.tif"):
                    raise ValueError(f"Unexpected official split row in {path}: {row}")
                chip = row[0].removesuffix("_S1Hand.tif")
                if row[1] != f"{chip}_LabelHand.tif":
                    raise ValueError(f"Mismatched official label for {chip}")
                if chip in membership:
                    raise ValueError(f"Chip appears in multiple official splits: {chip}")
                membership[chip] = split
    return membership


def build_catalog(root: Path) -> list[dict]:
    metadata = json.loads((root / "Sen1Floods11_Metadata.geojson").read_text(encoding="utf-8"))
    by_location = {f["properties"]["location"]: f for f in metadata["features"]}
    official = read_official_splits(root)
    catalog: list[dict] = []
    with zipfile.ZipFile(root / "catalog.zip") as archive:
        names = sorted(
            n for n in archive.namelist()
            if n.startswith("catalog/sen1floods11_hand_labeled_label/")
            and n.endswith(".json") and n.count("/") == 3
        )
        for name in names:
            item = json.loads(archive.read(name))
            chip = item["id"].removesuffix("_label")
            source_event = chip.split("_", 1)[0]
            location = EVENT_ALIASES.get(source_event, source_event)
            event = by_location.get(location)
            if event is None:
                raise ValueError(f"No event metadata for {chip}")
            properties = event["properties"]
            s1_date = normalized_date(properties["s1_date"])
            s2_date = normalized_date(properties["s2_date"])
            if item["properties"]["datetime"][:10] != s1_date:
                raise ValueError(f"STAC and source event dates disagree for {chip}")
            bbox = [float(v) for v in item["bbox"]]
            if len(bbox) != 4 or not all(map(math.isfinite, bbox)) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError(f"Invalid STAC bbox for {chip}")
            centroid = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if not shape(event["geometry"]).covers(centroid):
                raise ValueError(f"Chip centre outside event footprint: {chip}")
            if chip not in official:
                raise ValueError(f"STAC chip missing from official splits: {chip}")
            catalog.append({
                "tile_id": chip,
                "source_event_name": source_event,
                "event_id": f"{location.lower()}_{s1_date.replace('-', '')}",
                "event_location": location,
                "s1_date": s1_date,
                "s2_date": s2_date,
                "sensor_offset_days": (datetime.fromisoformat(s2_date) - datetime.fromisoformat(s1_date)).days,
                "bbox_epsg4326": bbox,
                "official_split": official[chip],
                "stac_label_item": name,
            })
    if set(official) != {item["tile_id"] for item in catalog}:
        raise ValueError("Official split and STAC chip sets differ")
    return sorted(catalog, key=lambda item: item["tile_id"])


def assign_event_split(catalog: list[dict], configuration: dict) -> tuple[list[dict], dict]:
    groups = configuration["event_groups"]
    assignments = {event: split for split, events in groups.items() for event in events}
    if len(assignments) != sum(map(len, groups.values())):
        raise ValueError("Event is assigned to more than one split")
    events = {item["event_location"] for item in catalog}
    if set(assignments) != events:
        raise ValueError(f"Split mismatch: missing={events-set(assignments)}, extra={set(assignments)-events}")
    result = [{**item, "analysis_split": assignments[item["event_location"]]} for item in catalog]
    overlap = []
    for a, b in combinations(result, 2):
        if boxes_overlap(a["bbox_epsg4326"], b["bbox_epsg4326"]):
            overlap.append({
                "tile_a": a["tile_id"], "tile_b": b["tile_id"],
                "event_a": a["event_id"], "event_b": b["event_id"],
                "split_a": a["analysis_split"], "split_b": b["analysis_split"],
            })
    leakage = [pair for pair in overlap if pair["split_a"] != pair["split_b"]]
    if leakage:
        raise ValueError(f"Overlapping chips cross analysis splits: {leakage[:3]}")
    summary = {
        "chip_count": len(result),
        "event_count": len(events),
        "split_chip_counts": dict(sorted(Counter(i["analysis_split"] for i in result).items())),
        "split_event_counts": {split: len(group) for split, group in groups.items()},
        "overlap_pairs": overlap,
        "cross_event_overlap_count": sum(p["event_a"] != p["event_b"] for p in overlap),
        "cross_split_overlap_count": len(leakage),
    }
    return result, summary
