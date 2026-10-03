"""Bounded immutable 2019 OpenStreetMap road extract for the Spain case."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


CONFIG = Path("configs/m6_case_v1.json")
INVENTORY = Path(".project/m6_source_inventory.json")


def query(config: dict, *, count_only: bool) -> str:
    south, west, north, east = config["osm_bbox_south_west_north_east"]
    if not (-90 < south < north < 90 and -180 < west < east < 180):
        raise ValueError("Invalid geographic bounding box")
    classes = config["osm_highway_classes"]
    if not classes or any(re.fullmatch(r"[a-z_]+", value) is None for value in classes):
        raise ValueError("Unsafe highway filter")
    pattern = "^(" + "|".join(classes) + ")$"
    header = (f'[out:json][timeout:{config["osm_request_timeout_seconds"]}]'
              f'[date:"{config["osm_as_of_utc"]}"];')
    base = f'way["highway"~"{pattern}"]({south},{west},{north},{east});'
    return header + base + ("out count;" if count_only else "out body; >; out skel qt;")


def request(config: dict, body: str):
    payload = urllib.parse.urlencode({"data": body}).encode("ascii")
    return urllib.request.Request(config["osm_endpoint"], data=payload,
                                  headers={"User-Agent": "flood-impact-accessibility-research/0.1"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download the bounded exact extract")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    body = query(config, count_only=not args.fetch)
    if not args.fetch:
        with urllib.request.urlopen(request(config, body), timeout=config["osm_request_timeout_seconds"] + 10) as source:
            response = json.load(source)
        counts = response["elements"][0]["tags"]
        print(json.dumps({"as_of_utc": config["osm_as_of_utc"], "bbox": config["osm_bbox_south_west_north_east"],
                          "selected_way_count": int(counts["ways"]), "full_query_sha256": hashlib.sha256(
                              query(config, count_only=False).encode()).hexdigest(),
                          "max_download_bytes": config["osm_max_download_bytes"]}, indent=2))
        return
    path = Path(config["osm_raw_path"])
    if path.exists() or INVENTORY.exists():
        raise FileExistsError("M6 source already exists; verify rather than overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if partial.exists():
        raise FileExistsError("Partial download exists; preserve it for inspection")
    count = 0
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request(config, body), timeout=config["osm_request_timeout_seconds"] + 20) as source, \
             partial.open("xb") as target:
            while chunk := source.read(1024 * 1024):
                count += len(chunk)
                if count > config["osm_max_download_bytes"]:
                    raise ValueError("M6 extract exceeded fixed byte cap")
                target.write(chunk)
                digest.update(chunk)
        parsed = json.loads(partial.read_text(encoding="utf-8"))
        if not parsed.get("elements") or any(element.get("type") not in {"node", "way"}
                                              for element in parsed["elements"]):
            raise ValueError("Unexpected or empty Overpass response")
        ways = [element for element in parsed["elements"] if element["type"] == "way"]
        nodes = [element for element in parsed["elements"] if element["type"] == "node"]
        if len(ways) < 100 or not nodes or any(not element.get("nodes") for element in ways):
            raise ValueError("Incomplete Overpass road extract")
        partial.replace(path)
        inventory = {"schema_version": "m6_osm_source_v1", "logical_path": path.as_posix(),
                     "source_url": config["osm_endpoint"], "as_of_utc": config["osm_as_of_utc"],
                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                     "query": body, "query_sha256": hashlib.sha256(body.encode()).hexdigest(),
                     "bytes": count, "sha256": digest.hexdigest(), "ways": len(ways), "nodes": len(nodes),
                     "licence": "OpenStreetMap contributors, ODbL 1.0",
                     "redistribution": "raw extract kept local; attribution and share-alike review before publication"}
        INVENTORY.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({key: inventory[key] for key in ("logical_path", "bytes", "sha256", "ways", "nodes")},
                         indent=2))
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


if __name__ == "__main__":
    main()
