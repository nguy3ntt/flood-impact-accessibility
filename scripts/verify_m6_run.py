"""Verify historical source, graph, case rasters and every M6 scenario result."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re

import numpy as np
import rasterio

from flood_access.raster_pipeline import sha256
from run_m6_case import (CONFIG_PATH, derive, fingerprint, references)


def _rows(path: Path, expected) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for observed, reference in zip_longest(source, expected, fillvalue=None):
            if observed is None or reference is None or json.loads(observed) != reference:
                raise ValueError(f"M6 compressed table replay differs at row {count}: {path}")
            count += 1
    return count


def _raster(path: Path, tile, expected: np.ndarray, descriptions: list[str], nodata) -> None:
    array = expected[None] if expected.ndim == 2 else expected
    with rasterio.open(path) as source:
        if source.shape != tile.label.shape or source.count != len(array) or source.crs != tile.crs or \
           source.transform != tile.transform or source.descriptions != tuple(descriptions) or \
           not ((np.isnan(source.nodata) and np.isnan(nodata)) if isinstance(nodata, float) and np.isnan(nodata)
                else source.nodata == nodata):
            raise ValueError(f"M6 case raster contract differs: {path}")
        observed = source.read()
    if observed.dtype != array.dtype or not np.array_equal(observed, array, equal_nan=True):
        raise ValueError(f"M6 case raster values differ: {path}")


def verify(run_id: str) -> dict:
    if re.fullmatch(r"M6-[0-9a-f]{12}", run_id) is None:
        raise ValueError("Invalid M6 run ID")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    m2, m5, m5_config, fit = references(config)
    stamp = fingerprint(config, m2, m5)
    expected_id = "M6-" + hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:12]
    if run_id != expected_id:
        raise ValueError("M6 fingerprint differs from source/code/config")
    root, report = Path("runs") / run_id, Path("reports/m6") / run_id
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id or manifest["status"] != "complete" or \
       status["run_id"] != run_id or status["status"] != "complete" or \
       status["manifest_sha256"] != sha256(manifest_path) or manifest["fingerprint"] != stamp or \
       manifest["m2_run_id"] != config["m2_run_id"] or manifest["m5_run_id"] != config["m5_run_id"] or \
       manifest["source_sha256"] != {"osm": sha256(Path(config["osm_raw_path"])),
                                     "m2_manifest": m2["manifest_sha256"],
                                     "m5_manifest": m5["manifest_sha256"]}:
        raise ValueError("M6 manifest/status provenance differs")
    files = manifest["output_sha256"]
    actual = {path.as_posix() for folder in (root, report) for path in folder.rglob("*")
              if path.is_file() and path not in (root / "status.json", manifest_path)}
    if actual != set(files):
        raise ValueError("M6 output set differs")
    for name, digest in files.items():
        path = Path(name)
        if ".." in path.parts or not (path.is_relative_to(root) or path.is_relative_to(report)) or \
           sha256(path) != digest:
            raise ValueError(f"M6 output changed: {name}")
    data = derive(config, m5_config, fit)
    nodes = data["nodes"]
    if _rows(root / "graph_nodes.jsonl.gz", ({"node_id": node, "x_m": pos[0], "y_m": pos[1],
                                             "lon": pos[2], "lat": pos[3]}
                                            for node, pos in sorted(nodes.items()))) != len(nodes):
        raise ValueError("M6 node count differs")
    if _rows(root / "graph_edges.jsonl.gz", data["edges"]) != len(data["edges"]) or \
       _rows(root / "road_evidence.jsonl.gz", (data["evidence"][key]
                                                for key in sorted(data["evidence"]))) != len(data["evidence"]):
        raise ValueError("M6 graph/evidence row count differs")
    for name, expected in (("graph_quality.json", data["graph_quality"]),
                           ("snaps.json", {"hospitals": data["hospitals"], "origins": data["origins"]}),
                           ("scenario_results.json", data["results"]),
                           ("summary.json", data["summary"])):
        if json.loads((root / name).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"M6 derived summary differs: {name}")
    tile = data["tile"]
    raster_root = root / "case_rasters"
    _raster(raster_root / "source_support.tif", tile, tile.eligible.astype(np.uint8),
            ["1_sensor_valid_index_defined_0_unknown"], 0)
    _raster(raster_root / "probability.tif", tile, data["probability"],
            ["empirical_observed_water_score_Spain_transfer_unvalidated"], float("nan"))
    _raster(raster_root / "quality_flags.tif", tile, data["flags"],
            ["bit1_weak_denominator_bit2_bright_TOA_255_unknown"], 255)
    state = np.full(tile.label.shape, 255, dtype=np.uint8)
    state[tile.eligible] = 0
    state[data["retained"]] = 1
    _raster(raster_root / "retained.tif", tile, state,
            ["1_retained_0_abstained_255_source_unknown"], 255)
    _raster(raster_root / "designed_cases.tif", tile, data["cases"], data["names"], 255)
    return {"run_id": run_id, "status": "verified", "output_files": len(files),
            "graph_nodes": len(nodes), "directed_edges": len(data["edges"]),
            "road_evidence_segments": len(data["evidence"]), "scenario_rows": len(data["results"]),
            "case_rasters_replayed": 5, "manifest_sha256": sha256(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(verify(args.run_id), indent=2))


if __name__ == "__main__":
    main()
