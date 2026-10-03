"""Build and validate immutable, local-only M7 analyst bundles.

The serving bundle is derived from verified M6 outputs. It contains no source
imagery, benchmark labels, credentials, or public release media.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib


SCHEMA = "m7_analyst_bundle_v1"
CODE_PATHS = (Path("src/flood_access/analysis_bundle.py"),
              Path("src/flood_access/serving.py"), Path("scripts/build_m7_bundle.py"),
              Path("web/static/app.js"), Path("web/static/app.css"),
              Path("web/static/index.html"),
              *(Path("web/static/fonts") / name for name in (
                  "ibm-plex-sans-condensed-latin-400-normal.woff2",
                  "ibm-plex-sans-condensed-latin-600-normal.woff2",
                  "ibm-plex-sans-condensed-latin-700-normal.woff2",
                  "ibm-plex-mono-latin-400-normal.woff2",
                  "ibm-plex-mono-latin-600-normal.woff2",
                  "OFL-IBM-Plex-Sans-Condensed.txt",
                  "OFL-IBM-Plex-Mono.txt")))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    part.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_gzip_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def closure_filename(key: tuple[str, str, str]) -> str:
    marker = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()[:16]
    return f"closures/{marker}.json.gz"


def write_closure(path: Path, ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(sorted(ids), separators=(",", ":")).encode()
    with path.open("wb") as target, gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as output:
        output.write(data)


def _png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    """Small dependency-free PNG writer for the clean-clone synthetic fixture."""
    if len(rgba) != width * height * 4:
        raise ValueError("RGBA dimensions differ")
    payload = b"".join(b"\x00" + rgba[row * width * 4:(row + 1) * width * 4]
                       for row in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(payload, 9)) + chunk(b"IEND", b""))


def _bundle_id(mode: str, source_hash: str) -> str:
    code = {p.as_posix(): digest(p) for p in CODE_PATHS}
    payload = json.dumps({"mode": mode, "source_sha256": source_hash, "code": code},
                         sort_keys=True, separators=(",", ":"))
    return ("M7-" if mode == "real" else "M7S-") + hashlib.sha256(payload.encode()).hexdigest()[:12]


def _finish(root: Path, mode: str, source_run: str | None, source_hash: str) -> dict:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in
                   {"manifest.json", "status.json"})
    manifest = {"schema_version": SCHEMA, "run_id": root.name, "status": "complete",
                "mode": mode, "source_run_id": source_run, "source_manifest_sha256": source_hash,
                "code_sha256": {p.as_posix(): digest(p) for p in CODE_PATHS},
                "output_sha256": {p.relative_to(root).as_posix(): digest(p) for p in files}}
    write_json(root / "manifest.json", manifest)
    write_json(root / "status.json", {"run_id": root.name, "status": "complete",
                                      "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                      "manifest_sha256": digest(root / "manifest.json")})
    return {"run_id": root.name, "status": "complete", "mode": mode,
            "output_files": len(files), "manifest_sha256": digest(root / "manifest.json")}


def _origin_rows(snaps: list[dict], summary: dict, nodes: dict[int, tuple]) -> list[dict]:
    baseline = {row["origin_id"]: row for row in summary["baseline"]}
    stability = {row["origin_id"]: row for row in summary["designed_scenario_stability"]}
    result = []
    for snap in snaps:
        if snap["status"] != "snapped":
            continue
        item = dict(snap)
        item["lon"], item["lat"] = nodes[snap["nearest_node_id"]]
        item["baseline"] = baseline[snap["feature_id"]]
        item["designed_stability"] = stability[snap["feature_id"]]
        result.append(item)
    return result


def build_real(m6_root: Path, output_root: Path = Path("runs")) -> dict:
    """Call only after verify_m6_run.verify has accepted the exact source run."""
    import numpy as np
    from PIL import Image
    import rasterio

    source_manifest = m6_root / "run_manifest.json"
    source_hash = digest(source_manifest)
    root = output_root / _bundle_id("real", source_hash)
    if root.exists():
        raise FileExistsError(f"M7 bundle {root.name} already exists; preserve and verify it")
    root.mkdir(parents=True)
    write_json(root / "status.json", {"run_id": root.name, "status": "running",
                                      "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        summary = read_json(m6_root / "summary.json")
        quality = read_json(m6_root / "graph_quality.json")
        snaps = read_json(m6_root / "snaps.json")
        scenarios = read_json(m6_root / "scenario_results.json")
        nodes = {row["node_id"]: (row["lon"], row["lat"])
                 for row in read_gzip_rows(m6_root / "graph_nodes.jsonl.gz")}
        roads = {}
        for edge in read_gzip_rows(m6_root / "graph_edges.jsonl.gz"):
            segment_id = edge["segment_id"]
            if segment_id not in roads:
                a, b = nodes[edge["source"]], nodes[edge["target"]]
                roads[segment_id] = [segment_id, a[0], a[1], b[0], b[1], edge["highway"],
                                     edge["length_m"], edge["bridge"], edge["tunnel"]]
        evidence = {row["segment_id"]: row
                    for row in read_gzip_rows(m6_root / "road_evidence.jsonl.gz")}
        if len(roads) != quality["physical_segments"] or len(evidence) != quality["road_evidence"]["segments_intersecting_pilot"]:
            raise ValueError("M6 road/evidence row counts differ from graph quality")
        from flood_access.case_scenarios import closed_segments
        segment_contract = {row[0]: {"length_m": row[6], "bridge": row[7], "tunnel": row[8]}
                            for row in roads.values()}
        minimum = read_json(Path("configs/m6_case_v1.json"))["minimum_exposed_supported_m"]
        closure_files = []
        for row in scenarios:
            closed, reasons = closed_segments(segment_contract, evidence, row["scenario_id"],
                                              policy=row["unknown_or_evidence_policy"],
                                              bridge_policy=row["bridge_policy"], minimum_exposed_m=minimum)
            if reasons != row["closure"]:
                raise ValueError("M6 scenario closure count/reasons differ")
            key = (row["scenario_id"], row["unknown_or_evidence_policy"], row["bridge_policy"])
            filename = closure_filename(key)
            write_closure(root / filename, closed)
            closure_files.append({"scenario_id": key[0], "policy": key[1],
                                  "bridge_policy": key[2], "path": filename})
        write_json(root / "roads.json", {"schema_version": SCHEMA, "columns": ["segment_id", "lon1", "lat1",
                   "lon2", "lat2", "road_class", "length_m", "bridge", "tunnel"],
                   "roads": [roads[key] for key in sorted(roads)]})
        write_json(root / "evidence.json", evidence)
        write_json(root / "scenarios.json", scenarios)
        hospitals = []
        for snap in snaps["hospitals"]:
            item = dict(snap)
            if snap["status"] == "snapped":
                item["lon"], item["lat"] = nodes[snap["nearest_node_id"]]
            hospitals.append(item)
        preview_by_scenario = {}
        with rasterio.open(m6_root / "case_rasters/designed_cases.tif") as masks, \
             rasterio.open(m6_root / "case_rasters/retained.tif") as retained_raster:
            if masks.crs.to_string() != "EPSG:4326" or masks.count != len({row["scenario_id"] for row in scenarios}):
                raise ValueError("M6 designed-case raster contract differs")
            retained = retained_raster.read(1)
            tile_bounds = [masks.bounds.bottom, masks.bounds.left, masks.bounds.top, masks.bounds.right]
            for band, name in enumerate(masks.descriptions, 1):
                values = masks.read(band)
                if not set(np.unique(values)) <= {0, 1, 255}:
                    raise ValueError("M6 preview mask has unexpected state")
                rgba = np.zeros((masks.height, masks.width, 4), dtype=np.uint8)
                rgba[values == 255] = (110, 119, 135, 135)
                rgba[(values == 0) & (retained == 0)] = (226, 163, 69, 65)
                rgba[(values == 1) & (retained == 0)] = (226, 163, 69, 180)
                rgba[(values == 1) & (retained == 1)] = (38, 169, 181, 210)
                filename = hashlib.sha256(name.encode()).hexdigest()[:12] + ".png"
                path = root / "previews" / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgba, "RGBA").save(path)
                preview_by_scenario[name] = f"previews/{filename}"
        if set(preview_by_scenario) != {row["scenario_id"] for row in scenarios}:
            raise ValueError("M6 scenario IDs and raster descriptions differ")
        about = {"schema_version": SCHEMA, "mode": "real", "run_id": root.name,
                 "case_id": summary["case_id"], "case_title": "Orihuela / Dolores · September 2019",
                 "source_run_id": m6_root.name, "source_manifest_sha256": source_hash,
                 "claim_status": "retrospective_scenario_only",
                 "dates": {"road_snapshot_utc": summary["road_snapshot_utc"],
                           "radar": summary["radar_date"], "optical": summary["optical_date"],
                           "independent_trace": summary["cems_interpretation_date"],
                           "hospital_snapshot": summary["hospital_snapshot_date"]},
                 "study_bounds": quality["study_bbox_wgs84"], "tile_bounds": tile_bounds,
                 "coverage": quality["road_evidence"], "graph_quality": {
                     key: quality[key] for key in ("graph_nodes", "directed_edges", "physical_segments",
                                                  "weak_components", "baseline_reachable_origins")},
                 "origins": _origin_rows(snaps["origins"], summary, nodes), "hospitals": hospitals,
                 "baseline": summary["baseline"], "interpretation": summary["interpretation"],
                 "inspection_ids": summary["inspection_ids"],
                 "primary_scenario_id": summary["primary_scenario"]["scenario_id"],
                 "minimum_exposed_supported_m": minimum,
                 "scenario_options": [{"scenario_id": row["scenario_id"],
                                       "policy": row["unknown_or_evidence_policy"],
                                       "bridge_policy": row["bridge_policy"],
                                       "summary": row["summary"], "closure": row["closure"]}
                                      for row in scenarios],
                 "closure_files": closure_files,
                 "preview_by_scenario": preview_by_scenario,
                 "attribution": ["© OpenStreetMap contributors (ODbL)",
                                 "Sen1Floods11 source imagery; redistribution rights unresolved",
                                 "Copernicus EMSR388 interpreted context",
                                 "INE 2019 section origins", "Valencia 2026 hospital catalogue"]}
        write_json(root / "bundle.json", about)
        return _finish(root, "real", m6_root.name, source_hash)
    except Exception as error:
        write_json(root / "status.json", {"run_id": root.name, "status": "failed",
                                          "error": f"{type(error).__name__}: {error}",
                                          "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def build_synthetic(output_root: Path = Path("runs")) -> dict:
    """Build an unmistakably invented fixture without raw data or raster packages."""
    source_hash = hashlib.sha256(b"m7_synthetic_fixture_v1").hexdigest()
    root = output_root / _bundle_id("synthetic", source_hash)
    if root.exists():
        raise FileExistsError(f"Synthetic bundle {root.name} already exists; preserve and verify it")
    root.mkdir(parents=True)
    write_json(root / "status.json", {"run_id": root.name, "status": "running"})
    roads = [["demo:west", 0.10, 0.10, 0.35, 0.10, "local", 1000.0, False, False],
             ["demo:middle", 0.35, 0.10, 0.60, 0.10, "local", 1000.0, False, False],
             ["demo:east", 0.60, 0.10, 0.85, 0.10, "local", 1000.0, False, False],
             ["demo:branch", 0.35, 0.10, 0.35, 0.35, "local", 1000.0, False, False]]
    evidence = {"demo:middle": {"segment_id": "demo:middle", "road_class": "local", "bridge": False,
                "tunnel": False, "within_tile_m": 1000.0, "outside_tile_m": 0.0,
                "source_supported_m": 1000.0, "source_unknown_m": 0.0,
                "retained_m": 800.0, "quality_flagged_m": 200.0,
                "coverage_fraction_of_segment": 1.0, "unknown_fraction_of_segment": 0.0,
                "retained_fraction_of_supported": 0.8, "mean_score_on_supported": 0.7,
                "source_supported_water_m_by_scenario": {"demo_water": 60.0, "demo_quiet": 0.0},
                "retained_water_m_by_scenario": {"demo_water": 45.0, "demo_quiet": 0.0},
                "interpretation": "Invented water-overlap example; passability unknown"}}
    baseline = [{"origin_id": "demo:origin:1", "nearest_facility_id": "demo:hospital",
                 "estimated_cost_seconds": 300.0, "baseline_cost_seconds": 300.0,
                 "cost_delta_seconds": 0.0, "baseline_unreachable": False,
                 "newly_disconnected": False, "reason": None},
                {"origin_id": "demo:origin:2", "nearest_facility_id": "demo:hospital",
                 "estimated_cost_seconds": 180.0, "baseline_cost_seconds": 180.0,
                 "cost_delta_seconds": 0.0, "baseline_unreachable": False,
                 "newly_disconnected": False, "reason": None}]
    disconnected = [{**baseline[0], "nearest_facility_id": None, "estimated_cost_seconds": None,
                     "cost_delta_seconds": None, "newly_disconnected": True, "reason": "no_graph_path"},
                    baseline[1]]
    scenarios = []
    closure_files = []
    for name, origins, closed in (("demo_water", disconnected, 1), ("demo_quiet", baseline, 0)):
        scenarios.append({"scenario_id": name, "unknown_or_evidence_policy": "supported_overlap",
                          "bridge_policy": "bridge_exempt", "origins": origins,
                          "closure": {"closed_physical_segments": closed,
                                      "reasons": {"supported_overlap": closed, "retained_overlap": 0,
                                                  "within_tile_unknown": 0, "outside_or_partial_unknown": 0},
                                      "assumption": "candidate segment removed from simulated graph; not an observed closure"},
                          "summary": {"origins": 2, "baseline_unreachable": 0,
                                      "newly_disconnected": closed, "paired_reachable": 2 - closed,
                                      "mean_paired_cost_increase_seconds": 0.0,
                                      "maximum_paired_cost_increase_seconds": 0.0,
                                      "ranking": {"comparable_pairs": 0 if closed else 1,
                                                  "reversed_pairs": 0,
                                                  "reversal_fraction": None if closed else 0.0}}})
        key = (name, "supported_overlap", "bridge_exempt")
        filename = closure_filename(key)
        write_closure(root / filename, {"demo:middle"} if closed else set())
        closure_files.append({"scenario_id": name, "policy": key[1],
                              "bridge_policy": key[2], "path": filename})
    previews = {}
    for name in ("demo_water", "demo_quiet"):
        pixels = bytearray(32 * 32 * 4)
        if name == "demo_water":
            for y in range(10, 22):
                for x in range(10, 22):
                    pixels[(y * 32 + x) * 4:(y * 32 + x) * 4 + 4] = bytes((38, 169, 181, 210))
        path = root / "previews" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png_rgba(32, 32, bytes(pixels)))
        previews[name] = f"previews/{name}.png"
    about = {"schema_version": SCHEMA, "mode": "synthetic", "run_id": root.name,
             "case_id": "synthetic_training_fixture", "case_title": "Synthetic demonstration · invented geography",
             "source_run_id": None, "source_manifest_sha256": source_hash,
             "claim_status": "invented_fixture_no_real_world_claim",
             "dates": {"road_snapshot_utc": "invented", "radar": "invented", "optical": "invented",
                       "independent_trace": "invented", "hospital_snapshot": "invented"},
             "study_bounds": [0, 0, 0.45, 1], "tile_bounds": [0.05, 0.28, 0.40, 0.66],
             "coverage": {"physical_segments_total": 4, "segments_intersecting_pilot": 1,
                          "outside_pilot_segments": 3, "source_supported_pixels": 1024,
                          "retained_pixels": 820, "cloud_known_pixels": 0},
             "graph_quality": {"graph_nodes": 5, "directed_edges": 8, "physical_segments": 4,
                               "weak_components": 1, "baseline_reachable_origins": 2},
             "origins": [{"feature_id": "demo:origin:1", "status": "snapped", "lon": 0.1, "lat": 0.1,
                          "nearest_distance_m": 0, "source_date": "invented", "baseline": baseline[0],
                          "designed_stability": {"reachable_members": 1, "designed_members": 2}},
                         {"feature_id": "demo:origin:2", "status": "snapped", "lon": 0.6, "lat": 0.1,
                          "nearest_distance_m": 0, "source_date": "invented", "baseline": baseline[1],
                          "designed_stability": {"reachable_members": 2, "designed_members": 2}}],
             "hospitals": [{"feature_id": "demo:hospital", "status": "snapped", "lon": 0.85, "lat": 0.1,
                            "nearest_distance_m": 0, "source_date": "invented",
                            "historical_status": "invented"}],
             "baseline": baseline,
             "interpretation": ["All coordinates, roads, evidence and access values are invented fixtures.",
                                "Road overlap is not passability or a safe route."],
             "inspection_ids": {"exposed_link": "demo:middle", "bridge_overlap": None,
                                "outside_footprint_link": "demo:west",
                                "newly_disconnected_origin": "demo:origin:1"},
             "primary_scenario_id": "demo_water", "minimum_exposed_supported_m": 15,
             "scenario_options": [{"scenario_id": row["scenario_id"],
                                   "policy": row["unknown_or_evidence_policy"],
                                   "bridge_policy": row["bridge_policy"],
                                   "summary": row["summary"], "closure": row["closure"]}
                                  for row in scenarios],
             "closure_files": closure_files,
             "preview_by_scenario": previews,
             "attribution": ["Synthetic fixture generated by this repository; no real location or source data"]}
    write_json(root / "bundle.json", about)
    write_json(root / "roads.json", {"schema_version": SCHEMA,
                                     "columns": ["segment_id", "lon1", "lat1", "lon2", "lat2",
                                                 "road_class", "length_m", "bridge", "tunnel"],
                                     "roads": roads})
    write_json(root / "evidence.json", evidence)
    write_json(root / "scenarios.json", scenarios)
    return _finish(root, "synthetic", None, source_hash)


def load_bundle(root: Path) -> tuple[dict, dict[str, list], dict[str, dict], dict]:
    """Validate all immutable bundle bytes and cross-file references before serving."""
    root = root.resolve(strict=True)
    manifest_path, status_path = root / "manifest.json", root / "status.json"
    manifest, status = read_json(manifest_path), read_json(status_path)
    if manifest.get("schema_version") != SCHEMA or manifest.get("run_id") != root.name or \
       manifest.get("status") != "complete" or status.get("status") != "complete" or \
       status.get("run_id") != root.name or status.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("M7 bundle manifest/status invalid")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and p not in (manifest_path, status_path)}
    if actual != set(manifest["output_sha256"]):
        raise ValueError("M7 bundle output set differs")
    for name, expected in manifest["output_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ValueError(f"M7 bundle output changed: {name}")
    about = read_json(root / "bundle.json")
    road_doc = read_json(root / "roads.json")
    evidence = read_json(root / "evidence.json")
    scenarios = read_json(root / "scenarios.json")
    if about.get("schema_version") != SCHEMA or about.get("run_id") != root.name or \
       about.get("mode") != manifest["mode"] or road_doc.get("schema_version") != SCHEMA or \
       about.get("source_manifest_sha256") != manifest["source_manifest_sha256"]:
        raise ValueError("M7 bundle schema/provenance invalid")
    roads = {}
    for row in road_doc["roads"]:
        if len(row) != 9 or not isinstance(row[0], str) or not row[0] or row[0] in roads or \
           not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row[1:5]) or \
           not (-180 <= row[1] <= 180 and -90 <= row[2] <= 90 and
                -180 <= row[3] <= 180 and -90 <= row[4] <= 90) or \
           not isinstance(row[5], str) or not row[5] or \
           not isinstance(row[6], (int, float)) or not math.isfinite(row[6]) or row[6] <= 0 or \
           type(row[7]) is not bool or type(row[8]) is not bool:
            raise ValueError("M7 road geometry invalid")
        roads[row[0]] = row
    if not set(evidence) <= roads.keys() or \
       len(roads) != about["coverage"]["physical_segments_total"] or \
       len(evidence) != about["coverage"]["segments_intersecting_pilot"]:
        raise ValueError("M7 road/evidence support differs")
    keys = [(r["scenario_id"], r["unknown_or_evidence_policy"], r["bridge_policy"]) for r in scenarios]
    if len(keys) != len(set(keys)) or set(r["scenario_id"] for r in scenarios) != \
       set(about["preview_by_scenario"]):
        raise ValueError("M7 scenario catalogue invalid")
    if len(about["scenario_options"]) != len(keys):
        raise ValueError("M7 option count differs")
    scenario_index = {key: row for key, row in zip(keys, scenarios)}
    for option in about["scenario_options"]:
        key = (option["scenario_id"], option["policy"], option["bridge_policy"])
        row = scenario_index.get(key)
        if row is None or option["summary"] != row["summary"] or option["closure"] != row["closure"]:
            raise ValueError("M7 option differs from source scenario")
    closure_keys = {(row["scenario_id"], row["policy"], row["bridge_policy"])
                    for row in about["closure_files"]}
    if closure_keys != set(keys) or len(about["closure_files"]) != len(keys) or any(
            row["path"] != closure_filename((row["scenario_id"], row["policy"], row["bridge_policy"]))
            for row in about["closure_files"]):
        raise ValueError("M7 closure index differs")
    scenario_ids = {key[0] for key in keys}
    for segment_id, row in evidence.items():
        if row.get("segment_id") != segment_id or \
           set(row.get("source_supported_water_m_by_scenario", {})) != scenario_ids or \
           set(row.get("retained_water_m_by_scenario", {})) != scenario_ids or \
           not 0 <= row.get("unknown_fraction_of_segment", -1) <= 1:
            raise ValueError("M7 evidence contract differs")
    origins = {row["feature_id"] for row in about["origins"]}
    if len(origins) != len(about["origins"]) or any(
            {item["origin_id"] for item in row["origins"]} != origins for row in scenarios):
        raise ValueError("M7 origin support differs")
    return about, roads, evidence, scenario_index
