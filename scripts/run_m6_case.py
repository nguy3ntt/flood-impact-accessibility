"""Build the bounded Spain road-evidence and simulated hospital-accessibility case."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import Point, box, shape

from flood_access.case_graph import build_graph, shortest_to_facilities
from flood_access.case_evidence import (case_scores, crossing_audit, load_case_observation,
                                        road_evidence, snap_points, tile_footprint_metric)
from flood_access.case_scenarios import access_rows, closed_segments, summarise_access
from flood_access.raster_pipeline import sha256
from acquire_m6_osm import INVENTORY, query
from verify_m2_run import verify as verify_m2
from verify_m5_run import verify as verify_m5


CONFIG_PATH = Path("configs/m6_case_v1.json")
M5_CONFIG_PATH = Path("configs/m5_evidence_v1.json")
CODE_FILES = (Path("scripts/run_m6_case.py"), Path("src/flood_access/case_graph.py"),
              Path("src/flood_access/case_evidence.py"), Path("src/flood_access/case_scenarios.py"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    part.replace(path)


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("wb") as target, gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as compressed:
        for row in rows:
            compressed.write((json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
                              + "\n").encode())
    part.replace(path)


def write_raster(path: Path, tile, array: np.ndarray, descriptions: list[str], nodata) -> None:
    data = array[None] if array.ndim == 2 else array
    if data.shape[1:] != tile.label.shape or len(data) != len(descriptions):
        raise ValueError("Case raster shape/band contract differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=tile.label.shape[0], width=tile.label.shape[1],
                       count=len(data), dtype=data.dtype, crs=tile.crs, transform=tile.transform,
                       nodata=nodata, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", predictor=3 if np.issubdtype(data.dtype, np.floating) else 2) as sink:
        sink.write(data)
        sink.descriptions = tuple(descriptions)


def references(config: dict) -> tuple[dict, dict, dict, dict]:
    m2 = verify_m2(config["m2_run_id"])
    m5 = verify_m5(config["m5_run_id"])
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    source = Path(config["osm_raw_path"])
    full_query = query(config, count_only=False)
    if inventory["logical_path"] != source.as_posix() or inventory["query"] != full_query or \
       inventory["query_sha256"] != hashlib.sha256(full_query.encode()).hexdigest() or \
       inventory["as_of_utc"] != config["osm_as_of_utc"] or \
       inventory["bytes"] != source.stat().st_size or inventory["sha256"] != sha256(source):
        raise ValueError("Historical OSM source inventory differs")
    m5_config = json.loads(M5_CONFIG_PATH.read_text(encoding="utf-8"))
    fit = json.loads((Path("runs") / config["m5_run_id"] / "calibrator.json").read_text(encoding="utf-8"))
    return m2, m5, m5_config, fit


def fingerprint(config: dict, m2: dict, m5: dict) -> dict:
    return {"config": config, "config_sha256": sha256(CONFIG_PATH),
            "m2_manifest_sha256": m2["manifest_sha256"],
            "m5_manifest_sha256": m5["manifest_sha256"],
            "m5_config_sha256": sha256(M5_CONFIG_PATH),
            "m5_calibrator_sha256": sha256(Path("runs") / config["m5_run_id"] / "calibrator.json"),
            "osm_source_sha256": sha256(Path(config["osm_raw_path"])),
            "osm_inventory_sha256": sha256(INVENTORY),
            "code_sha256": {path.as_posix(): sha256(path) for path in CODE_FILES}}


def _geojson(path: Path) -> list[dict]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("type") != "FeatureCollection":
        raise ValueError(f"Invalid case vector collection: {path}")
    return content["features"]


def _route_boundary(origins: dict[str, int], facilities: dict[str, int], edges: list[dict],
                    next_edge: dict[int, str], nodes: dict[int, tuple], bbox, guard_m: float) -> dict:
    index = {edge["edge_id"]: edge for edge in edges}
    facility_nodes = set(facilities.values())
    result = {}
    for origin, start in origins.items():
        node = start
        visited = set()
        minimum = float("inf")
        leaves = False
        count = 0
        while node not in facility_nodes and node in next_edge:
            if node in visited:
                raise ValueError("Cycle in baseline predecessor chain")
            visited.add(node)
            point = Point(nodes[node][:2])
            minimum = min(minimum, point.distance(bbox.boundary))
            leaves |= not bbox.covers(point)
            node = index[next_edge[node]]["target"]
            count += 1
        if node in facility_nodes:
            point = Point(nodes[node][:2])
            minimum = min(minimum, point.distance(bbox.boundary))
            leaves |= not bbox.covers(point)
        result[origin] = {"baseline_path_edge_count": count if node in facility_nodes else None,
                          "minimum_study_boundary_distance_m": minimum if node in facility_nodes else None,
                          "within_boundary_guard": bool(node in facility_nodes and minimum < guard_m),
                          "leaves_study_bbox": bool(node in facility_nodes and leaves)}
    return result


def _qa_tile(path: Path, tile, probability, retained, segments, nodes, evidence,
             origins: list[dict], traces: list[dict], scenario: str, minimum_m: float) -> None:
    size = tile.label.shape[0]
    gray = np.clip(probability * 255, 0, 255).astype(np.uint8)
    rgb = np.full((*tile.label.shape, 3), 130, dtype=np.uint8)
    rgb[tile.eligible] = np.column_stack((gray[tile.eligible],
                                         (70 + gray[tile.eligible] // 2).astype(np.uint8),
                                         255 - gray[tile.eligible]))
    image = Image.fromarray(rgb).resize((size * 2, size * 2), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    for trace in traces:
        geometry = shape(trace["geometry"])
        polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        for polygon in polygons:
            coords = [rasterio.transform.rowcol(tile.transform, x, y)
                      for x, y in polygon.exterior.coords]
            draw.line([(col * 2, row * 2) for row, col in coords], fill=(45, 225, 225), width=1)
    from rasterio.warp import transform
    for segment_id, row in evidence.items():
        edge = segments[segment_id]
        a, b = nodes[edge["source"]], nodes[edge["target"]]
        lon, lat = transform("EPSG:25830", str(tile.crs), [a[0], b[0]], [a[1], b[1]])
        pixels = [rasterio.transform.rowcol(tile.transform, x, y) for x, y in zip(lon, lat)]
        water = row["source_supported_water_m_by_scenario"][scenario]
        color = (230, 48, 48) if water >= minimum_m else \
                (255, 190, 30) if row["retained_m"] < row["within_tile_m"] - 1e-6 else (20, 20, 20)
        draw.line([(c * 2, r * 2) for r, c in pixels], fill=color, width=2 if water >= minimum_m else 1)
    for feature in origins:
        point = shape(feature["geometry"])
        row, col = rasterio.transform.rowcol(tile.transform, point.x, point.y)
        draw.ellipse((col * 2 - 4, row * 2 - 4, col * 2 + 4, row * 2 + 4),
                     fill=(255, 255, 255), outline=(60, 40, 180), width=2)
    canvas = Image.new("RGB", (size * 2, size * 2 + 70), "white")
    canvas.paste(image, (0, 30))
    label = ImageDraw.Draw(canvas)
    label.text((8, 8), "Spain pilot: empirical score + 2019 OSM road overlap", fill="black")
    label.text((8, size * 2 + 38), "red overlap candidate | amber withheld | cyan CEMS trace | white origin; no closures",
               fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _qa_overview(path: Path, config: dict, nodes: dict[int, tuple], segments: dict[str, dict],
                 hospital_features: list[dict], origins: list[dict], tile, bbox) -> None:
    width = height = 950
    image = Image.new("RGB", (width, height + 55), "white")
    draw = ImageDraw.Draw(image)
    minx, miny, maxx, maxy = bbox.bounds
    def point(x, y):
        return (int(35 + (x - minx) / (maxx - minx) * (width - 70)),
                int(35 + (maxy - y) / (maxy - miny) * (height - 70)))
    for edge in segments.values():
        a, b = nodes[edge["source"]], nodes[edge["target"]]
        if bbox.intersects(box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))):
            draw.line((point(a[0], a[1]), point(b[0], b[1])), fill=(200, 200, 200), width=1)
    bounds = tile_footprint_metric(tile, config["metric_crs"]).bounds
    x1, y1 = point(bounds[0], bounds[3]); x2, y2 = point(bounds[2], bounds[1])
    draw.rectangle((x1, y1, x2, y2), outline=(210, 40, 40), width=3)
    for features, fill, radius in ((hospital_features, (30, 80, 220), 7), (origins, (25, 145, 60), 4)):
        for feature in features:
            geometry = shape(feature["geometry"])
            from rasterio.warp import transform
            x, y = transform("EPSG:4326", config["metric_crs"], [geometry.x], [geometry.y])
            if bbox.covers(Point(x[0], y[0])):
                px, py = point(x[0], y[0])
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=fill)
    draw.text((8, 8), "2019 road snapshot | pilot footprint in red | 2026 hospitals blue | 2019 origins green",
              fill="black")
    draw.text((8, height + 18), "Static estimated graph; roads outside the red pilot have no mapped water evidence.",
              fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def derive(config: dict, m5_config: dict, fit: dict) -> dict:
    osm = json.loads(Path(config["osm_raw_path"]).read_text(encoding="utf-8"))
    nodes, edges, graph = build_graph(osm["elements"], config["speed_kmh_by_class"], config["osm_as_of_utc"])
    if len(osm["elements"]) != json.loads(INVENTORY.read_text(encoding="utf-8"))["ways"] + \
       json.loads(INVENTORY.read_text(encoding="utf-8"))["nodes"]:
        raise ValueError("OSM element count differs from source inventory")
    tile_root = Path("data/processed/m2_runs") / config["m2_run_id"] / "tiles" / config["spain_tile_id"]
    tile = load_case_observation(tile_root, config["spain_tile_id"])
    probability, flags, retained, cases, names = case_scores(tile, fit, m5_config, config)
    segments = {edge["segment_id"]: edge for edge in edges}
    evidence = road_evidence(segments, nodes, tile, probability, flags, retained, cases, names,
                             sample_step_m=config["road_sample_step_m"], metric_crs=config["metric_crs"])
    for segment_id, row in evidence.items():
        length = segments[segment_id]["length_m"]
        if row["within_tile_m"] > length + 1e-5 or row["source_supported_m"] > row["within_tile_m"] + 1e-5 or \
           row["retained_m"] > row["source_supported_m"] + 1e-5 or \
           any(value > row["source_supported_m"] + 1e-5 for value in row["source_supported_water_m_by_scenario"].values()):
            raise ValueError(f"Road exposure length contract differs: {segment_id}")
    crossing = crossing_audit(segments, nodes, evidence)
    south, west, north, east = config["osm_bbox_south_west_north_east"]
    bbox = shape(transform_geom("EPSG:4326", config["metric_crs"],
                                box(west, south, east, north).__geo_interface__, precision=9))
    vector_root = Path("data/processed/m2_runs") / config["m2_run_id"] / "vectors/spain_7370579"
    hospital_features = _geojson(vector_root / "hospitals.geojson")
    origin_features = _geojson(vector_root / "origins.geojson")
    traces = _geojson(vector_root / "flood_traces.geojson")
    hospitals = snap_points(hospital_features, nodes, metric_crs=config["metric_crs"],
                            max_distance_m=config["hospital_snap_max_m"], study_bbox_metric=bbox)
    origins = snap_points(origin_features, nodes, metric_crs=config["metric_crs"],
                          max_distance_m=config["origin_snap_max_m"], study_bbox_metric=bbox)
    for row in [*hospitals, *origins]:
        row["weak_component_id"] = str(graph["component_by_node"][row["nearest_node_id"]]) \
                                   if row["status"] == "snapped" else None
    facility_nodes = {row["feature_id"]: row["nearest_node_id"] for row in hospitals if row["status"] == "snapped"}
    origin_nodes = {row["feature_id"]: row["nearest_node_id"] for row in origins if row["status"] == "snapped"}
    if not facility_nodes or not origin_nodes:
        raise ValueError("No in-bounds snapped facility/origin support")
    base_distances, _, next_edge = shortest_to_facilities(edges, facility_nodes, set())
    baseline = access_rows(edges, facility_nodes, origin_nodes, set())
    baseline_by_origin = {row["origin_id"]: row for row in baseline}
    boundary_paths = _route_boundary(origin_nodes, facility_nodes, edges, next_edge, nodes, bbox,
                                     config["boundary_guard_m"])
    graph.pop("component_by_node")
    graph["geometric_crossings_in_pilot"] = crossing
    graph["study_bbox_wgs84"] = config["osm_bbox_south_west_north_east"]
    graph["facility_snap_status"] = {status: sum(row["status"] == status for row in hospitals)
                                     for status in sorted({row["status"] for row in hospitals})}
    graph["origin_snap_status"] = {status: sum(row["status"] == status for row in origins)
                                   for status in sorted({row["status"] for row in origins})}
    graph["baseline_reachable_origins"] = sum(row["estimated_cost_seconds"] is not None for row in baseline)
    graph["baseline_path_boundary_audit"] = boundary_paths
    graph["road_evidence"] = {"physical_segments_total": len(segments),
                              "segments_intersecting_pilot": len(evidence),
                              "within_pilot_length_m": sum(row["within_tile_m"] for row in evidence.values()),
                              "source_supported_length_m": sum(row["source_supported_m"] for row in evidence.values()),
                              "retained_length_m": sum(row["retained_m"] for row in evidence.values()),
                              "outside_pilot_segments": len(segments) - len(evidence),
                              "cloud_known_pixels": 0,
                              "source_supported_pixels": int(np.count_nonzero(tile.eligible)),
                              "retained_pixels": int(np.count_nonzero(retained))}
    results = []
    primary = config["primary_mapping_scenario"]
    for scenario in names:
        policies = ("supported_overlap", "retained_overlap")
        if scenario == primary:
            policies += ("supported_plus_within_tile_unknown", "supported_plus_all_unknown")
        for policy in policies:
            for bridge_policy in config["bridge_policies"]:
                closed, closure = closed_segments(segments, evidence, scenario, policy=policy,
                                                  bridge_policy=bridge_policy,
                                                  minimum_exposed_m=config["minimum_exposed_supported_m"])
                changed = access_rows(edges, facility_nodes, origin_nodes, closed, baseline_by_origin)
                summary = summarise_access(baseline, changed)
                results.append({"scenario_id": scenario, "unknown_or_evidence_policy": policy,
                                "bridge_policy": bridge_policy, "closure": closure,
                                "summary": summary, "origins": changed})
    designed = [row for row in results if row["unknown_or_evidence_policy"] == "supported_overlap"
                and row["bridge_policy"] == "bridge_exempt"]
    boundary_by_scenario = {}
    for row in designed:
        closed, _ = closed_segments(segments, evidence, row["scenario_id"], policy="supported_overlap",
                                    bridge_policy="bridge_exempt",
                                    minimum_exposed_m=config["minimum_exposed_supported_m"])
        _, _, predecessors = shortest_to_facilities(edges, facility_nodes, closed)
        paths = _route_boundary(origin_nodes, facility_nodes, edges, predecessors, nodes, bbox,
                                config["boundary_guard_m"])
        boundary_by_scenario[row["scenario_id"]] = {
            "reachable_paths_within_boundary_guard": sum(item["within_boundary_guard"] for item in paths.values()),
            "reachable_paths_leaving_bbox": sum(item["leaves_study_bbox"] for item in paths.values()),
            "minimum_boundary_distance_m": min((item["minimum_study_boundary_distance_m"] for item in paths.values()
                                                if item["minimum_study_boundary_distance_m"] is not None), default=None)}
    stability = []
    for origin in sorted(origin_nodes):
        values = [next(item for item in row["origins"] if item["origin_id"] == origin)["estimated_cost_seconds"]
                  for row in designed]
        present = [value for value in values if value is not None]
        stability.append({"origin_id": origin, "designed_members": len(values),
                          "reachable_members": len(present),
                          "newly_disconnected_members": len(values) - len(present)
                          if baseline_by_origin[origin]["estimated_cost_seconds"] is not None else 0,
                          "reachable_fraction_of_designed_members": len(present) / len(values),
                          "minimum_cost_seconds_when_reachable": min(present) if present else None,
                          "maximum_cost_seconds_when_reachable": max(present) if present else None,
                          "interpretation": "frequency across designed cases, not a probability"})
    inspection = {}
    candidates = [row for row in evidence.values()
                  if row["source_supported_water_m_by_scenario"][primary] >= config["minimum_exposed_supported_m"]]
    if candidates:
        inspection["exposed_link"] = max(candidates, key=lambda row:
                                          row["source_supported_water_m_by_scenario"][primary])["segment_id"]
        bridge_candidates = [row for row in candidates if row["bridge"]]
        inspection["bridge_overlap"] = bridge_candidates[0]["segment_id"] if bridge_candidates else None
    inspection["outside_footprint_link"] = next(segment for segment in sorted(segments)
                                                 if segment not in evidence)
    primary_row = next(row for row in results if row["scenario_id"] == primary and
                       row["unknown_or_evidence_policy"] == "supported_overlap" and
                       row["bridge_policy"] == "bridge_exempt")
    inspection["newly_disconnected_origin"] = next((row["origin_id"] for row in primary_row["origins"]
                                                       if row["newly_disconnected"]), None)
    summary = {"schema_version": "m6_case_summary_v1", "case_id": config["case_id"],
               "road_snapshot_utc": config["osm_as_of_utc"],
               "radar_date": "2019-09-17", "optical_date": "2019-09-18",
               "cems_interpretation_date": "2019-09-18", "hospital_snapshot_date": "2026-08-05",
               "baseline": baseline, "designed_scenario_stability": stability,
               "designed_scenario_boundary_audit": boundary_by_scenario,
               "primary_scenario": {"scenario_id": primary,
                                    "supported_overlap_bridge_exempt": primary_row["summary"]},
               "inspection_ids": inspection,
               "interpretation": ["Retrospective graph-node travel cost under explicit static speed and edge-removal assumptions.",
                                  "Road-water overlap is exposure evidence, never observed closure or passability.",
                                  "M5 calibration is not validated on Spain; M3 NDWI threshold is a separate fixed reference.",
                                  "Roads outside the single image have unknown water evidence, not verified dry status.",
                                  "Scenario-member frequencies are design frequencies, not probabilities.",
                                  "2026 hospitals are a retrospective mismatch; one candidate lies outside the study box.",
                                  "No result is a public travel or emergency route recommendation."]}
    return {"nodes": nodes, "edges": edges, "segments": segments, "tile": tile,
            "probability": probability, "flags": flags, "retained": retained,
            "cases": cases, "names": names, "evidence": evidence,
            "hospitals": hospitals, "origins": origins, "hospital_features": hospital_features,
            "origin_features": origin_features, "traces": traces, "bbox": bbox,
            "graph_quality": graph, "results": results, "summary": summary}


def run() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    m2, m5, m5_config, fit = references(config)
    stamp = fingerprint(config, m2, m5)
    run_id = "M6-" + hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:12]
    root, report = Path("runs") / run_id, Path("reports/m6") / run_id
    status_path = root / "status.json"
    if status_path.exists():
        raise FileExistsError(f"M6 run {run_id} exists; preserve and verify it")
    write_json(status_path, {"run_id": run_id, "status": "running",
                             "started_at_utc": datetime.now(timezone.utc).isoformat()})
    start = time.perf_counter()
    try:
        data = derive(config, m5_config, fit)
        tile = data["tile"]
        print(json.dumps({"run_id": run_id, "phase": "derived", "graph_nodes": len(data["nodes"]),
                          "directed_edges": len(data["edges"]), "scenario_rows": len(data["results"])}), flush=True)
        write_rows(root / "graph_nodes.jsonl.gz", ({"node_id": node, "x_m": pos[0], "y_m": pos[1],
                                                    "lon": pos[2], "lat": pos[3]}
                                                   for node, pos in sorted(data["nodes"].items())))
        write_rows(root / "graph_edges.jsonl.gz", data["edges"])
        write_rows(root / "road_evidence.jsonl.gz", (data["evidence"][key]
                                                       for key in sorted(data["evidence"])))
        write_json(root / "graph_quality.json", data["graph_quality"])
        write_json(root / "snaps.json", {"hospitals": data["hospitals"], "origins": data["origins"]})
        write_json(root / "scenario_results.json", data["results"])
        write_json(root / "summary.json", data["summary"])
        raster_root = root / "case_rasters"
        write_raster(raster_root / "source_support.tif", tile, tile.eligible.astype(np.uint8),
                     ["1_sensor_valid_index_defined_0_unknown"], 0)
        write_raster(raster_root / "probability.tif", tile, data["probability"],
                     ["empirical_observed_water_score_Spain_transfer_unvalidated"], np.nan)
        write_raster(raster_root / "quality_flags.tif", tile, data["flags"],
                     ["bit1_weak_denominator_bit2_bright_TOA_255_unknown"], 255)
        state = np.full(tile.label.shape, 255, dtype=np.uint8)
        state[tile.eligible] = 0
        state[data["retained"]] = 1
        write_raster(raster_root / "retained.tif", tile, state,
                     ["1_retained_0_abstained_255_source_unknown"], 255)
        write_raster(raster_root / "designed_cases.tif", tile, data["cases"], data["names"], 255)
        _qa_tile(report / "qa_tile_overlap.png", tile, data["probability"], data["retained"],
                 data["segments"], data["nodes"], data["evidence"], data["origin_features"],
                 data["traces"], config["primary_mapping_scenario"],
                 config["minimum_exposed_supported_m"])
        _qa_overview(report / "qa_graph_overview.png", config, data["nodes"], data["segments"],
                     data["hospital_features"], data["origin_features"], tile, data["bbox"])
        files = sorted(path for folder in (root, report) for path in folder.rglob("*")
                       if path.is_file() and path not in (status_path, root / "run_manifest.json"))
        manifest = {"schema_version": "m6_case_run_v1", "run_id": run_id, "status": "complete",
                    "fingerprint": stamp, "m2_run_id": config["m2_run_id"],
                    "m5_run_id": config["m5_run_id"], "osm_source_inventory": INVENTORY.as_posix(),
                    "source_sha256": {"osm": sha256(Path(config["osm_raw_path"])),
                                      "m2_manifest": m2["manifest_sha256"],
                                      "m5_manifest": m5["manifest_sha256"]},
                    "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                    "rasterio": rasterio.__version__},
                    "output_sha256": {path.as_posix(): sha256(path) for path in files},
                    "wall_seconds": time.perf_counter() - start}
        write_json(root / "run_manifest.json", manifest)
        write_json(status_path, {"run_id": run_id, "status": "complete",
                                 "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "manifest_sha256": sha256(root / "run_manifest.json")})
        return {"run_id": run_id, "status": "complete", "graph_nodes": len(data["nodes"]),
                "directed_edges": len(data["edges"]), "scenario_rows": len(data["results"]),
                "output_files": len(files), "wall_seconds": manifest["wall_seconds"]}
    except Exception as error:
        write_json(status_path, {"run_id": run_id, "status": "failed",
                                 "error": f"{type(error).__name__}: {error}",
                                 "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
