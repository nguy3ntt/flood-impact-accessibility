"""Small Spain case-study vector tables, with explicit CRS and evidence status."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import zipfile

import rasterio
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape
from shapely.validation import explain_validity


METRIC_CRS = "EPSG:25830"
GEOGRAPHIC_CRS = "EPSG:4326"


def _project(geometry: dict, source_crs: str, target_crs: str):
    return shape(transform_geom(source_crs, target_crs, geometry, precision=9))


def _read_zip_layer(archive: zipfile.ZipFile, keyword: str) -> list[dict]:
    names = [name for name in archive.namelist() if keyword in name and name.endswith(".json")]
    if len(names) != 1:
        raise ValueError(f"Expected one {keyword} GeoJSON layer, got {names}")
    collection = json.loads(archive.read(names[0]))
    if collection.get("type") != "FeatureCollection":
        raise ValueError(f"Invalid {keyword} collection")
    return collection["features"]


def validate_features(features: list[dict], geometry_type: str) -> None:
    ids: set[str] = set()
    for feature in features:
        identifier = feature.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError(f"Missing or duplicate feature ID: {identifier}")
        ids.add(identifier)
        geometry = shape(feature["geometry"])
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Invalid geometry for {identifier}: {explain_validity(geometry)}")
        if geometry_type == "point" and geometry.geom_type != "Point":
            raise ValueError(f"Expected point for {identifier}")
        if geometry_type == "line" and geometry.geom_type not in ("LineString", "MultiLineString"):
            raise ValueError(f"Expected line for {identifier}")
        if geometry_type == "polygon" and geometry.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"Expected polygon for {identifier}")
        if not all(math.isfinite(value) for value in geometry.bounds):
            raise ValueError(f"Non-finite geometry for {identifier}")
        if not (-180 <= geometry.bounds[0] <= geometry.bounds[2] <= 180
                and -90 <= geometry.bounds[1] <= geometry.bounds[3] <= 90):
            raise ValueError(f"Coordinates are not EPSG:4326 for {identifier}")


def write_geojson(path: Path, features: list[dict], geometry_type: str) -> None:
    validate_features(features, geometry_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    collection = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(collection, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def build_spain_vectors(cems_zip: Path, hospital_geojson: Path, origin_geojson: Path,
                        pilot_label: Path, output_dir: Path) -> dict:
    """Clip reference evidence to one chip; do not construct a road graph."""
    with rasterio.open(pilot_label) as label:
        if str(label.crs) != GEOGRAPHIC_CRS:
            raise ValueError("Pilot label must be EPSG:4326")
        pilot_box = box(*label.bounds)
    metric_box = _project(mapping(pilot_box), GEOGRAPHIC_CRS, METRIC_CRS)
    with zipfile.ZipFile(cems_zip) as archive:
        source_roads = _read_zip_layer(archive, "transportationL")
        source_traces = _read_zip_layer(archive, "observedEventA")
    roads = []
    rejected_roads = []
    for index, source in enumerate(source_roads):
        geometry = shape(source["geometry"])
        if not geometry.is_valid or geometry.is_empty:
            rejected_roads.append({"source_index": index, "reason": explain_validity(geometry)})
            continue
        if not geometry.intersects(pilot_box):
            continue
        clipped = geometry.intersection(pilot_box)
        if clipped.is_empty or clipped.geom_type not in ("LineString", "MultiLineString"):
            continue
        length_m = _project(mapping(clipped), GEOGRAPHIC_CRS, METRIC_CRS).length
        if not math.isfinite(length_m) or length_m <= 0:
            rejected_roads.append({"source_index": index, "reason": "nonpositive projected length"})
            continue
        properties = source["properties"]
        roads.append({
            "type": "Feature", "id": f"EMSR388_AOI07_road_{index:05d}",
            "geometry": mapping(clipped),
            "properties": {
                "source_index": index, "source_id": "cems:EMSR388:AOI07",
                "road_class": properties.get("info"), "name": properties.get("name"),
                "damage_grade": properties.get("damage_gra"),
                "damage_source_id": properties.get("dmg_src_id"),
                "road_geometry_source_id": properties.get("or_src_id"),
                "length_m": round(length_m, 3), "length_crs": METRIC_CRS,
                "grading_image_date": "2019-09-18",
                "road_geometry_snapshot": "pre-event OpenStreetMap; exact date unknown",
                "passability": "unknown",
                "direction": "unknown", "grade_separation": "unknown",
                "graph_eligible": False,
            },
        })
    traces = []
    for index, source in enumerate(source_traces):
        geometry = shape(source["geometry"])
        if not geometry.is_valid or geometry.is_empty:
            raise ValueError(f"Invalid flood trace at source index {index}")
        if not geometry.intersects(pilot_box):
            continue
        clipped = geometry.intersection(pilot_box)
        if clipped.is_empty or clipped.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        traces.append({"type": "Feature", "id": f"EMSR388_AOI07_trace_{index:05d}",
                       "geometry": mapping(clipped),
                       "properties": {"source_index": index, "source_id": "cems:EMSR388:AOI07",
                                      "class": "flood_trace", "image_date": "2019-09-18",
                                      "instantaneous_water_truth": False}})
    hospital_data = json.loads(hospital_geojson.read_text(encoding="utf-8"))
    hospital_crs = hospital_data.get("crs", {}).get("properties", {}).get("name", "")
    if "25830" not in hospital_crs:
        raise ValueError(f"Unexpected hospital CRS: {hospital_crs}")
    hospitals = []
    for source in hospital_data["features"]:
        metric_point = shape(source["geometry"])
        if metric_point.geom_type != "Point" or not metric_point.is_valid:
            raise ValueError("Invalid hospital point")
        distance = metric_point.distance(metric_box)
        if distance > 25_000:
            continue
        properties = source["properties"]
        code = properties.get("CEN_COD")
        if code is None or not math.isfinite(float(code)):
            raise ValueError("Missing hospital code")
        hospitals.append({
            "type": "Feature", "id": f"GVA_hospital_{int(code)}",
            "geometry": mapping(_project(mapping(metric_point), METRIC_CRS, GEOGRAPHIC_CRS)),
            "properties": {"source_id": "gva:hospitals:20260805",
                           "name": properties.get("CEN_DESCLA"), "snapshot_date": "2026-08-05",
                           "distance_to_tile_m": round(distance, 3), "distance_crs": METRIC_CRS,
                           "operating_status_2019": "unknown", "snap_status": "not_attempted"},
        })
    origin_data = json.loads(origin_geojson.read_text(encoding="utf-8"))
    origin_crs = origin_data.get("crs", {}).get("properties", {}).get("name", "")
    if "4326" not in origin_crs:
        raise ValueError(f"Unexpected origin CRS: {origin_crs}")
    origins = []
    rejected_origin_sections = []
    for source in origin_data["features"]:
        if source["properties"].get("TIPO") != "SECCIONADO":
            continue
        geometry = shape(source["geometry"])
        if not geometry.is_valid or geometry.is_empty:
            rejected_origin_sections.append(source["properties"].get("CUSEC"))
            continue
        if not geometry.intersects(pilot_box):
            continue
        point = geometry.intersection(pilot_box).representative_point()
        section_id = source["properties"].get("CUSEC")
        if not section_id:
            raise ValueError("Missing census section ID")
        origins.append({
            "type": "Feature", "id": f"INE2019_section_{section_id}",
            "geometry": mapping(point),
            "properties": {"source_id": "ine:secciones_2019:spain_pilot",
                           "section_id": section_id, "definition": "representative point of section clipped to pilot tile",
                           "geometry_year": 2019, "population": None, "snap_status": "not_attempted"},
        })
    roads.sort(key=lambda item: item["id"])
    traces.sort(key=lambda item: item["id"])
    hospitals.sort(key=lambda item: item["id"])
    origins.sort(key=lambda item: item["id"])
    write_geojson(output_dir / "roads.geojson", roads, "line")
    write_geojson(output_dir / "flood_traces.geojson", traces, "polygon")
    write_geojson(output_dir / "hospitals.geojson", hospitals, "point")
    write_geojson(output_dir / "origins.geojson", origins, "point")
    return {
        "schema_version": "m2_vector_v1", "crs": GEOGRAPHIC_CRS,
        "metric_crs": METRIC_CRS, "pilot_tile": pilot_label.stem.removesuffix("_LabelHand"),
        "source_counts": {"roads": len(source_roads), "flood_traces": len(source_traces),
                          "hospitals": len(hospital_data["features"]), "origin_sections": len(origin_data["features"])},
        "selected_counts": {"roads": len(roads), "flood_traces": len(traces),
                            "hospitals_within_25km": len(hospitals), "origins_in_tile": len(origins)},
        "road_damage_grades": dict(sorted(Counter(f["properties"]["damage_grade"] for f in roads).items())),
        "rejected_roads": rejected_roads,
        "rejected_origin_sections": rejected_origin_sections,
        "road_graph_status": "not_constructed; directions, grade separation, outside-tile continuity and facility snaps unknown",
        "facility_temporal_mismatch": "2026-08-05 facility file over 2019 flood event",
    }
