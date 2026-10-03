"""Label-independent Spain observation support and road-water overlap evidence."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform, transform_geom
from shapely.geometry import LineString, Point, box, shape
from shapely.strtree import STRtree

from .evidence import (calibrated_probability, quality_flags,
                       retained_mask, spatial_scenarios)
from .mapping import MappingTile, optical_index
from .raster_pipeline import S1_BANDS, S2_BANDS, VALIDITY_BANDS


def load_case_observation(tile_dir: Path, expected_id: str) -> MappingTile:
    """Open sensors and validity only; source labels never determine case support."""
    metadata = json.loads((tile_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["tile_id"] != expected_id or metadata["analysis_split"] != "case_study_exploratory":
        raise ValueError("Unexpected case tile identity/split")
    with rasterio.open(tile_dir / "radar_db.tif") as radar_source, \
         rasterio.open(tile_dir / "optical_toa.tif") as optical_source, \
         rasterio.open(tile_dir / "validity.tif") as validity_source, \
         rasterio.open(tile_dir / "cloud_status.tif") as cloud_source:
        sources = (optical_source, validity_source, cloud_source)
        if any(source.shape != radar_source.shape or source.crs != radar_source.crs or
               source.transform != radar_source.transform for source in sources):
            raise ValueError("Case sensor grids differ")
        if radar_source.descriptions != S1_BANDS or optical_source.descriptions != S2_BANDS or \
           validity_source.descriptions != VALIDITY_BANDS or cloud_source.count != 1:
            raise ValueError("Case band contract differs")
        radar = radar_source.read().astype(np.float32)
        optical = optical_source.read().astype(np.float32)
        validity = validity_source.read() == 1
        cloud = cloud_source.read(1)
        crs, affine = radar_source.crs, radar_source.transform
    if not np.all(cloud == 255) or not np.array_equal(validity[3], validity[1] & validity[2]) or \
       not np.array_equal(validity[3], np.isfinite(radar).all(axis=0) & np.isfinite(optical).all(axis=0)):
        raise ValueError("Case source validity/cloud semantics differ")
    green = optical[S2_BANDS.index("B3")]
    nir = optical[S2_BANDS.index("B8")]
    swir = optical[S2_BANDS.index("B11")]
    defined = (np.abs(green + nir) > 1e-6) & (np.abs(green + swir) > 1e-6)
    support = validity[3] & defined
    # MappingTile carries a label field, but no label file is opened or consulted.
    unknown_label = np.full(support.shape, -1, dtype=np.int8)
    return MappingTile(expected_id, metadata["event_id"], "case_study_exploratory",
                       unknown_label, radar, optical, support, affine, crs)


def case_scores(tile: MappingTile, fit: dict, config: dict, case_config: dict) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    probability = calibrated_probability(tile, fit)
    flags = quality_flags(tile,
                          weak_denominator_below=config["quality"]["weak_index_denominator_below"],
                          radiometric_extreme_above=config["quality"]["radiometric_extreme_above_toa"])
    retained = retained_mask(tile, probability, flags, ambiguity_margin=0.15, require_quality=True)
    cases, names = spatial_scenarios(probability, tile.eligible, tile.tile_id,
                                      block_pixels=config["scenarios"]["block_pixels"],
                                      thresholds=config["scenarios"]["fixed_probability_thresholds"],
                                      offset_seeds=config["scenarios"]["structured_offset_seeds"],
                                      maximum_offset=config["scenarios"]["maximum_block_probability_offset"])
    training_threshold = case_config["m3_ndwi_training_threshold"]
    ndwi = optical_index(tile, "ndwi")
    reference = np.full(tile.label.shape, 255, dtype=np.uint8)
    reference[tile.eligible] = (ndwi[tile.eligible] >= training_threshold).astype(np.uint8)
    extra = [reference]
    extra_names = ["m3_ndwi_reference"]
    for offset in case_config["m3_ndwi_sensitivity_offsets"]:
        band = np.full(tile.label.shape, 255, dtype=np.uint8)
        band[tile.eligible] = (ndwi[tile.eligible] >= training_threshold + offset).astype(np.uint8)
        extra.append(band)
        extra_names.append(f"m3_ndwi_threshold_{training_threshold + offset:+.2f}")
    # Shared score offsets are sensitivity designs, centred on the fixed M3
    # training threshold rather than the M5 probability decision cutoff.
    base_p = float(1 / (1 + np.exp(-(fit["slope"] * training_threshold + fit["intercept"]))))
    block_size = config["scenarios"]["block_pixels"]
    amplitude = case_config["m3_maximum_block_score_offset"]
    for seed in case_config["m3_spatial_offset_seeds"]:
        band = np.full(tile.label.shape, 255, dtype=np.uint8)
        for top in range(0, tile.label.shape[0], block_size):
            for left in range(0, tile.label.shape[1], block_size):
                window = np.s_[top:top + block_size, left:left + block_size]
                eligible = tile.eligible[window]
                if not eligible.any():
                    continue
                digest = hashlib.sha256(f"m6-block-v1:{tile.tile_id}:{top}:{left}:{seed}".encode()).digest()
                delta = (2 * int.from_bytes(digest[:8], "little") / (2 ** 64 - 1) - 1) * amplitude
                view = band[window]
                view[eligible] = (np.clip(probability[window][eligible] + delta, 0, 1)
                                  >= base_p).astype(np.uint8)
        extra.append(band)
        extra_names.append(f"m3_shared_offset_seed_{seed}")
    cases = np.concatenate((cases, np.stack(extra)), axis=0)
    names.extend(extra_names)
    return probability, flags, retained, cases, names


def tile_footprint_metric(tile: MappingTile, metric_crs: str):
    west, south, east, north = rasterio.transform.array_bounds(*tile.label.shape, tile.transform)
    geometry = transform_geom(str(tile.crs), metric_crs, box(west, south, east, north).__geo_interface__,
                              precision=9)
    return shape(geometry)


def snap_points(features: list[dict], nodes: dict[int, tuple], *, metric_crs: str,
                max_distance_m: float, study_bbox_metric) -> list[dict]:
    if max_distance_m <= 0 or not nodes:
        raise ValueError("Invalid snapping settings")
    ids = sorted(nodes)
    tree = STRtree([Point(nodes[node][0], nodes[node][1]) for node in ids])
    rows = []
    for feature in features:
        geometry = shape(feature["geometry"])
        if geometry.geom_type != "Point" or geometry.is_empty:
            raise ValueError("Snap input must be a point")
        x, y = transform("EPSG:4326", metric_crs, [geometry.x], [geometry.y])
        point = Point(x[0], y[0])
        inside = study_bbox_metric.covers(point)
        closest_index = int(tree.nearest(point))
        node_id = ids[closest_index]
        distance = point.distance(Point(nodes[node_id][0], nodes[node_id][1]))
        status = "snapped" if inside and distance <= max_distance_m else \
                 "outside_study_bbox" if not inside else "beyond_snap_limit"
        rows.append({"feature_id": feature["id"], "status": status,
                     "nearest_node_id": node_id if status == "snapped" else None,
                     "nearest_distance_m": distance,
                     "max_distance_m": max_distance_m,
                     "boundary_distance_m": point.distance(study_bbox_metric.boundary) if inside else 0.0,
                     "source_id": feature["properties"]["source_id"],
                     "source_date": feature["properties"].get("snapshot_date") or
                                    feature["properties"].get("geometry_year"),
                     "historical_status": feature["properties"].get("operating_status_2019", "not_applicable")})
    return rows


def road_evidence(segments: dict[str, dict], nodes: dict[int, tuple], tile: MappingTile,
                  probability: np.ndarray, flags: np.ndarray, retained: np.ndarray,
                  cases: np.ndarray, names: list[str], *, sample_step_m: float,
                  metric_crs: str) -> dict[str, dict]:
    if sample_step_m <= 0 or not np.array_equal(np.isnan(probability), ~tile.eligible):
        raise ValueError("Invalid road sampling support")
    footprint = tile_footprint_metric(tile, metric_crs)
    candidates = {}
    for segment_id, edge in segments.items():
        a, b = nodes[edge["source"]], nodes[edge["target"]]
        line = LineString(((a[0], a[1]), (b[0], b[1])))
        if line.intersects(footprint):
            candidates[segment_id] = line
    result = {}
    for segment_id, line in candidates.items():
        edge = segments[segment_id]
        overlap = line.intersection(footprint)
        parts = [overlap] if overlap.geom_type == "LineString" else \
                list(overlap.geoms) if overlap.geom_type == "MultiLineString" else []
        inside_m = sum(part.length for part in parts)
        lengths = []
        locations = []
        for part in parts:
            if part.length <= 0:
                continue
            count = max(1, math.ceil(part.length / sample_step_m))
            lengths.extend([part.length / count] * count)
            locations.extend([part.interpolate((i + 0.5) / count, normalized=True)
                              for i in range(count)])
        if locations:
            lon, lat = transform(metric_crs, str(tile.crs), [p.x for p in locations],
                                 [p.y for p in locations])
            rows, cols = rowcol(tile.transform, lon, lat)
            valid_indices = [i for i, (r, c) in enumerate(zip(rows, cols))
                             if 0 <= r < tile.label.shape[0] and 0 <= c < tile.label.shape[1]]
        else:
            valid_indices = []
        supported = retained_m = flagged_m = 0.0
        weighted_score = 0.0
        water = {name: 0.0 for name in names}
        retained_water = {name: 0.0 for name in names}
        for i in valid_indices:
            r, c, length = rows[i], cols[i], lengths[i]
            if tile.eligible[r, c]:
                supported += length
                weighted_score += length * float(probability[r, c])
                if flags[r, c] != 0:
                    flagged_m += length
                if retained[r, c]:
                    retained_m += length
                for band, name in enumerate(names):
                    if cases[band, r, c] == 1:
                        water[name] += length
                        if retained[r, c]:
                            retained_water[name] += length
        result[segment_id] = {"segment_id": segment_id, "within_tile_m": inside_m,
                              "outside_tile_m": max(0.0, edge["length_m"] - inside_m),
                              "source_supported_m": supported, "retained_m": retained_m,
                              "source_unknown_m": max(0.0, edge["length_m"] - supported),
                              "coverage_fraction_of_segment": min(1.0, supported / edge["length_m"]),
                              "unknown_fraction_of_segment": max(0.0, 1 - supported / edge["length_m"]),
                              "retained_fraction_of_supported": retained_m / supported if supported else None,
                              "quality_flagged_m": flagged_m,
                              "mean_score_on_supported": weighted_score / supported if supported else None,
                              "source_supported_water_m_by_scenario": water,
                              "retained_water_m_by_scenario": retained_water,
                              "bridge": edge["bridge"], "tunnel": edge["tunnel"],
                              "road_class": edge["highway"],
                              "evidence_source": "Spain_7370579 optical 2019-09-18; radar 2019-09-17",
                              "interpretation": "observed-water overlap only; passability unknown"}
    return result


def crossing_audit(segments: dict[str, dict], nodes: dict[int, tuple],
                   evidence: dict[str, dict], *, sample_limit: int = 10) -> dict:
    """Find apparent crossings; graph still connects only at shared OSM IDs."""
    ids = sorted(evidence)
    lines = [LineString(((nodes[segments[key]["source"]][0], nodes[segments[key]["source"]][1]),
                         (nodes[segments[key]["target"]][0], nodes[segments[key]["target"]][1])))
             for key in ids]
    tree = STRtree(lines)
    unconnected = tagged_grade = untagged_same_layer = 0
    examples = []
    for index, line in enumerate(lines):
        for other in tree.query(line, predicate="crosses"):
            other = int(other)
            if other <= index:
                continue
            left, right = segments[ids[index]], segments[ids[other]]
            if {left["source"], left["target"]} & {right["source"], right["target"]}:
                continue
            unconnected += 1
            tagged = (left["bridge"] or right["bridge"] or left["tunnel"] or right["tunnel"] or
                      left["layer"] != right["layer"])
            tagged_grade += tagged
            untagged_same_layer += not tagged
            if len(examples) < sample_limit:
                point = line.intersection(lines[other])
                examples.append({"first": ids[index], "second": ids[other],
                                 "x_m": point.centroid.x, "y_m": point.centroid.y,
                                 "grade_tag_evidence": bool(tagged)})
    return {"unconnected_geometric_crossings": unconnected,
            "with_bridge_tunnel_or_layer_difference": tagged_grade,
            "same_layer_without_grade_tag": untagged_same_layer,
            "examples": examples,
            "interpretation": "No junction is invented; same-layer unconnected crosses need local inspection."}
