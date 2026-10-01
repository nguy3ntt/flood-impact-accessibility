# Canonical data contracts

Implement schemas after M1 confirms real source fields. These are target contracts, not claims about existing files. Use UTC acquisition timestamps and explicit local-event context. All geometries carry CRS; metre-based operations use an appropriate local projected CRS, not degree distances.

## Source manifest
`source_id`, `dataset/version`, `source_url`, `retrieved_at_utc`, `original_filename`, `bytes`, `sha256`, `licence_url`, `licence_status`, `attribution`, `redistribution_status`, `event/date coverage`, `access_method`, `notes`. Use relative logical paths; do not publish credentials or personal absolute paths.

## Raster catalog
`tile_id`, `event_id`, `sensor`, `product/version`, `source_id`, `acquired_at_utc`, `crs`, `affine_transform`, `bounds`, `width/height`, `band_names/order`, `units`, `scale/offset`, `nodata`, `valid_mask_ref`, `cloud_mask_ref`, `label_ref`, `label_semantics`, `split_id`, `content_hash`, `processing_revision`.
Validate paired footprints and alignment. Use nearest-neighbour for categorical labels/masks; document continuous-band resampling. Never fill missing labels as background. Audit overlap across chips and event splits. Optical/radar temporal offsets are features of evidence quality, not silently ignored.

## Road graph
Nodes: `node_id`, coordinates, CRS, component ID. Edges: `edge_id`, `from/to`, geometry, directedness, length_m, cost_seconds, speed-source/assumption, bridge/tunnel/layer flags, road class, snapshot_date, source_id, topology_quality. Prevent false intersection connections at grade-separated crossings. Missing topology/direction must be surfaced.
Facilities: `facility_id`, type, point, source/date, snapped_node, snap_distance_m, verified_status. Origins: `origin_id`, point, source/definition, optional population/date, snap evidence. Do not silently snap across disconnected components or beyond a declared maximum distance. AOI buffer must permit relevant paths; boundary truncation is not flood-induced isolation.

## Mapping and road evidence
Raster outputs: score, thresholded water, valid/unknown, permanent-water context where supported, model/run reference. Road evidence: `edge_id`, `event_id`, exposed_length_m, supported_length_m, water_score_summary, acquisition/date offset, coverage_fraction, unknown_fraction, grade_separation_status, evidence_refs, quality_flags`.
An exposed edge has no implicit closure probability. Calibrated pixel scores are not calibrated edge-closure probabilities.

## Scenario and accessibility bundle
Scenario: `scenario_id`, evidence/model versions, disruption rule, unknown policy, bridge policy, spatial-coherence method, assumptions, reproducibility seed. Results: `origin_id`, baseline/scenario reachable, nearest_facility_id, estimated_cost_seconds, cost_delta where both reachable, baseline_unreachable, newly_disconnected, scenario_set_reachability_fraction and explicit interpretation.
Missing routes return null costs and reason codes, never zero. Scenario frequency is not a population probability without justified sampling. Bundles include manifest checksums and supported area/event dates.
