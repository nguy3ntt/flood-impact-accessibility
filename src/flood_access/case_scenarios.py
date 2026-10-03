"""Explicit road-disruption assumptions and paired access summaries."""

from __future__ import annotations

import math

from .case_graph import shortest_to_facilities


POLICIES = {"supported_overlap", "retained_overlap", "supported_plus_within_tile_unknown",
            "supported_plus_all_unknown"}
BRIDGE_POLICIES = {"bridge_exempt", "bridge_candidate"}


def closed_segments(segments: dict[str, dict], evidence: dict[str, dict], scenario: str, *,
                    policy: str, bridge_policy: str, minimum_exposed_m: float) -> tuple[set[str], dict]:
    if policy not in POLICIES or bridge_policy not in BRIDGE_POLICIES or minimum_exposed_m <= 0:
        raise ValueError("Unknown road-disruption assumption")
    closed = set()
    reasons = {"supported_overlap": 0, "retained_overlap": 0,
               "within_tile_unknown": 0, "outside_or_partial_unknown": 0}
    for segment_id, edge in segments.items():
        row = evidence.get(segment_id)
        supported_water = row["source_supported_water_m_by_scenario"][scenario] if row else 0.0
        retained_water = row["retained_water_m_by_scenario"][scenario] if row else 0.0
        observed = retained_water if policy == "retained_overlap" else supported_water
        closure_reason = None
        if observed >= minimum_exposed_m and not edge["tunnel"] and \
           (not edge["bridge"] or bridge_policy == "bridge_candidate"):
            closure_reason = "retained_overlap" if policy == "retained_overlap" else "supported_overlap"
        elif policy == "supported_plus_within_tile_unknown" and row and row["within_tile_m"] > 0 and \
             row["retained_m"] < row["within_tile_m"] - 1e-6:
            closure_reason = "within_tile_unknown"
        elif policy == "supported_plus_all_unknown" and (not row or
             row["retained_m"] < edge["length_m"] - 1e-6):
            closure_reason = "outside_or_partial_unknown"
        if closure_reason:
            closed.add(segment_id)
            reasons[closure_reason] += 1
    return closed, {"closed_physical_segments": len(closed), "reasons": reasons,
                    "assumption": "candidate segment removed from simulated graph; not an observed closure"}


def access_rows(edges: list[dict], facilities: dict[str, int], origins: dict[str, int],
                closed: set[str], baseline: dict[str, dict] | None = None) -> list[dict]:
    distances, closest, _ = shortest_to_facilities(edges, facilities, closed)
    rows = []
    for origin, node in sorted(origins.items()):
        cost = distances.get(node)
        base = baseline.get(origin) if baseline else None
        base_cost = base["estimated_cost_seconds"] if base else cost
        rows.append({"origin_id": origin, "nearest_facility_id": closest.get(node),
                     "estimated_cost_seconds": cost,
                     "baseline_cost_seconds": base_cost,
                     "cost_delta_seconds": cost - base_cost if cost is not None and base_cost is not None else None,
                     "baseline_unreachable": base_cost is None,
                     "newly_disconnected": base_cost is not None and cost is None,
                     "reason": "no_graph_path" if cost is None else None})
    return rows


def ranking_reversals(baseline: list[dict], scenario: list[dict]) -> dict:
    base = {row["origin_id"]: row["estimated_cost_seconds"] for row in baseline}
    changed = {row["origin_id"]: row["estimated_cost_seconds"] for row in scenario}
    ids = sorted(base)
    pairs = reversals = 0
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            a, b, c, d = base[left], base[right], changed[left], changed[right]
            if any(value is None for value in (a, b, c, d)):
                continue
            if math.isclose(a, b) or math.isclose(c, d):
                continue
            pairs += 1
            reversals += (a < b) != (c < d)
    return {"comparable_pairs": pairs, "reversed_pairs": reversals,
            "reversal_fraction": reversals / pairs if pairs else None}


def summarise_access(baseline: list[dict], changed: list[dict]) -> dict:
    if [row["origin_id"] for row in baseline] != [row["origin_id"] for row in changed]:
        raise ValueError("Access origin support differs")
    paired = [row for row in changed if row["baseline_cost_seconds"] is not None and
              row["estimated_cost_seconds"] is not None]
    increases = [row["cost_delta_seconds"] for row in paired]
    if any(value < -1e-7 for value in increases):
        raise ValueError("Edge removal improved shortest cost")
    return {"origins": len(changed),
            "baseline_unreachable": sum(row["baseline_unreachable"] for row in changed),
            "newly_disconnected": sum(row["newly_disconnected"] for row in changed),
            "paired_reachable": len(paired),
            "mean_paired_cost_increase_seconds": sum(increases) / len(increases) if increases else None,
            "maximum_paired_cost_increase_seconds": max(increases) if increases else None,
            "ranking": ranking_reversals(baseline, changed)}
