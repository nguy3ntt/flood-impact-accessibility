"""Auditable historical OSM road graph for retrospective scenario analysis.

Only shared OSM node IDs connect ways. Geometric line crossings create no junction.
Travel costs are static class/tag estimates, never observed flood travel times.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import math
import re

from rasterio.warp import transform


NO_MOTOR_ACCESS = {"no", "private", "agricultural", "forestry"}
ONEWAY_FORWARD = {"yes", "1", "true"}
ONEWAY_REVERSE = {"-1", "reverse"}
ONEWAY_BOTH = {"no", "0", "false"}


def travel_direction(tags: dict) -> tuple[int | None, str]:
    """Return +1, -1, 0 (two-way), or None when direction is unsupported."""
    value = tags.get("oneway")
    if value in ONEWAY_FORWARD:
        return 1, "explicit_forward"
    if value in ONEWAY_REVERSE:
        return -1, "explicit_reverse"
    if value in ONEWAY_BOTH:
        return 0, "explicit_two_way"
    if value is not None:
        return None, f"unsupported_oneway:{value}"
    if tags.get("junction") == "roundabout" or tags.get("highway") in {"motorway", "motorway_link"}:
        return 1, "inferred_one_way"
    return 0, "assumed_two_way_unmarked"


def road_speed_kmh(tags: dict, defaults: dict[str, float]) -> tuple[float, str]:
    fallback = float(defaults[tags["highway"]])
    raw = tags.get("maxspeed", "")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kph)?\s*", raw.lower())
    if match:
        speed = float(match.group(1)) * (1.609344 if match.group(2) == "mph" else 1)
        if 5 <= speed <= 140:
            return speed, "explicit_maxspeed"
    return fallback, "class_assumption"


def build_graph(elements: list[dict], defaults: dict[str, float],
                road_snapshot_utc: str) -> tuple[dict[int, tuple], list[dict], dict]:
    source_nodes = {int(row["id"]): row for row in elements if row.get("type") == "node"}
    ways = [row for row in elements if row.get("type") == "way"]
    if not ways or len(source_nodes) != len([row for row in elements if row.get("type") == "node"]):
        raise ValueError("Empty graph or duplicate source node ID")
    used = {int(node) for way in ways for node in way["nodes"]}
    if not used <= set(source_nodes):
        raise ValueError("OSM way references a missing node")
    ids = sorted(used)
    lon = [float(source_nodes[node]["lon"]) for node in ids]
    lat = [float(source_nodes[node]["lat"]) for node in ids]
    if not all(math.isfinite(a) and math.isfinite(b) and -180 <= a <= 180 and -90 <= b <= 90
               for a, b in zip(lon, lat)):
        raise ValueError("Invalid OSM source coordinates")
    xs, ys = transform("EPSG:4326", "EPSG:25830", lon, lat)
    nodes = {node: (float(x), float(y), a, b) for node, x, y, a, b in zip(ids, xs, ys, lon, lat)}
    edges = []
    rejected = Counter()
    direction_counts = Counter()
    speed_counts = Counter()
    bridge_ways = tunnel_ways = 0
    for way in ways:
        tags = way.get("tags", {})
        highway = tags.get("highway")
        if highway not in defaults:
            rejected["class_outside_contract"] += 1
            continue
        if tags.get("area") == "yes" or any(tags.get(key) in NO_MOTOR_ACCESS
                                            for key in ("access", "vehicle", "motor_vehicle", "motorcar")):
            rejected["motor_access_restricted_or_area"] += 1
            continue
        direction, source = travel_direction(tags)
        if direction is None:
            rejected["unsupported_direction"] += 1
            continue
        speed, speed_source = road_speed_kmh(tags, defaults)
        direction_counts[source] += 1
        speed_counts[speed_source] += 1
        bridge = tags.get("bridge") not in (None, "no", "0", "false")
        tunnel = tags.get("tunnel") not in (None, "no", "0", "false")
        bridge_ways += bridge
        tunnel_ways += tunnel
        layer_raw = tags.get("layer", "0")
        try:
            layer = int(layer_raw)
        except (ValueError, TypeError):
            layer = None
        way_nodes = [int(node) for node in way["nodes"]]
        for index, (a, b) in enumerate(zip(way_nodes, way_nodes[1:])):
            if a == b:
                rejected["zero_node_step"] += 1
                continue
            length = math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
            if not math.isfinite(length) or length < 0.01:
                rejected["near_zero_length_step"] += 1
                continue
            segment_id = f"osm:{way['id']}:{index}"
            for start, end in ((a, b), (b, a)) if direction == 0 else \
                              (((a, b),) if direction == 1 else ((b, a),)):
                edges.append({"edge_id": f"{segment_id}:{start}>{end}", "segment_id": segment_id,
                              "source": start, "target": end, "way_id": int(way["id"]),
                              "highway": highway, "direction_source": source,
                              "length_m": length, "speed_kmh": speed, "speed_source": speed_source,
                              "cost_seconds": length / speed * 3.6, "bridge": bridge,
                              "tunnel": tunnel, "layer": layer,
                              "road_snapshot_utc": road_snapshot_utc})
    if not edges:
        raise ValueError("No usable directed road edges")
    graph_nodes = {node for edge in edges for node in (edge["source"], edge["target"])}
    nodes = {node: nodes[node] for node in graph_nodes}
    parent = {node: node for node in nodes}
    size = {node: 1 for node in nodes}

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for edge in edges:
        left, right = root(edge["source"]), root(edge["target"])
        if left != right:
            if size[left] < size[right]:
                left, right = right, left
            parent[right] = left
            size[left] += size[right]
    components = Counter(root(node) for node in nodes)
    component_by_node = {node: root(node) for node in nodes}
    report = {"source_ways": len(ways), "source_nodes": len(source_nodes),
              "graph_nodes": len(nodes), "directed_edges": len(edges),
              "physical_segments": len({edge["segment_id"] for edge in edges}),
              "weak_components": len(components), "largest_weak_component_nodes": max(components.values()),
              "component_by_node": component_by_node,
              "rejected": dict(rejected), "direction_ways": dict(direction_counts),
              "speed_ways": dict(speed_counts), "bridge_ways": bridge_ways,
              "tunnel_ways": tunnel_ways,
              "total_directed_length_m": sum(edge["length_m"] for edge in edges),
              "topology_rule": "shared OSM node ID only; line crossings do not create junctions"}
    return nodes, edges, report


def shortest_to_facilities(edges: list[dict], facility_nodes: dict[str, int],
                           closed_segments: set[str]) -> tuple[dict[int, float], dict[int, str], dict[int, str]]:
    """Reverse multisource Dijkstra; same directed edge set for all scenarios."""
    if not facility_nodes:
        raise ValueError("At least one snapped facility is required")
    incoming: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
    graph_nodes = set()
    for edge in edges:
        if not math.isfinite(edge["cost_seconds"]) or edge["cost_seconds"] < 0:
            raise ValueError("Non-finite or negative edge cost")
        graph_nodes.update((edge["source"], edge["target"]))
        if edge["segment_id"] not in closed_segments:
            incoming[edge["target"]].append((edge["source"], edge["cost_seconds"], edge["edge_id"]))
    if not set(facility_nodes.values()) <= graph_nodes:
        raise ValueError("Facility lies outside baseline graph")
    distances = {}
    closest = {}
    next_edge = {}
    queue = []
    for facility, node in sorted(facility_nodes.items()):
        if node not in distances or facility < closest[node]:
            distances[node], closest[node] = 0.0, facility
            heapq.heappush(queue, (0.0, facility, node))
    while queue:
        cost, facility, node = heapq.heappop(queue)
        if cost != distances.get(node) or facility != closest.get(node):
            continue
        for source, weight, edge_id in incoming.get(node, ()):
            candidate = cost + weight
            if candidate < distances.get(source, math.inf) - 1e-9 or \
               (abs(candidate - distances.get(source, math.inf)) <= 1e-9 and
                facility < closest.get(source, "~")):
                distances[source], closest[source], next_edge[source] = candidate, facility, edge_id
                heapq.heappush(queue, (candidate, facility, source))
    return distances, closest, next_edge
