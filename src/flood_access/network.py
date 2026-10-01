"""Synthetic graph scenario engine. Edge states are assumptions, not closures."""
from dataclasses import dataclass
import heapq
import math

@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    cost: float
    state: str = "clear"

    def __post_init__(self):
        if not self.source or not self.target:
            raise ValueError("Edges require nonempty endpoints")
        if not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("Costs must be finite and nonnegative")
        if self.state not in {"clear", "exposed", "unknown"}:
            raise ValueError("Invalid evidence state")

def shortest_cost(edges: list[Edge], origin: str, facility: str,
                  scenario: str = "baseline") -> float | None:
    if scenario not in {"baseline", "exposed_only", "conservative"}:
        raise ValueError("Unknown scenario")
    nodes = {node for edge in edges for node in (edge.source, edge.target)}
    if origin not in nodes or facility not in nodes:
        raise ValueError("Origin and facility must exist in the baseline graph")
    graph: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        disabled = scenario != "baseline" and edge.state == "exposed"
        disabled |= scenario == "conservative" and edge.state == "unknown"
        if not disabled:
            graph.setdefault(edge.source, []).append((edge.target, edge.cost))
    best = {origin: 0.0}
    queue = [(0.0, origin)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best.get(node):
            continue
        if node == facility:
            return cost
        for target, weight in graph.get(node, []):
            candidate = cost + weight
            if candidate < best.get(target, math.inf):
                best[target] = candidate
                heapq.heappush(queue, (candidate, target))
    return None

def synthetic_demo(scenario: str) -> dict:
    edges = [Edge("A", "H", 10, "exposed"), Edge("A", "B", 8),
             Edge("B", "H", 9, "unknown"), Edge("C", "H", 6)]
    rows = []
    for origin in ("A", "B", "C"):
        base = shortest_cost(edges, origin, "H")
        changed = shortest_cost(edges, origin, "H", scenario)
        rows.append({"origin": origin, "baseline_cost": base,
                     "scenario_cost": changed,
                     "newly_disconnected": base is not None and changed is None})
    return {"data_mode": "synthetic", "scenario": scenario,
            "cost_unit": "invented_units", "results": rows,
            "interpretation": "Assumed edge disruption; no real flood or safe-route claim"}
