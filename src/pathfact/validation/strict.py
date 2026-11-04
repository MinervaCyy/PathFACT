"""Independent CPU-only validation of candidate shortest paths.

This is a new public validation API, separate from the
preserved historical validate_case() implementation.

Graphs must be simple, undirected and have strictly positive,
finite numeric edge weights. Nodes and candidate paths use
integer identifiers.

Optimality uses absolute cost tolerance 1e-9.
"""

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real

import networkx as nx

from pathfact.solver import shortest_path_hop_bounds_from_source


@dataclass(frozen=True)
class PathValidationResult:
    feasible: bool
    optimal: bool
    path_cost: int | float | None
    optimal_cost: int | float | None
    path_hops: int | None
    min_optimal_hops: int | None
    max_optimal_hops: int | None
    reasons: tuple[str, ...]


def validate_candidate_path(
    graph: nx.Graph,
    source: int,
    destination: int,
    path: list[int] | tuple[int, ...] | None,
) -> PathValidationResult:
    """Validate feasibility and optimality independently.

    Invalid candidate paths return structured reasons.
    Unsupported graph types or weights raise ValueError.
    """

    if not isinstance(graph, nx.Graph):
        raise ValueError("Expected a NetworkX graph")

    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("Expected a simple undirected graph")

    for _, _, attributes in graph.edges(data=True):
        weight = attributes.get("weight")

        if (
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or not isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(
                "All edges require positive finite numeric weights"
            )

    def result(
        feasible=False,
        optimal=False,
        path_cost=None,
        optimal_cost=None,
        path_hops=None,
        min_hops=None,
        max_hops=None,
        reasons=(),
    ):
        return PathValidationResult(
            feasible=feasible,
            optimal=optimal,
            path_cost=path_cost,
            optimal_cost=optimal_cost,
            path_hops=path_hops,
            min_optimal_hops=min_hops,
            max_optimal_hops=max_hops,
            reasons=tuple(reasons),
        )

    if source not in graph or destination not in graph:
        return result(reasons=("unknown_endpoint",))

    dist, min_hops, max_hops, _ = (
        shortest_path_hop_bounds_from_source(graph, source)
    )

    if destination not in dist:
        return result(reasons=("unreachable_destination",))

    optimum = dist[destination]
    minimum_hops = min_hops[destination]
    maximum_hops = max_hops[destination]

    common = {
        "optimal_cost": optimum,
        "min_hops": minimum_hops,
        "max_hops": maximum_hops,
    }

    if path is None:
        return result(reasons=("missing_path",), **common)

    if not isinstance(path, (list, tuple)):
        return result(reasons=("invalid_path_type",), **common)

    if not path:
        return result(reasons=("empty_path",), **common)

    if any(
        not isinstance(node, int) or isinstance(node, bool)
        for node in path
    ):
        return result(reasons=("invalid_node_type",), **common)

    reasons = []

    if path[0] != source:
        reasons.append("wrong_source")

    if path[-1] != destination:
        reasons.append("wrong_destination")

    if len(path) != len(set(path)):
        reasons.append("repeated_node")

    if any(
        not graph.has_edge(u, v)
        for u, v in zip(path, path[1:])
    ):
        reasons.append("missing_edge")

    hops = len(path) - 1

    if reasons:
        return result(
            path_hops=hops,
            reasons=reasons,
            **common,
        )

    cost = sum(
        graph[u][v]["weight"]
        for u, v in zip(path, path[1:])
    )

    optimal = isclose(
        cost,
        optimum,
        rel_tol=0.0,
        abs_tol=1e-9,
    )

    if not optimal:
        reasons.append("non_optimal")

    if optimal and not (
        minimum_hops <= hops <= maximum_hops
    ):
        optimal = False
        reasons.append("optimal_hop_bounds_mismatch")

    return result(
        feasible=True,
        optimal=optimal,
        path_cost=cost,
        path_hops=hops,
        reasons=reasons,
        **common,
    )
