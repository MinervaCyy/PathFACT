"""Historical shortest-path solving and task-selection algorithms.

Extracted from generate_region_conditioned_pilot.py.
Scientific logic preserved without modification.
Canonical parity remains subject to regression testing.
"""

import math
import random

import networkx as nx


def shortest_path_hop_bounds_from_source(G, src):
    dist = nx.single_source_dijkstra_path_length(
        G,
        src,
        weight="weight",
    )

    ordered = sorted(
        dist,
        key=lambda node: (dist[node], int(node)),
    )

    min_hops = {src: 0}
    max_hops = {src: 0}
    canonical_pred = {src: None}

    for v in ordered:
        if v == src:
            continue

        parents = []

        for u in G.neighbors(v):
            if u not in dist:
                continue

            weight = G[u][v]["weight"]

            if math.isclose(
                dist[u] + weight,
                dist[v],
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                parents.append(u)

        if not parents:
            raise RuntimeError(
                f"No shortest-path predecessor for source={src}, node={v}"
            )

        min_hops[v] = min(min_hops[u] + 1 for u in parents)
        max_hops[v] = max(max_hops[u] + 1 for u in parents)

        canonical_pred[v] = min(
            parents,
            key=lambda u: (
                min_hops[u] + 1,
                int(u),
            ),
        )

    return dist, min_hops, max_hops, canonical_pred


def reconstruct_path(pred, src, dst):
    current = dst
    path = [current]

    while current != src:
        current = pred[current]

        if current is None:
            raise RuntimeError(
                f"Broken predecessor chain for {src}->{dst}"
            )

        path.append(current)

    path.reverse()
    return path


def eligible_pairs_by_exact_hop(G, hop_low, hop_high):
    strata = {}
    nodes = sorted(int(node) for node in G.nodes())

    for index, src in enumerate(nodes):
        dist, min_hops, max_hops, pred = (
            shortest_path_hop_bounds_from_source(G, src)
        )

        for dst in nodes[index + 1:]:
            min_h = int(min_hops[dst])
            max_h = int(max_hops[dst])

            if min_h < hop_low or max_h > hop_high:
                continue

            path = reconstruct_path(pred, src, dst)

            item = {
                "u": int(src),
                "v": int(dst),
                "min_cost": int(dist[dst]),
                "min_optimal_hops": min_h,
                "max_optimal_hops": max_h,
                "canonical_path_uv": [int(x) for x in path],
            }

            strata.setdefault(min_h, []).append(item)

    return strata


def select_pair_exact_hop_aware(strata, pair_seed):
    rng = random.Random(pair_seed)

    feasible_hops = sorted(strata)
    selected_exact_hop = feasible_hops[
        rng.randrange(len(feasible_hops))
    ]

    candidates = strata[selected_exact_hop]
    selected = dict(candidates[rng.randrange(len(candidates))])

    swapped = bool(rng.randrange(2))

    if not swapped:
        src = selected["u"]
        dst = selected["v"]
        path = selected["canonical_path_uv"]
    else:
        src = selected["v"]
        dst = selected["u"]
        path = list(reversed(selected["canonical_path_uv"]))

    selected.update(
        {
            "src": int(src),
            "dst": int(dst),
            "canonical_path": [int(x) for x in path],
            "selected_exact_hop": int(selected_exact_hop),
            "pair_orientation_swapped": swapped,
            "eligible_pair_count_selected_exact_hop": int(
                len(candidates)
            ),
            "eligible_pair_count_total": int(
                sum(len(items) for items in strata.values())
            ),
            "feasible_exact_hops": [
                int(x) for x in feasible_hops
            ],
        }
    )

    return selected
