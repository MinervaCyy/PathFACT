"""Historical graph generation and provenance functions.

Extracted from generate_region_conditioned_pilot.py.
Preserves the original scientific algorithms.
Historical parity remains subject to regression testing.
"""

import ast
import hashlib
import json
import random

import networkx as nx
import numpy as np


def stable_seed(*parts):
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32)


def parse_dimension(x):
    if isinstance(x, dict):
        return x
    return ast.literal_eval(str(x))


def parse_topology(x):
    obj = ast.literal_eval(str(x))
    out = {}
    for k, vals in obj.items():
        u = int(k)
        out[u] = [(int(v), int(w)) for v, w in vals]
    return out


def graph_from_topology_dict(topo):
    G = nx.Graph()
    for u, vals in topo.items():
        G.add_node(int(u))
        for v, w in vals:
            G.add_edge(int(u), int(v), weight=int(w))
    return G


def canonical_unweighted_edges(G):
    return sorted(
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in G.edges()
    )


def canonical_weighted_edges(G):
    return sorted(
        (
            min(int(u), int(v)),
            max(int(u), int(v)),
            int(G[u][v]["weight"]),
        )
        for u, v in G.edges()
    )


def sha256_json(obj):
    blob = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def graph_structure_sha256(G):
    return sha256_json(
        {
            "num_nodes": G.number_of_nodes(),
            "edges": canonical_unweighted_edges(G),
        }
    )


def weighted_graph_sha256(G):
    return sha256_json(
        {
            "num_nodes": G.number_of_nodes(),
            "edges": canonical_weighted_edges(G),
        }
    )


def case_instance_sha256(G, src, dst):
    return sha256_json(
        {
            "weighted_graph_sha256": weighted_graph_sha256(G),
            "source": int(src),
            "destination": int(dst),
        }
    )


def generate_jellyfish_graph(n, max_degree, graph_seed):
    rng = random.Random(graph_seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    while True:
        nodes = list(G.nodes())
        rng.shuffle(nodes)
        added = False

        for i in nodes:
            if G.degree[i] >= max_degree:
                continue

            for j in nodes:
                if (
                    i != j
                    and G.degree[j] < max_degree
                    and not G.has_edge(i, j)
                ):
                    G.add_edge(i, j)
                    added = True
                    break

        if not added:
            break

    return G


def generate_graph(topology_type, dimension, graph_seed):
    if topology_type == "barabasi_albert":
        return nx.barabasi_albert_graph(
            n=int(dimension["n"]),
            m=int(dimension["m"]),
            seed=graph_seed,
        )

    if topology_type == "erdos_renyi":
        return nx.erdos_renyi_graph(
            n=int(dimension["n"]),
            p=float(dimension["p"]),
            seed=graph_seed,
        )

    if topology_type == "grid":
        n = int(dimension["n"])
        G = nx.grid_2d_graph(n, n)
        return nx.convert_node_labels_to_integers(G)

    if topology_type == "fat_tree":
        return nx.balanced_tree(
            r=int(dimension["r"]),
            h=int(dimension["h"]),
        )

    if topology_type == "jellyfish":
        return generate_jellyfish_graph(
            n=int(dimension["n"]),
            max_degree=int(dimension["max_degree"]),
            graph_seed=graph_seed,
        )

    raise ValueError(f"Unknown topology_type: {topology_type}")


def assign_weights(G, mode, weight_seed):
    G = G.copy()
    py_rng = random.Random(weight_seed)
    np_rng = np.random.default_rng(weight_seed)

    for u, v in sorted(G.edges()):
        if mode == "uniform":
            weight = py_rng.randint(50, 200)

        elif mode == "discrete":
            weight = py_rng.choice([100, 1000, 10000])

        elif mode == "lognormal":
            weight = int(np_rng.lognormal(mean=5, sigma=1))
            weight = max(1, min(weight, 10000))

        else:
            raise ValueError(f"Unknown weight mode: {mode}")

        G[u][v]["weight"] = int(weight)

    return G


def topology_to_adjacency(G):
    return {
        int(u): [
            (
                int(v),
                int(G[u][v]["weight"]),
            )
            for v in sorted(G.neighbors(u))
        ]
        for u in sorted(G.nodes())
    }
