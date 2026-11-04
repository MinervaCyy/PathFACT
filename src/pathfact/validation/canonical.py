"""Historical G00 shortest-path record validator.

Scientific validation logic preserved from the original source.
"""

import networkx as nx


def validate_case(G, record, hop_low, hop_high):
    assert nx.is_connected(G)

    src, dst = record["node_pair"]
    path = record["ground_truth_shortest_path"]

    assert path[0] == src
    assert path[-1] == dst
    assert len(path) == len(set(path))

    path_cost = 0

    for u, v in zip(path, path[1:]):
        assert G.has_edge(u, v)
        path_cost += int(G[u][v]["weight"])

    true_cost = nx.shortest_path_length(
        G,
        src,
        dst,
        weight="weight",
    )

    assert path_cost == true_cost
    assert true_cost == record["gold_cost"]
    assert record["gold_cost"] == record["min_total_cost"]

    assert (
        hop_low
        <= record["min_optimal_hops"]
        <= record["max_optimal_hops"]
        <= hop_high
    )

    assert (
        record["gold_path_hops"]
        == len(record["ground_truth_shortest_path"]) - 1
    )

    assert (
        record["gold_path_hops"]
        == record["min_optimal_hops"]
    )
