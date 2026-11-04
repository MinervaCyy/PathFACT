"""Portable CPU tests for PathFACT graph and solver functions."""

import unittest

import networkx as nx

from pathfact.graphs import (
    assign_weights,
    generate_graph,
    graph_from_topology_dict,
    graph_structure_sha256,
    stable_seed,
    topology_to_adjacency,
    weighted_graph_sha256,
)
from pathfact.solver import (
    eligible_pairs_by_exact_hop,
    reconstruct_path,
    select_pair_exact_hop_aware,
    shortest_path_hop_bounds_from_source,
)


class TestM00Core(unittest.TestCase):

    def test_stable_seed(self):
        seed = stable_seed("pathfact", "graph", 42)

        self.assertEqual(
            seed,
            stable_seed("pathfact", "graph", 42),
        )
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32)

    def test_graph_generation_determinism(self):
        configurations = [
            ("barabasi_albert", {"n": 12, "m": 2}),
            ("erdos_renyi", {"n": 12, "p": 0.5}),
            ("grid", {"n": 3}),
            ("fat_tree", {"r": 2, "h": 2}),
            ("jellyfish", {"n": 12, "max_degree": 3}),
        ]

        for topology, dimension in configurations:
            for distribution in (
                "uniform",
                "discrete",
                "lognormal",
            ):
                with self.subTest(
                    topology=topology,
                    distribution=distribution,
                ):
                    first = assign_weights(
                        generate_graph(topology, dimension, 1234),
                        distribution,
                        5678,
                    )
                    second = assign_weights(
                        generate_graph(topology, dimension, 1234),
                        distribution,
                        5678,
                    )

                    self.assertEqual(
                        weighted_graph_sha256(first),
                        weighted_graph_sha256(second),
                    )

    def test_graph_serialization_roundtrip(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=5)
        graph.add_edge(1, 2, weight=3)
        graph.add_edge(0, 2, weight=8)

        adjacency = topology_to_adjacency(graph)
        reconstructed = graph_from_topology_dict(adjacency)

        self.assertEqual(
            graph_structure_sha256(graph),
            graph_structure_sha256(reconstructed),
        )
        self.assertEqual(
            weighted_graph_sha256(graph),
            weighted_graph_sha256(reconstructed),
        )

    def test_shortest_path_hop_ties(self):
        graph = nx.Graph()
        graph.add_weighted_edges_from([
            (0, 3, 2),
            (0, 1, 1),
            (1, 3, 1),
            (0, 2, 1),
            (2, 3, 1),
        ])

        dist, min_hops, max_hops, pred = (
            shortest_path_hop_bounds_from_source(graph, 0)
        )

        self.assertEqual(dist[3], 2)
        self.assertEqual(min_hops[3], 1)
        self.assertEqual(max_hops[3], 2)
        self.assertEqual(
            reconstruct_path(pred, 0, 3),
            [0, 3],
        )

        # Remove the direct route: two optimal two-hop
        # routes remain, so node-ID tie-breaking applies.
        graph.remove_edge(0, 3)

        dist, min_hops, max_hops, pred = (
            shortest_path_hop_bounds_from_source(graph, 0)
        )

        self.assertEqual(dist[3], 2)
        self.assertEqual(min_hops[3], 2)
        self.assertEqual(max_hops[3], 2)
        self.assertEqual(
            reconstruct_path(pred, 0, 3),
            [0, 1, 3],
        )

    def test_pair_selection_determinism(self):
        graph = assign_weights(
            generate_graph("grid", {"n": 3}, 1234),
            "uniform",
            5678,
        )

        strata = eligible_pairs_by_exact_hop(graph, 1, 4)

        first = select_pair_exact_hop_aware(strata, 13579)
        second = select_pair_exact_hop_aware(strata, 13579)

        self.assertEqual(first, second)
        self.assertEqual(
            first["canonical_path"][0],
            first["src"],
        )
        self.assertEqual(
            first["canonical_path"][-1],
            first["dst"],
        )
        self.assertEqual(
            first["min_cost"],
            nx.shortest_path_length(
                graph,
                first["src"],
                first["dst"],
                weight="weight",
            ),
        )


if __name__ == "__main__":
    unittest.main()
