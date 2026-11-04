"""Tests for the public shortest-path validation API."""

import json
import os
import unittest
from pathlib import Path

import networkx as nx

from pathfact.graphs import graph_from_topology_dict
from pathfact.validation import validate_candidate_path


class TestStrictValidation(unittest.TestCase):

    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from([
            (0, 1, 2),
            (1, 2, 3),
            (0, 2, 10),
        ])

    def test_optimal_path(self):
        result = validate_candidate_path(
            self.graph, 0, 2, [0, 1, 2]
        )

        self.assertTrue(result.feasible)
        self.assertTrue(result.optimal)
        self.assertEqual(result.path_cost, 5)
        self.assertEqual(result.optimal_cost, 5)
        self.assertEqual(result.path_hops, 2)
        self.assertEqual(result.min_optimal_hops, 2)
        self.assertEqual(result.max_optimal_hops, 2)
        self.assertEqual(result.reasons, ())

    def test_feasible_nonoptimal_path(self):
        result = validate_candidate_path(
            self.graph, 0, 2, [0, 2]
        )

        self.assertTrue(result.feasible)
        self.assertFalse(result.optimal)
        self.assertEqual(result.path_cost, 10)
        self.assertEqual(result.optimal_cost, 5)
        self.assertEqual(result.reasons, ("non_optimal",))

    def test_invalid_candidate_paths(self):
        cases = [
            (None, "missing_path"),
            (42, "invalid_path_type"),
            ([], "empty_path"),
            ([0, True, 2], "invalid_node_type"),
            ([1, 2], "wrong_source"),
            ([0, 1], "wrong_destination"),
            ([0, 1, 0, 1, 2], "repeated_node"),
            ([0, 9, 2], "missing_edge"),
        ]

        for candidate, expected_reason in cases:
            with self.subTest(candidate=candidate):
                result = validate_candidate_path(
                    self.graph, 0, 2, candidate
                )

                self.assertFalse(result.feasible)
                self.assertFalse(result.optimal)
                self.assertIn(
                    expected_reason, result.reasons
                )

    def test_optimal_hop_bounds(self):
        graph = nx.Graph()
        graph.add_weighted_edges_from([
            (0, 3, 2),
            (0, 1, 1),
            (1, 3, 1),
        ])

        for candidate in ([0, 3], [0, 1, 3]):
            with self.subTest(candidate=candidate):
                result = validate_candidate_path(
                    graph, 0, 3, candidate
                )

                self.assertTrue(result.optimal)
                self.assertEqual(result.optimal_cost, 2)
                self.assertEqual(result.min_optimal_hops, 1)
                self.assertEqual(result.max_optimal_hops, 2)

    def test_unknown_and_unreachable_endpoints(self):
        unknown = validate_candidate_path(
            self.graph, 0, 99, [0, 99]
        )
        self.assertEqual(
            unknown.reasons, ("unknown_endpoint",)
        )

        graph = self.graph.copy()
        graph.add_node(99)

        unreachable = validate_candidate_path(
            graph, 0, 99, [0, 99]
        )
        self.assertEqual(
            unreachable.reasons,
            ("unreachable_destination",),
        )

    def test_rejects_unsupported_graphs_and_weights(self):
        directed = nx.DiGraph()
        directed.add_edge(0, 1, weight=1)

        multigraph = nx.MultiGraph()
        multigraph.add_edge(0, 1, weight=1)

        for graph in (directed, multigraph):
            with self.subTest(graph_type=type(graph).__name__):
                with self.assertRaises(ValueError):
                    validate_candidate_path(graph, 0, 1, [0, 1])

        for weight in (
            0, -1, float("nan"), float("inf"), True
        ):
            with self.subTest(weight=weight):
                graph = self.graph.copy()
                graph[0][1]["weight"] = weight

                with self.assertRaises(ValueError):
                    validate_candidate_path(
                        graph, 0, 2, [0, 1, 2]
                    )

    def test_all_frozen_g00_gold_paths(self):
        reference = os.environ.get("PATHFACT_G00_REFERENCE")

        if not reference:
            self.skipTest(
                "PATHFACT_G00_REFERENCE not configured"
            )

        count = 0

        with Path(reference).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                record = json.loads(line)
                count += 1

                with self.subTest(
                    case_id=record["production_case_id"]
                ):
                    graph = graph_from_topology_dict(
                        record["topology"]
                    )
                    src, dst = record["node_pair"]

                    result = validate_candidate_path(
                        graph,
                        src,
                        dst,
                        record["ground_truth_shortest_path"],
                    )

                    self.assertTrue(result.feasible)
                    self.assertTrue(result.optimal)
                    self.assertEqual(result.reasons, ())
                    self.assertEqual(
                        result.path_cost, record["gold_cost"]
                    )
                    self.assertEqual(
                        result.optimal_cost,
                        record["min_total_cost"],
                    )
                    self.assertEqual(
                        result.path_hops,
                        record["gold_path_hops"],
                    )
                    self.assertEqual(
                        result.min_optimal_hops,
                        record["min_optimal_hops"],
                    )
                    self.assertEqual(
                        result.max_optimal_hops,
                        record["max_optimal_hops"],
                    )

        self.assertEqual(count, 2200)


if __name__ == "__main__":
    unittest.main()
