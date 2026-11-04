"""Tests for the preserved historical shortest-path validator."""

import unittest

import networkx as nx

from pathfact.validation import validate_case


class TestHistoricalValidator(unittest.TestCase):

    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from([
            (0, 1, 2),
            (1, 2, 3),
            (0, 2, 10),
        ])

        self.record = {
            "node_pair": [0, 2],
            "ground_truth_shortest_path": [0, 1, 2],
            "gold_cost": 5,
            "min_total_cost": 5,
            "min_optimal_hops": 2,
            "max_optimal_hops": 2,
            "gold_path_hops": 2,
        }

    def test_valid_record(self):
        validate_case(self.graph, self.record, 1, 3)

    def test_rejects_invalid_records(self):
        invalid_cases = {
            "wrong_start": {
                "ground_truth_shortest_path": [1, 2],
            },
            "repeated_nodes": {
                "ground_truth_shortest_path": [0, 1, 0, 1, 2],
            },
            "nonexistent_edge": {
                "ground_truth_shortest_path": [0, 8, 2],
            },
            "nonoptimal_path": {
                "ground_truth_shortest_path": [0, 2],
            },
            "incorrect_gold_cost": {
                "gold_cost": 6,
            },
            "inconsistent_cost_fields": {
                "min_total_cost": 6,
            },
            "invalid_hop_bound": {
                "max_optimal_hops": 4,
            },
            "incorrect_path_hops": {
                "gold_path_hops": 1,
            },
            "incorrect_minimum_hops": {
                "min_optimal_hops": 1,
            },
        }

        for name, changes in invalid_cases.items():
            with self.subTest(case=name):
                record = self.record.copy()
                record.update(changes)

                with self.assertRaises(AssertionError):
                    validate_case(self.graph, record, 1, 3)

    def test_rejects_disconnected_graph(self):
        graph = self.graph.copy()
        graph.add_node(99)

        with self.assertRaises(AssertionError):
            validate_case(graph, self.record, 1, 3)

    def test_historical_validator_scope(self):
        # Historical behaviour: max_optimal_hops is checked
        # against the configured bounds but not independently
        # recomputed from the graph.
        record = self.record.copy()
        record["max_optimal_hops"] = 3

        validate_case(self.graph, record, 1, 3)


if __name__ == "__main__":
    unittest.main()
