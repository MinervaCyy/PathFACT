"""Portable regression tests for historical VSP scoring."""

import unittest

from pathfact.evaluation import (
    extract_path,
    normalise_node_pair,
    normalise_topology,
    split_reasoning_and_content,
    verify_path,
)
from pathfact.graphs import graph_from_topology_dict
from pathfact.validation import validate_candidate_path


class TestHistoricalVSP(unittest.TestCase):

    def setUp(self):
        self.record = {
            "topology": {
                "0": [[1, 2], [2, 10]],
                "1": [[0, 2], [2, 3]],
                "2": [[0, 10], [1, 3]],
            },
            "node_pair": [0, 2],
            "gold_cost": 5,
        }

    def test_answer_marker_precedence(self):
        cases = (
            (
                "Intermediate [0, 2]. Answer: [0, 1, 2]",
                [0, 1, 2],
            ),
            (
                "Answer: [0, 2] Answer: [0, 1, 2]",
                [0, 1, 2],
            ),
            (
                "ANSWER: [-1, 0, 2]",
                [-1, 0, 2],
            ),
            (
                "Answer: [0, 2] Other [0, 1, 2]",
                [0, 2],
            ),
        )

        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(extract_path(text), expected)

    def test_bracket_fallback_and_malformed(self):
        cases = (
            ("First [0, 2] then [0, 1, 2]", [0, 1, 2]),
            ("Answer: [0 1 2]", []),
            ("No path provided", []),
            ("", []),
        )

        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(extract_path(text), expected)

    def test_reasoning_split_fallback(self):
        reasoning, final = split_reasoning_and_content({
            "content": (
                "<think>\nSteps\n</think>\n"
                "Answer: [0, 1, 2]"
            )
        })

        self.assertEqual(reasoning, "Steps")
        self.assertEqual(final, "Answer: [0, 1, 2]")

        reasoning, final = split_reasoning_and_content({
            "reasoning_content": "Explicit reasoning",
            "content": "Answer: [0, 1, 2]",
        })

        self.assertEqual(reasoning, "Explicit reasoning")
        self.assertEqual(final, "Answer: [0, 1, 2]")

    def test_historical_normalisation(self):
        topology = normalise_topology(
            "{0: [(1, 2)], 1: [(0, 2)]}"
        )

        self.assertEqual(
            topology,
            {0: [(1, 2.0)], 1: [(0, 2.0)]},
        )

        self.assertEqual(
            normalise_node_pair("(0, 2)"),
            [0, 2],
        )

    def test_vsp_categories_and_priority(self):
        cases = (
            ([0, 1, 2], True, "valid"),
            (
                [0, 2],
                False,
                "continuous_but_suboptimal",
            ),
            (
                [0, 9, 2],
                False,
                "nonexistent_edge",
            ),
            (
                [0, 1, 0, 1, 2],
                False,
                "repeated_nodes",
            ),
            (
                [1, 0, 2],
                False,
                "wrong_source_or_destination",
            ),
            (
                [],
                False,
                "no_path_extracted",
            ),
        )

        for candidate, expected_valid, category in cases:
            with self.subTest(candidate=candidate):
                result = verify_path(self.record, candidate)

                self.assertIs(
                    result["vsp_valid"],
                    expected_valid,
                )
                self.assertEqual(
                    result["failure_category"],
                    category,
                )

        missing = verify_path(
            self.record, [0, 9, 2]
        )

        self.assertEqual(
            missing["missing_edges"],
            [[0, 9], [9, 2]],
        )
        self.assertIsNone(missing["predicted_cost"])

        valid = verify_path(
            self.record, [0, 1, 2]
        )

        self.assertEqual(valid["predicted_cost"], 5.0)
        self.assertEqual(valid["gold_cost"], 5.0)

    def test_tolerance_and_reference_cost_policy(self):
        near = dict(self.record)
        near["gold_cost"] = 5 + 5e-10

        self.assertTrue(
            verify_path(near, [0, 1, 2])["vsp_valid"]
        )

        far = dict(self.record)
        far["gold_cost"] = 5 + 2e-9

        self.assertFalse(
            verify_path(far, [0, 1, 2])["vsp_valid"]
        )

        # Historical VSP trusts the supplied reference cost.
        # Strict validation recomputes the graph optimum.
        incorrect_reference = dict(self.record)
        incorrect_reference["gold_cost"] = 10

        historical = verify_path(
            incorrect_reference, [0, 2]
        )

        graph = graph_from_topology_dict(
            self.record["topology"]
        )

        strict = validate_candidate_path(
            graph, 0, 2, [0, 2]
        )

        self.assertTrue(historical["vsp_valid"])
        self.assertTrue(strict.feasible)
        self.assertFalse(strict.optimal)
        self.assertEqual(strict.optimal_cost, 5)


if __name__ == "__main__":
    unittest.main()
