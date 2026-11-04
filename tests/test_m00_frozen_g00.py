"""Optional regression against the frozen G00 dataset.

Set PATHFACT_G00_REFERENCE to the reference JSONL path.
The public CPU test suite does not require historical datasets.
"""

import json
import os
import unittest
from pathlib import Path

from pathfact.graphs import (
    assign_weights,
    case_instance_sha256,
    generate_graph,
    graph_structure_sha256,
    topology_to_adjacency,
    weighted_graph_sha256,
)
from pathfact.validation import validate_case

from pathfact.solver import (
    eligible_pairs_by_exact_hop,
    reconstruct_path,
    select_pair_exact_hop_aware,
    shortest_path_hop_bounds_from_source,
)


class TestFrozenG00(unittest.TestCase):

    def test_all_2200_cases(self):
        reference = os.environ.get("PATHFACT_G00_REFERENCE")

        if not reference:
            self.skipTest(
                "Set PATHFACT_G00_REFERENCE to enable frozen regression"
            )

        path = Path(reference)
        self.assertTrue(path.is_file())

        count = 0
        case_ids = set()

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                record = json.loads(line)
                case_id = record["production_case_id"]

                with self.subTest(case_id=case_id):
                    self.assertNotIn(case_id, case_ids)
                    case_ids.add(case_id)
                    count += 1

                    graph = generate_graph(
                        record["topology_type"],
                        record["topology_dimension"],
                        record["graph_seed"],
                    )
                    graph = assign_weights(
                        graph,
                        record["weight_distribution"],
                        record["weight_seed"],
                    )

                    src, dst = record["node_pair"]

                    self.assertEqual(
                        graph.number_of_nodes(),
                        record["num_nodes"],
                    )
                    self.assertEqual(
                        graph.number_of_edges(),
                        record["num_edges"],
                    )
                    self.assertEqual(
                        graph_structure_sha256(graph),
                        record["graph_structure_sha256"],
                    )
                    self.assertEqual(
                        weighted_graph_sha256(graph),
                        record["weighted_graph_sha256"],
                    )
                    self.assertEqual(
                        case_instance_sha256(graph, src, dst),
                        record["case_instance_sha256"],
                    )

                    expected_adjacency = {
                        int(u): [
                            (int(v), int(w))
                            for v, w in edges
                        ]
                        for u, edges in record["topology"].items()
                    }
                    self.assertEqual(
                        topology_to_adjacency(graph),
                        expected_adjacency,
                    )

                    u, v = sorted((src, dst))
                    dist, min_hops, max_hops, pred = (
                        shortest_path_hop_bounds_from_source(graph, u)
                    )

                    path_uv = reconstruct_path(pred, u, v)
                    actual_path = (
                        path_uv if src == u
                        else list(reversed(path_uv))
                    )

                    self.assertEqual(dist[v], record["gold_cost"])
                    self.assertEqual(
                        dist[v], record["min_total_cost"]
                    )
                    self.assertEqual(
                        min_hops[v], record["min_optimal_hops"]
                    )
                    self.assertEqual(
                        max_hops[v], record["max_optimal_hops"]
                    )
                    self.assertEqual(
                        actual_path,
                        record["ground_truth_shortest_path"],
                    )
                    self.assertEqual(
                        len(actual_path) - 1,
                        record["gold_path_hops"],
                    )

                    hop_low, hop_high = map(
                        int, record["hop_bucket"].split("-")
                    )
                    validate_case(
                        graph, record, hop_low, hop_high
                    )

                    strata = eligible_pairs_by_exact_hop(
                        graph, hop_low, hop_high
                    )
                    self.assertTrue(strata)

                    selected = select_pair_exact_hop_aware(
                        strata, record["pair_seed"]
                    )

                    self.assertEqual(
                        [selected["src"], selected["dst"]],
                        record["node_pair"],
                    )
                    self.assertEqual(
                        selected["canonical_path"],
                        record["ground_truth_shortest_path"],
                    )

                    for field in (
                        "selected_exact_hop",
                        "pair_orientation_swapped",
                        "eligible_pair_count_total",
                        "eligible_pair_count_selected_exact_hop",
                        "feasible_exact_hops",
                    ):
                        self.assertEqual(
                            selected[field], record[field]
                        )

        self.assertEqual(count, 2200)


if __name__ == "__main__":
    unittest.main()
