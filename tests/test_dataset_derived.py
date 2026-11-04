"""Portable validation tests for derived dataset adapters."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pathfact.datasets.derived import (
    BOUNDARY_FIELDS,
    BOUNDARY_TRAIN_SCRIPT_SHA256,
    CF_FIELDS,
    adapt_boundary_local,
    adapt_counterfactual_triples,
    _eligibility_record,
    _graph,
    _path_cost,
    _sha,
    _shortest,
    _task_from_prompt,
    export_derived_candidate,
)
from pathfact.datasets.io import DatasetFormatError


def prompt(graph, pair=(0, 2)):
    topology = {
        str(u): [[v, w] for v, w in sorted(adj.items())]
        for u, adj in graph.items()
    }
    return "Task\nTopology: " + repr(topology) + "\nNode pair: " + repr(list(pair))


class TestDerivedHelpers(unittest.TestCase):

    def setUp(self):
        self.original = {
            0: {1: 2, 2: 5},
            1: {0: 2, 2: 2},
            2: {0: 5, 1: 2},
        }
        self.switch = {
            0: {1: 2, 2: 3},
            1: {0: 2, 2: 2},
            2: {0: 3, 1: 2},
        }

    def test_prompt_graph_and_costs(self):
        graph, pair = _task_from_prompt(prompt(self.original))
        self.assertEqual(graph, self.original)
        self.assertEqual(pair, [0, 2])
        self.assertEqual(_path_cost(graph, [0, 1, 2], pair), 4)
        self.assertEqual(_shortest(graph, pair), 4)
        self.assertEqual(_shortest(self.switch, pair), 3)

    def test_rejects_invalid_graph_and_path(self):
        with self.assertRaises(DatasetFormatError):
            _graph({"0": [[1, 1]], "1": [[0, 2]]})
        with self.assertRaises(DatasetFormatError):
            _path_cost(self.original, [0, 2, 0], [0, 2])
        with self.assertRaises(DatasetFormatError):
            _path_cost(self.original, [0, 1], [0, 2])

    def test_ranking_semantics_use_same_graph_candidate_costs(self):
        source = {
            "production_case_id": "fixture-1",
            "cf_available": True,
            "prompt": prompt(self.original),
            "cf_prompt": prompt(self.switch),
            "old_path": [0, 1, 2],
            "cf_optimal_path": [0, 2],
            "old_optimal_cost": 4,
            "old_path_cost_on_cf": 4,
            "cf_optimal_cost": 3,
            "edited_edge": [0, 2],
            "old_edge_weight": 5,
            "new_edge_weight": 3,
        }
        eligibility = {
            "mode": "BIDIRECTIONAL",
            "original_old_cost": 4,
            "original_new_cost": 5,
            "switch_old_cost": 4,
            "switch_new_cost": 3,
        }
        output = _eligibility_record(source, eligibility)
        self.assertEqual(output["eligibility_category"], "strict_bidirectional")
        self.assertEqual(output["path_costs"], {
            "original_old": 4, "original_new": 5,
            "switch_old": 4, "switch_new": 3,
        })

        bad = dict(source, old_path_cost_on_cf=99)
        with self.assertRaises(DatasetFormatError):
            _eligibility_record(bad, eligibility)

        bad = dict(source, new_edge_weight=4)
        with self.assertRaises(DatasetFormatError):
            _eligibility_record(bad, eligibility)

        bad_eligibility = dict(eligibility, mode="SWITCH_ONLY_ORIGINAL_TIE")
        with self.assertRaises(DatasetFormatError):
            _eligibility_record(source, bad_eligibility)

    def test_duplicate_ids_rejected_by_jsonl_reader(self):
        from pathfact.datasets.derived import _jsonl

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.jsonl"
            row = {"production_case_id": "dup"}
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DatasetFormatError):
                _jsonl(path)

    def test_missing_id_rejected(self):
        from pathfact.datasets.derived import _jsonl
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-id.jsonl"
            path.write_text('{"other":"value"}\n', encoding="utf-8")
            with self.assertRaises(DatasetFormatError):
                _jsonl(path)

    def test_counterfactual_triple_adapter_allowlists_and_rejects_duplicate_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topology = {"0": [[1, 2], [2, 5]], "1": [[0, 2], [2, 2]], "2": [[0, 5], [1, 2]]}
            stay = {"0": [[1, 2], [2, 4]], "1": [[0, 2], [2, 2]], "2": [[0, 4], [1, 2]]}
            switch = {"0": [[1, 2], [2, 3]], "1": [[0, 2], [2, 2]], "2": [[0, 3], [1, 2]]}
            pair = [0, 2]
            case_hash = "a" * 64

            def record(variant, graph, path, cost):
                task = "Task\nTopology: " + repr(graph) + "\nNode pair: " + repr(pair)
                return {
                    "production_case_id": "case-1", "case_id": "case-1",
                    "counterfactual_parent_case_id": "case-1", "condition": "task_only",
                    "counterfactual_variant": variant, "source_views": ["G09_Balanced"],
                    "source_case_instance_sha256": case_hash,
                    "node_pair": pair, "topology": graph,
                    "messages": [{"role": "user", "content": task}],
                    "edited_edge": [0, 2], "edit_direction": "decrease",
                    "gold_path": path, "ground_truth_shortest_path": path,
                    "gold_cost": cost, "request_id": "excluded-internal-field",
                }

            rows = [
                record("original", topology, [0, 1, 2], 4),
                record("stay", stay, [0, 1, 2], 4),
                record("switch", switch, [0, 2], 3),
            ]
            source = root / "cf.jsonl"
            parent = root / "g09.jsonl"
            parent.write_text(json.dumps({
                "production_case_id": "case-1",
                "model_input": {"topology": topology, "node_pair": pair},
                "reference": {"ground_truth_shortest_path": [0, 1, 2], "gold_cost": 4},
                "provenance": {"case_instance_sha256": case_hash},
            }) + "\n", encoding="utf-8")
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            adapted = adapt_counterfactual_triples(
                input_path=source, g09_balanced_path=parent, g09_train_dist_path=parent,
                expected_records=3, expected_cases=1, expected_sha256=_sha(source),
                g09_balanced_sha256=_sha(parent), g09_train_dist_sha256=_sha(parent),
            )
            self.assertEqual(len(adapted), 1)
            self.assertEqual(set(adapted[0]), set(CF_FIELDS))
            self.assertNotIn("request_id", adapted[0])
            self.assertEqual(adapted[0]["source_views"], ["G09_Balanced"])

            source.write_text("".join(json.dumps(row) + "\n" for row in rows + [rows[0]]), encoding="utf-8")
            with self.assertRaises(DatasetFormatError):
                adapt_counterfactual_triples(
                    input_path=source, g09_balanced_path=parent, g09_train_dist_path=parent,
                    expected_records=4, expected_cases=1, expected_sha256=_sha(source),
                    g09_balanced_sha256=_sha(parent), g09_train_dist_sha256=_sha(parent),
                )

    def test_boundary_local_adapter_checks_membership_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = {"0": [[1, 2], [2, 6]], "1": [[0, 2], [2, 2]], "2": [[0, 6], [1, 2]]}
            stay = {"0": [[1, 2], [2, 5]], "1": [[0, 2], [2, 2]], "2": [[0, 5], [1, 2]]}
            switch = {"0": [[1, 2], [2, 3]], "1": [[0, 2], [2, 2]], "2": [[0, 3], [1, 2]]}

            def prompt_for(graph):
                return "Task\nTopology: " + repr(graph) + "\nNode pair: [0, 2]"

            row = {
                "case_id": "case-1", "original_prompt": prompt_for(original),
                "stay_prompt": prompt_for(stay), "switch_prompt": prompt_for(switch),
                "edited_edge": [0, 2], "old_path": [0, 1, 2], "new_path": [0, 2],
                "original_edge_weight": 6, "stay_edge_weight": 5, "switch_edge_weight": 3,
                "boundary_tie_weight": 4, "original_weight_vocabulary": [2, 6],
                "original_old_cost": 4, "original_new_cost": 6,
                "original_pair_gap_new_minus_old": 2,
                "stay_old_cost": 4, "stay_new_cost": 5,
                "switch_old_cost": 4, "switch_new_cost": 3,
                "stay_solver_optimal_cost": 4, "switch_solver_optimal_cost": 3,
                "stay_solver_path": [0, 1, 2], "switch_solver_path": [0, 2],
                "stay_designated_old_is_global_optimum": True,
                "switch_designated_new_is_global_optimum": True,
                "both_local_weights_in_original_vocabulary": False,
                "stay_weight_in_original_vocabulary": False,
                "switch_weight_in_original_vocabulary": False,
                "old_uses_edited_edge": False, "new_uses_edited_edge": True,
            }
            boundary = root / "boundary.jsonl"
            protocol = root / "protocol.json"
            boundary.write_text(json.dumps(row) + "\n", encoding="utf-8")
            protocol.write_text(json.dumps({"coverage": {"strict_boundary_local": 1}, "objective": {"margin": 1.0}, "canonical_source_script_sha256": BOUNDARY_TRAIN_SCRIPT_SHA256}), encoding="utf-8")
            eligible = [{
                "production_case_id": "case-1", "eligibility_category": "strict_bidirectional",
                "original_graph": {"topology": original, "node_pair": [0, 2]},
                "old_path": [0, 1, 2], "new_path": [0, 2],
                "edited_edge": [0, 2], "old_edge_weight": 6,
            }]
            output = adapt_boundary_local(boundary_path=boundary, eligibility_records=eligible,
                                          protocol_path=protocol, expected_records=1,
                                          boundary_sha256=_sha(boundary), protocol_sha256=_sha(protocol))
            self.assertEqual(len(output), 1)
            self.assertEqual(set(output[0]), set(BOUNDARY_FIELDS))
            self.assertTrue(output[0]["boundary_constraints"]["stay_switch_are_opposite_sides_of_tie"])

            with self.assertRaises(DatasetFormatError):
                adapt_boundary_local(boundary_path=boundary, eligibility_records=[],
                                     protocol_path=protocol, expected_records=1,
                                     boundary_sha256=_sha(boundary), protocol_sha256=_sha(protocol))

            row["switch_edge_weight"] = 4
            boundary.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(DatasetFormatError):
                adapt_boundary_local(boundary_path=boundary, eligibility_records=eligible,
                                     protocol_path=protocol, expected_records=1,
                                     boundary_sha256=_sha(boundary), protocol_sha256=_sha(protocol))

    def test_synthetic_candidate_is_deterministic_and_output_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_paths = [root / f"source-{i}.json" for i in range(12)]
            for i, path in enumerate(source_paths):
                path.write_text(f"source-{i}\n", encoding="utf-8")
            scoring = source_paths[11]
            scoring.write_text(json.dumps({
                "input_sha256": "f5879593a53f38943e8b3b991e1ee704c51c58ddd2a3379d524996d62053fb13",
                "model_summaries": {f"condition-{i}": {"n": 1062} for i in range(8)},
            }), encoding="utf-8")
            cf_row = {"production_case_id": "cf-1", "source_views": ["G09_Balanced"]}
            rank_row = {"production_case_id": "rank-1", "eligibility_category": "strict_bidirectional"}
            boundary_row = {"production_case_id": "rank-1", "eligibility_category": "strict_bidirectional"}

            def build(target):
                with (
                    patch("pathfact.datasets.derived.adapt_counterfactual_triples", return_value=[cf_row]),
                    patch("pathfact.datasets.derived.adapt_ranking_eligibility", return_value=([rank_row], {"strict_bidirectional": 1, "original_tie_switch_strict": 0, "no_strict_switch": 0})),
                    patch("pathfact.datasets.derived.adapt_boundary_local", return_value=[boundary_row]),
                    patch("pathfact.datasets.derived.CF_SCRIPT_SHA256", _sha(source_paths[8])),
                    patch("pathfact.datasets.derived.RANK_SCRIPT_SHA256", _sha(source_paths[9])),
                    patch("pathfact.datasets.derived.BOUNDARY_BUILDER_SHA256", _sha(source_paths[10])),
                ):
                    return export_derived_candidate(
                        counterfactual_path=source_paths[0], g09_balanced_path=source_paths[1],
                        g09_train_dist_path=source_paths[2], ranking_path=source_paths[3],
                        eligibility_path=source_paths[4], core_g06_train_path=source_paths[5],
                        boundary_path=source_paths[6], boundary_protocol_path=source_paths[7],
                        counterfactual_script_path=source_paths[8], ranking_script_path=source_paths[9],
                        boundary_script_path=source_paths[10],
                        scoring_report_path=scoring, output_dir=target, dataset_version="synthetic-v0.1",
                    )

            first, second = build(root / "candidate-a"), build(root / "candidate-b")
            first_files = [first["manifest"], first["counterfactual"], first["ranking"], first["boundary"], first["summary"]]
            second_files = [second["manifest"], second["counterfactual"], second["ranking"], second["boundary"], second["summary"]]
            self.assertEqual([_sha(p) for p in first_files], [_sha(p) for p in second_files])
            manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["release_status"], "draft_not_approved")
            self.assertIsNone(manifest["license"])

            occupied = root / "occupied"
            occupied.mkdir()
            with self.assertRaises(FileExistsError):
                export_derived_candidate(
                    counterfactual_path=source_paths[0], g09_balanced_path=source_paths[1],
                    g09_train_dist_path=source_paths[2], ranking_path=source_paths[3],
                    eligibility_path=source_paths[4], core_g06_train_path=source_paths[5],
                    boundary_path=source_paths[6], boundary_protocol_path=source_paths[7],
                    counterfactual_script_path=source_paths[8], ranking_script_path=source_paths[9],
                    boundary_script_path=source_paths[10],
                    scoring_report_path=scoring, output_dir=occupied, dataset_version="synthetic-v0.1",
                )


if __name__ == "__main__":
    unittest.main()
