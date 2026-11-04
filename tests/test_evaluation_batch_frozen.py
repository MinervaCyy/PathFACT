"""Frozen G08/G09 historical batch-scoring regression.

Requires PATHFACT_PIPELINE_REFERENCE pointing to the canonical
pipeline directory. Skipped in the standalone public environment.

This test performs no inference and modifies no research records.
"""

import os
import unittest
from pathlib import Path

from pathfact.datasets import iter_jsonl
from pathfact.evaluation.batch import (
    score_saved_record,
    summarize_scored_records,
)


EXPECTED = {
    "G08_final_test_sets/evaluation": {
        "balanced_base_contrastive": (300, 5),
        "balanced_base_task": (300, 42),
        "balanced_contrastive_sft": (300, 16),
        "balanced_contrastive_sft_taskonly": (300, 57),
        "balanced_gold_task": (300, 40),
        "traindist_base_contrastive": (300, 9),
        "traindist_base_task": (300, 48),
        "traindist_contrastive_sft": (300, 19),
        "traindist_contrastive_sft_taskonly": (300, 70),
        "traindist_gold_task": (300, 64),
    },
    "G09_large_training_disjoint_eval/evaluation": {
        "g09_bal_base_taskonly": (846, 174),
        "g09_bal_conSFT_taskonly": (846, 206),
        "g09_bal_goldSFT_taskonly": (846, 176),
        "g09_td_base_taskonly": (1600, 443),
        "g09_td_conSFT_taskonly": (1600, 501),
        "g09_td_goldSFT_taskonly": (1600, 462),
    },
}

FIELDS = (
    "syntactic_valid",
    "source_target_consistent",
    "no_repeated_nodes",
    "path_continuity",
    "optimal_cost",
    "vsp_valid",
    "failure_category",
    "predicted_cost",
    "gold_cost",
    "missing_edges",
    "edge_checks",
)


class TestFrozenG08G09Batch(unittest.TestCase):

    def test_all_frozen_evaluation_scores(self):
        reference = os.environ.get(
            "PATHFACT_PIPELINE_REFERENCE"
        )

        if not reference:
            self.skipTest(
                "PATHFACT_PIPELINE_REFERENCE not configured"
            )

        base = Path(reference)

        total_records = 0
        total_valid = 0
        total_runs = 0

        for relative_dir, expected_runs in EXPECTED.items():
            directory = base / relative_dir
            paths = sorted(directory.rglob("results.jsonl"))

            observed_names = [
                path.parent.name for path in paths
            ]

            self.assertEqual(
                len(observed_names),
                len(set(observed_names)),
                msg=f"Duplicate run directory names in {relative_dir}",
            )

            self.assertEqual(
                set(observed_names),
                set(expected_runs),
                msg=f"Unexpected evaluation runs in {relative_dir}",
            )

            for path in paths:
                run_name = path.parent.name
                expected_count, expected_valid = (
                    expected_runs[run_name]
                )

                with self.subTest(
                    stage=relative_dir,
                    run=run_name,
                ):

                    def audited_records():
                        for original in iter_jsonl(path):
                            recomputed = score_saved_record(
                                original
                            )

                            case_id = recomputed["case_id"]

                            for field in FIELDS:
                                self.assertIn(
                                    field,
                                    original,
                                    msg=(
                                        f"{run_name}/{case_id}: "
                                        f"missing {field}"
                                    ),
                                )

                                self.assertEqual(
                                    original[field],
                                    recomputed[field],
                                    msg=(
                                        f"{run_name}/{case_id}: "
                                        f"mismatch in {field}"
                                    ),
                                )

                            yield recomputed

                    summary = summarize_scored_records(
                        audited_records(),
                        expected_records=expected_count,
                    )

                    self.assertEqual(
                        summary["vsp_valid"],
                        expected_valid,
                    )

                    self.assertEqual(
                        summary["unique_case_ids"],
                        expected_count,
                    )

                    self.assertEqual(
                        summary["unique_request_ids"],
                        expected_count,
                    )

                    total_records += summary["records_present"]
                    total_valid += summary["vsp_valid"]
                    total_runs += 1

        self.assertEqual(total_runs, 16)
        self.assertEqual(total_records, 10338)
        self.assertEqual(total_valid, 2332)


if __name__ == "__main__":
    unittest.main()
