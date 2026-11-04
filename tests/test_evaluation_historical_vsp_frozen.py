"""Optional regression against frozen Student and Teacher VSP scores.

Set PATHFACT_PIPELINE_REFERENCE to the canonical pipeline directory.
The public test suite runs without historical datasets.
"""

import os
import unittest
from pathlib import Path

from pathfact.datasets import iter_jsonl
from pathfact.evaluation import verify_path


SOURCES = {
    "G02": (
        "G02_tiera_wrong_rt_961/"
        "G02_CANONICAL_TIERA_CASES_961_V1.jsonl",
        "canonical_student_record",
        961,
    ),
    "G04a": (
        "G04_teacher_valid/gold/"
        "G04A_GOLD_VALID_CASES_559_V1.jsonl",
        "canonical_teacher_record",
        559,
    ),
    "G04b": (
        "G04_teacher_valid/contrastive/"
        "G04B_CONTRASTIVE_VALID_CASES_259_V1.jsonl",
        "canonical_teacher_record",
        259,
    ),
}

SCORING_FIELDS = (
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


class TestFrozenHistoricalVSP(unittest.TestCase):

    def test_all_historical_scores_match(self):
        reference = os.environ.get(
            "PATHFACT_PIPELINE_REFERENCE"
        )

        if not reference:
            self.skipTest(
                "PATHFACT_PIPELINE_REFERENCE not configured"
            )

        root = Path(reference)
        total = 0

        for stage, (filename, record_key, expected) in SOURCES.items():
            count = 0

            for row in iter_jsonl(
                root / filename,
                unique_key="production_case_id",
            ):
                count += 1
                case_id = row["production_case_id"]
                original = row[record_key]

                recomputed = verify_path(
                    original,
                    original["predicted_path"],
                )

                with self.subTest(
                    stage=stage,
                    case_id=case_id,
                ):
                    for field in SCORING_FIELDS:
                        self.assertIn(field, original)

                        self.assertEqual(
                            original[field],
                            recomputed[field],
                            msg=(
                                f"{stage}/{case_id}: "
                                f"scoring mismatch in {field}"
                            ),
                        )

            self.assertEqual(
                count,
                expected,
                msg=f"Unexpected {stage} record count",
            )

            total += count

        self.assertEqual(total, 1779)


if __name__ == "__main__":
    unittest.main()
