"""Frozen regression for G08/G09 answer extraction.

Requires PATHFACT_PIPELINE_REFERENCE.
Skipped when the historical research data is unavailable.
"""

import os
import unittest
from pathlib import Path

from pathfact.datasets import iter_jsonl
from pathfact.evaluation.extraction import (
    extract_historical_message,
)


RESPONSE_FIELDS = (
    "reasoning_trace",
    "final_content",
    "reasoning_present",
    "final_content_present",
    "predicted_path",
    "answer_extracted",
)


class TestFrozenResponseExtraction(unittest.TestCase):

    def test_all_g08_g09_responses(self):
        reference = os.environ.get(
            "PATHFACT_PIPELINE_REFERENCE"
        )

        if not reference:
            self.skipTest(
                "PATHFACT_PIPELINE_REFERENCE not configured"
            )

        root = Path(reference)

        directories = {
            "G08": (
                root
                / "G08_final_test_sets"
                / "evaluation"
            ),
            "G09": (
                root
                / "G09_large_training_disjoint_eval"
                / "evaluation"
            ),
        }

        total = 0
        successful = 0
        failed = 0
        runs = 0

        for stage, directory in directories.items():
            paths = sorted(
                directory.rglob("results.jsonl")
            )

            self.assertEqual(
                len(paths),
                10 if stage == "G08" else 6,
            )

            for path in paths:
                runs += 1

                for record in iter_jsonl(path):
                    total += 1

                    case_id = record.get("case_id")
                    status = record.get("status")

                    context = (
                        f"{stage}/{path.parent.name}/"
                        f"{case_id}"
                    )

                    if status == "ok":
                        successful += 1

                        reconstructed = (
                            extract_historical_message(
                                record["raw_message"]
                            )
                        )

                        for field in RESPONSE_FIELDS:
                            self.assertIn(
                                field,
                                record,
                                msg=f"{context}: missing {field}",
                            )

                            self.assertEqual(
                                record[field],
                                reconstructed[field],
                                msg=(
                                    f"{context}: "
                                    f"mismatch in {field}"
                                ),
                            )

                    elif status == "error":
                        failed += 1

                        self.assertEqual(
                            record["predicted_path"],
                            [],
                            msg=context,
                        )

                        self.assertIs(
                            record["answer_extracted"],
                            False,
                            msg=context,
                        )

                    else:
                        self.fail(
                            f"{context}: unknown status {status!r}"
                        )

        self.assertEqual(runs, 16)
        self.assertEqual(total, 10338)
        self.assertEqual(successful, 10302)
        self.assertEqual(failed, 36)


if __name__ == "__main__":
    unittest.main()
