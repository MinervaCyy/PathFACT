"""End-to-end CLI regression against frozen G08 Balanced-300.

Requires PATHFACT_PIPELINE_REFERENCE.
Skipped in the standalone public environment.
No inference is performed and historical files are not modified.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

from itertools import zip_longest
from pathlib import Path


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


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


class TestFrozenEvaluationCLI(unittest.TestCase):

    def test_g08_balanced_base_task_end_to_end(self):
        reference = os.environ.get(
            "PATHFACT_PIPELINE_REFERENCE"
        )

        if not reference:
            self.skipTest(
                "PATHFACT_PIPELINE_REFERENCE not configured"
            )

        source = (
            Path(reference)
            / "G08_final_test_sets"
            / "evaluation"
            / "analysis"
            / "results"
            / "balanced_base_task"
            / "results.jsonl"
        )

        self.assertTrue(source.is_file())

        original_hash = file_sha256(source)

        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "evaluation"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pathfact.cli",
                    "evaluate",
                    "saved",
                    "--input",
                    str(source),
                    "--output-dir",
                    str(output_dir),
                    "--expected-records",
                    "300",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stderr,
            )

            summary_path = output_dir / "summary.json"
            scored_path = output_dir / "per_case.jsonl"

            self.assertTrue(summary_path.is_file())
            self.assertTrue(scored_path.is_file())

            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )

            cli_report = json.loads(result.stdout)

            self.assertEqual(summary, cli_report)

            total = 0
            valid = 0

            for original, scored in zip_longest(
                read_jsonl(source),
                read_jsonl(scored_path),
            ):
                self.assertIsNotNone(original)
                self.assertIsNotNone(scored)

                total += 1

                self.assertEqual(
                    original["case_id"],
                    scored["case_id"],
                )

                self.assertEqual(
                    original["request_id"],
                    scored["request_id"],
                )

                for field in SCORING_FIELDS:
                    self.assertIn(field, scored)

                    self.assertEqual(
                        original[field],
                        scored[field],
                        msg=(
                            f"{original['case_id']}: "
                            f"mismatch in {field}"
                        ),
                    )

                if scored["vsp_valid"] is True:
                    valid += 1

            self.assertEqual(total, 300)
            self.assertEqual(valid, 42)

            self.assertEqual(
                summary["records_present"], 300
            )
            self.assertEqual(
                summary["unique_case_ids"], 300
            )
            self.assertEqual(
                summary["unique_request_ids"], 300
            )
            self.assertEqual(summary["vsp_valid"], 42)
            self.assertAlmostEqual(
                summary["vsp_rate"], 0.14
            )
            self.assertEqual(
                summary["stopped_normally"], 227
            )
            self.assertEqual(
                summary["token_limit_truncation"], 73
            )
            self.assertEqual(
                summary["answer_extracted"], 225
            )
            self.assertEqual(
                summary["status_counts"], {"ok": 300}
            )
            self.assertEqual(
                summary["scoring_policy"],
                "historical_vsp",
            )
            self.assertEqual(
                summary["input_mode"],
                "saved_predicted_path",
            )

        self.assertEqual(
            file_sha256(source),
            original_hash,
            msg="Historical source file was modified",
        )


if __name__ == "__main__":
    unittest.main()
