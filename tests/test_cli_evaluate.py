"""Portable CLI tests for saved-path offline evaluation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestEvaluateCLI(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)
        self.source = self.root / "responses.jsonl"
        self.output = self.root / "evaluation"

        base = {
            "topology": {
                "0": [[1, 2], [2, 10]],
                "1": [[0, 2], [2, 3]],
                "2": [[0, 10], [1, 3]],
            },
            "node_pair": [0, 2],
            "gold_cost": 5,
        }

        self.records = [
            {
                **base,
                "case_id": "PR00001",
                "request_id": "REQ001",
                "status": "ok",
                "finish_reason": "stop",
                "stopped_normally": True,
                "token_limit_truncation": False,
                "answer_extracted": True,
                "predicted_path": [0, 1, 2],
            },
            {
                **base,
                "case_id": "PR00002",
                "request_id": "REQ002",
                "status": "ok",
                "finish_reason": "length",
                "stopped_normally": False,
                "token_limit_truncation": True,
                "answer_extracted": True,
                "predicted_path": [0, 2],
            },
            {
                **base,
                "case_id": "PR00003",
                "request_id": "REQ003",
                "status": "error",
                "finish_reason": None,
                "stopped_normally": False,
                "token_limit_truncation": False,
                "answer_extracted": False,
                "predicted_path": [],
            },
        ]

        self.write_records(self.records)

    def write_records(self, records):
        with self.source.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record) + "\n")

    def run_cli(self, *options):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pathfact.cli",
                "evaluate",
                "saved",
                "--input",
                str(self.source),
                "--output-dir",
                str(self.output),
                *options,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_outputs_and_full_denominator(self):
        result = self.run_cli(
            "--expected-records", "3"
        )

        self.assertEqual(
            result.returncode, 0, msg=result.stderr
        )

        summary = json.loads(
            (self.output / "summary.json").read_text(
                encoding="utf-8"
            )
        )

        rows = [
            json.loads(line)
            for line in (
                self.output / "per_case.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]

        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["records_present"], 3)
        self.assertEqual(summary["vsp_valid"], 1)
        self.assertAlmostEqual(summary["vsp_rate"], 1 / 3)
        self.assertEqual(
            summary["token_limit_truncation"], 1
        )
        self.assertEqual(
            summary["status_counts"]["error"], 1
        )
        self.assertEqual(
            summary["failure_categories"]["valid"], 1
        )
        self.assertEqual(
            summary["failure_categories"][
                "continuous_but_suboptimal"
            ],
            1,
        )
        self.assertEqual(
            summary["failure_categories"][
                "no_path_extracted"
            ],
            1,
        )
        self.assertEqual(
            summary["scoring_policy"], "historical_vsp"
        )

        self.assertTrue(rows[0]["vsp_valid"])
        self.assertFalse(rows[1]["vsp_valid"])
        self.assertFalse(rows[2]["vsp_valid"])

    def test_rejects_existing_output(self):
        self.output.mkdir()
        sentinel = self.output / "existing.txt"
        sentinel.write_text("preserve", encoding="utf-8")

        result = self.run_cli()

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "preserve",
        )

    def test_incomplete_dataset_leaves_no_outputs(self):
        result = self.run_cli(
            "--expected-records", "4"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_missing_path_leaves_no_outputs(self):
        incomplete = dict(self.records[1])
        del incomplete["predicted_path"]

        self.write_records([
            self.records[0],
            incomplete,
        ])

        result = self.run_cli()

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_missing_input_rejected(self):
        self.source.unlink()

        result = self.run_cli()

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_evaluate_help(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pathfact.cli",
                "evaluate",
                "saved",
                "--help",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--expected-records", result.stdout)


if __name__ == "__main__":
    unittest.main()
