"""Portable tests for CPU-only batch VSP re-scoring."""

import unittest

from pathfact.evaluation.batch import (
    score_saved_record,
    summarize_scored_records,
)


class TestBatchEvaluation(unittest.TestCase):

    def setUp(self):
        self.base = {
            "case_id": "PR00001",
            "request_id": "REQ001",
            "topology": {
                "0": [[1, 2], [2, 10]],
                "1": [[0, 2], [2, 3]],
                "2": [[0, 10], [1, 3]],
            },
            "node_pair": [0, 2],
            "gold_cost": 5,
            "status": "ok",
            "finish_reason": "stop",
            "stopped_normally": True,
            "token_limit_truncation": False,
            "answer_extracted": True,
            "predicted_path": [0, 1, 2],
        }

    def test_recomputes_saved_path_score(self):
        result = score_saved_record(self.base)

        self.assertTrue(result["vsp_valid"])
        self.assertEqual(result["failure_category"], "valid")
        self.assertEqual(result["predicted_cost"], 5.0)

    def test_truncation_does_not_gate_scoring(self):
        record = dict(
            self.base,
            finish_reason="length",
            stopped_normally=False,
            token_limit_truncation=True,
        )

        result = score_saved_record(record)

        # This is historical scoring behaviour, not a claim
        # about any observed G08/G09 truncated response.
        self.assertTrue(result["vsp_valid"])

        summary = summarize_scored_records([result])

        self.assertEqual(summary["vsp_valid"], 1)
        self.assertEqual(
            summary["token_limit_truncation"], 1
        )

    def test_full_denominator_and_failure_categories(self):
        valid = score_saved_record(self.base)

        suboptimal = score_saved_record(dict(
            self.base,
            case_id="PR00002",
            request_id="REQ002",
            predicted_path=[0, 2],
        ))

        failed = score_saved_record(dict(
            self.base,
            case_id="PR00003",
            request_id="REQ003",
            status="error",
            finish_reason=None,
            stopped_normally=False,
            answer_extracted=False,
            predicted_path=[],
        ))

        summary = summarize_scored_records(
            [valid, suboptimal, failed],
            expected_records=3,
        )

        self.assertEqual(summary["records_present"], 3)
        self.assertEqual(summary["vsp_valid"], 1)
        self.assertAlmostEqual(summary["vsp_rate"], 1 / 3)
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
            summary["failure_categories"]["no_path_extracted"],
            1,
        )
        self.assertEqual(summary["status_counts"]["error"], 1)

    def test_rejects_missing_path_or_incomplete_dataset(self):
        record = dict(self.base)
        del record["predicted_path"]

        with self.assertRaises(ValueError):
            score_saved_record(record)

        scored = score_saved_record(self.base)

        with self.assertRaises(ValueError):
            summarize_scored_records(
                [scored],
                expected_records=2,
            )


if __name__ == "__main__":
    unittest.main()
