"""Optional frozen G00-G06 identity and Student-trace lineage tests.

Set PATHFACT_PIPELINE_REFERENCE to the canonical pipeline directory.
No historical dataset is required for ordinary public tests.
"""

import hashlib
import os
import unittest
from pathlib import Path

from pathfact.datasets import iter_jsonl


STAGES = {
    "G00": "G00_original_2200/G00_ORIGINAL_TASK_UNIVERSE_2200_V1.jsonl",
    "G01": "G01_usable_failure_1525/G01_USABLE_STUDENT_FAILURE_CASES_1525_V1.jsonl",
    "G02": "G02_tiera_wrong_rt_961/G02_CANONICAL_TIERA_CASES_961_V1.jsonl",
    "G03": "G03_teacher_entry_958/G03_TEACHER_ENTRY_CASES_958_V1.jsonl",
    "G04a": "G04_teacher_valid/gold/G04A_GOLD_VALID_CASES_559_V1.jsonl",
    "G04b": "G04_teacher_valid/contrastive/G04B_CONTRASTIVE_VALID_CASES_259_V1.jsonl",
    "G05": "G05_joint_valid_244/G05_JOINT_VALID_CASES_244_V1.jsonl",
}

TRACE_STAGES = ("G02", "G03", "G04b", "G05")

G06_FILES = (
    "gold_train_200_prompt_completion_v1.jsonl",
    "contrastive_train_200_prompt_completion_v1.jsonl",
    "gold_loss_val_44_prompt_completion_v1.jsonl",
    "contrastive_loss_val_44_prompt_completion_v1.jsonl",
)


class TestFrozenDatasetLineage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reference = os.environ.get("PATHFACT_PIPELINE_REFERENCE")

        if not reference:
            raise unittest.SkipTest(
                "PATHFACT_PIPELINE_REFERENCE not configured"
            )

        root = Path(reference)

        cls.ids = {}
        cls.traces = {}
        cls.hash_failures = {}

        for stage, relative in STAGES.items():
            identifiers = set()
            fingerprints = {}
            mismatches = []

            for record in iter_jsonl(
                root / relative,
                unique_key="production_case_id",
            ):
                pid = record["production_case_id"]
                identifiers.add(pid)

                if stage in TRACE_STAGES:
                    trace = record["canonical_wrong_reasoning_trace"]
                    recorded = record[
                        "canonical_wrong_reasoning_trace_sha256"
                    ]

                    actual = hashlib.sha256(
                        trace.encode("utf-8")
                    ).hexdigest()

                    fingerprints[pid] = recorded

                    if recorded != actual:
                        mismatches.append(pid)

            cls.ids[stage] = identifiers
            cls.traces[stage] = fingerprints
            cls.hash_failures[stage] = mismatches

        g06_ids = set()

        for filename in G06_FILES:
            path = root / "G06_train200_lossval44" / filename

            for record in iter_jsonl(
                path,
                unique_key="production_case_id",
            ):
                g06_ids.add(record["production_case_id"])

        cls.ids["G06"] = g06_ids

    def test_case_identity_lineage(self):
        expected = {
            "G00": 2200,
            "G01": 1525,
            "G02": 961,
            "G03": 958,
            "G04a": 559,
            "G04b": 259,
            "G05": 244,
            "G06": 244,
        }

        for stage, count in expected.items():
            with self.subTest(stage=stage):
                self.assertEqual(len(self.ids[stage]), count)

        relationships = (
            ("G01", "G00"),
            ("G02", "G01"),
            ("G03", "G02"),
            ("G04a", "G03"),
            ("G04b", "G03"),
            ("G05", "G04a"),
            ("G05", "G04b"),
            ("G06", "G05"),
        )

        for child, parent in relationships:
            with self.subTest(child=child, parent=parent):
                self.assertLessEqual(
                    self.ids[child],
                    self.ids[parent],
                )

        self.assertEqual(
            self.ids["G05"],
            self.ids["G04a"] & self.ids["G04b"],
        )
        self.assertEqual(self.ids["G06"], self.ids["G05"])

    def test_student_reasoning_trace_continuity(self):
        for stage in TRACE_STAGES:
            with self.subTest(stage=stage):
                self.assertEqual(
                    self.hash_failures[stage], []
                )

        relationships = (
            ("G02", "G03"),
            ("G03", "G04b"),
            ("G04b", "G05"),
        )

        for earlier, later in relationships:
            shared = (
                self.traces[earlier].keys()
                & self.traces[later].keys()
            )

            with self.subTest(earlier=earlier, later=later):
                self.assertEqual(
                    len(shared),
                    len(self.traces[later]),
                )

                for pid in shared:
                    self.assertEqual(
                        self.traces[earlier][pid],
                        self.traces[later][pid],
                        msg=f"Student trace mismatch: {pid}",
                    )


if __name__ == "__main__":
    unittest.main()
