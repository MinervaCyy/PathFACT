"""Optional regression for frozen G04-to-G05 Teacher provenance.

Set PATHFACT_PIPELINE_REFERENCE to the canonical pipeline directory.
Historical datasets are not required for public CPU unit tests.
"""

import hashlib
import os
import unittest
from collections import Counter
from pathlib import Path

from pathfact.datasets import iter_jsonl


FILES = {
    "gold": (
        "G04_teacher_valid/gold/"
        "G04A_GOLD_VALID_CASES_559_V1.jsonl"
    ),
    "contrastive": (
        "G04_teacher_valid/contrastive/"
        "G04B_CONTRASTIVE_VALID_CASES_259_V1.jsonl"
    ),
    "g05": (
        "G05_joint_valid_244/"
        "G05_JOINT_VALID_CASES_244_V1.jsonl"
    ),
}

COMMON_FIELDS = {
    "teacher_request_id",
    "teacher_reasoning_trace",
    "teacher_reasoning_trace_sha256",
    "teacher_final_content",
    "teacher_final_content_sha256",
    "teacher_vsp_valid",
    "validation_profile",
}

TARGET_FIELDS = {
    "gold": COMMON_FIELDS | {
        "canonical_teacher_protocol",
        "selection_reason",
    },
    "contrastive": COMMON_FIELDS | {
        "selected_rollout_family",
        "selection_basis",
        "wrong_rt_continuity_verified",
    },
}


def read_by_case(path):
    return {
        record["production_case_id"]: record
        for record in iter_jsonl(
            path,
            unique_key="production_case_id",
        )
    }


class TestFrozenTeacherProvenance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reference = os.environ.get(
            "PATHFACT_PIPELINE_REFERENCE"
        )

        if not reference:
            raise unittest.SkipTest(
                "PATHFACT_PIPELINE_REFERENCE not configured"
            )

        root = Path(reference)

        cls.records = {
            name: read_by_case(root / relative)
            for name, relative in FILES.items()
        }

    def test_g04_teacher_content_hashes(self):
        expected_counts = {
            "gold": 559,
            "contrastive": 259,
        }

        for name, expected_count in expected_counts.items():
            records = self.records[name]

            self.assertEqual(len(records), expected_count)

            for case_id, record in records.items():
                with self.subTest(
                    condition=name,
                    case_id=case_id,
                ):
                    for field in (
                        "teacher_reasoning_trace",
                        "teacher_final_content",
                    ):
                        content = record[field]
                        self.assertIsInstance(content, str)

                        actual = hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest()

                        self.assertEqual(
                            actual,
                            record[field + "_sha256"],
                        )

    def test_g05_exact_teacher_provenance(self):
        gold = self.records["gold"]
        contrastive = self.records["contrastive"]
        paired = self.records["g05"]

        intersection = set(gold) & set(contrastive)

        self.assertEqual(len(intersection), 244)
        self.assertEqual(set(paired), intersection)

        profiles = Counter()

        for case_id, record in paired.items():
            with self.subTest(case_id=case_id):
                self.assertIs(record["joint_valid"], True)

                for name, target_key in (
                    ("gold", "gold_target"),
                    ("contrastive", "contrastive_target"),
                ):
                    source = self.records[name][case_id]
                    target = record[target_key]

                    self.assertEqual(
                        set(target),
                        TARGET_FIELDS[name],
                    )

                    for field in TARGET_FIELDS[name]:
                        self.assertEqual(
                            target[field],
                            source[field],
                            msg=f"{case_id}: {name}: {field}",
                        )

                    profiles[
                        name,
                        target["validation_profile"],
                    ] += 1

        self.assertEqual(
            profiles,
            Counter({
                (
                    "gold",
                    "solver_verified_vsp",
                ): 103,
                (
                    "gold",
                    "vsp_plus_deterministic_rt_factuality_audit",
                ): 141,
                (
                    "contrastive",
                    "solver_verified_vsp",
                ): 103,
                (
                    "contrastive",
                    "vsp_plus_corrective_rt_factuality_validation",
                ): 141,
            }),
        )


if __name__ == "__main__":
    unittest.main()
