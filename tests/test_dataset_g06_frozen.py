"""Optional frozen G06 dataset integrity regression.

Set PATHFACT_G06_REFERENCE to the G06 dataset directory.
No historical research dataset is required for public CPU tests.
"""

import hashlib
import os
import unittest
from pathlib import Path

from pathfact.datasets import iter_jsonl


class TestFrozenG06(unittest.TestCase):

    def test_g06_integrity_and_pairing(self):
        reference = os.environ.get("PATHFACT_G06_REFERENCE")

        if not reference:
            self.skipTest(
                "Set PATHFACT_G06_REFERENCE to enable G06 regression"
            )

        directory = Path(reference)
        self.assertTrue(directory.is_dir())

        files = {
            "gold_train": (
                "gold_train_200_prompt_completion_v1.jsonl",
                "gold",
                200,
            ),
            "contrastive_train": (
                "contrastive_train_200_prompt_completion_v1.jsonl",
                "contrastive",
                200,
            ),
            "gold_val": (
                "gold_loss_val_44_prompt_completion_v1.jsonl",
                "gold",
                44,
            ),
            "contrastive_val": (
                "contrastive_loss_val_44_prompt_completion_v1.jsonl",
                "contrastive",
                44,
            ),
        }

        case_ids = {}

        for name, (filename, condition, expected_count) in files.items():
            path = directory / filename

            with self.subTest(dataset=name):
                self.assertTrue(path.is_file())

                records = list(iter_jsonl(
                    path,
                    required_fields=(
                        "condition",
                        "prompt",
                        "prompt_sha256",
                        "completion",
                        "completion_sha256",
                    ),
                    unique_key="production_case_id",
                ))

                self.assertEqual(len(records), expected_count)

                identifiers = set()

                for record in records:
                    case_id = record["production_case_id"]
                    identifiers.add(case_id)

                    with self.subTest(
                        dataset=name,
                        case_id=case_id,
                    ):
                        self.assertEqual(
                            record["condition"],
                            condition,
                        )

                        for field in ("prompt", "completion"):
                            value = record[field]
                            self.assertIsInstance(value, str)

                            actual_hash = hashlib.sha256(
                                value.encode("utf-8")
                            ).hexdigest()

                            self.assertEqual(
                                actual_hash,
                                record[f"{field}_sha256"],
                            )

                self.assertEqual(
                    len(identifiers),
                    expected_count,
                )

                case_ids[name] = identifiers

        # Gold and Contrastive use matching case identifiers.
        self.assertEqual(
            case_ids["gold_train"],
            case_ids["contrastive_train"],
        )
        self.assertEqual(
            case_ids["gold_val"],
            case_ids["contrastive_val"],
        )

        # Training and validation are case-disjoint.
        self.assertFalse(
            case_ids["gold_train"] & case_ids["gold_val"]
        )

        # 200 training cases + 44 validation cases.
        self.assertEqual(
            len(case_ids["gold_train"] | case_ids["gold_val"]),
            244,
        )


if __name__ == "__main__":
    unittest.main()
