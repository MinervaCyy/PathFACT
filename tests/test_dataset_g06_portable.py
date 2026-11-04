"""Portable tests for configurable paired SFT construction."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pathfact.datasets import iter_jsonl
from pathfact.datasets.g06 import (
    materialize_g06,
    materialize_paired_sft,
)


NAMESPACE = "pathfact-portable-test-v1"

TRAIN_QUOTAS = {
    "Easy": 1,
    "Medium": 1,
    "Hard": 0,
}

LOSSVAL_QUOTAS = {
    "Easy": 1,
    "Medium": 0,
    "Hard": 1,
}


class TestPortableG06(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

        self.cases = [
            ("PR00001", 2),
            ("PR00002", 3),
            ("PR00003", 4),
            ("PR00004", 6),
        ]

        self.paths = {
            "g00": self.root / "g00.jsonl",
            "g05": self.root / "g05.jsonl",
            "g04b": self.root / "g04b.jsonl",
            "prompts": self.root / "prompts.jsonl",
        }

        g00 = []
        g05 = []
        g04b = []
        prompts = []

        for pid, hops in self.cases:
            task = f"Shortest-path task for {pid}"
            wrong = f"Incorrect Student reasoning {pid}"
            path = list(range(hops + 1))

            g00.append({
                "production_case_id": pid,
                "gold_path_hops": hops,
                "ground_truth_shortest_path": path,
            })

            g05.append({
                "production_case_id": pid,
                "joint_valid": True,
                "canonical_wrong_reasoning_trace": wrong,
                "gold_target": {
                    "teacher_reasoning_trace": f"Gold reasoning {pid}"
                },
                "contrastive_target": {
                    "teacher_reasoning_trace":
                        f"Corrective reasoning {pid}"
                },
            })

            g04b.append({
                "production_case_id": pid,
                "canonical_teacher_record": {
                    "messages": [{
                        "role": "user",
                        "content": task + "\n" + wrong,
                    }]
                },
            })

            prompts.append({
                "production_case_id": pid,
                "messages": [{
                    "role": "user",
                    "content": task,
                }],
            })

        for name, records in {
            "g00": g00,
            "g05": g05,
            "g04b": g04b,
            "prompts": prompts,
        }.items():
            self.write_jsonl(self.paths[name], records)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_jsonl(path, records):
        path.write_text(
            "".join(
                json.dumps(record) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def construct(self, **overrides):
        arguments = {
            "g00_path": self.paths["g00"],
            "g05_path": self.paths["g05"],
            "g04b_path": self.paths["g04b"],
            "task_prompt_paths": [self.paths["prompts"]],
            "output_dir": self.root / "generated",
            "namespace": NAMESPACE,
            "train_quotas": TRAIN_QUOTAS,
            "lossval_quotas": LOSSVAL_QUOTAS,
        }
        arguments.update(overrides)
        return materialize_paired_sft(**arguments)

    def test_custom_split_and_paired_outputs(self):
        outputs = self.construct()

        self.assertEqual(len(outputs), 7)

        self.assertEqual(
            {name: path.name for name, path in outputs.items()},
            {
                "gold_train": "gold_train.jsonl",
                "contrastive_train": "contrastive_train.jsonl",
                "gold_loss_val": "gold_loss_val.jsonl",
                "contrastive_loss_val": "contrastive_loss_val.jsonl",
                "ranking": "ranking.tsv",
                "train_ids": "train_ids.txt",
                "loss_val_ids": "loss_val_ids.txt",
            },
        )

        datasets = {
            name: list(iter_jsonl(outputs[name]))
            for name in (
                "gold_train",
                "contrastive_train",
                "gold_loss_val",
                "contrastive_loss_val",
            )
        }

        for split, expected_count in (
            ("train", 2),
            ("loss_val", 2),
        ):
            gold = datasets[f"gold_{split}"]
            contrastive = datasets[f"contrastive_{split}"]

            self.assertEqual(len(gold), expected_count)
            self.assertEqual(len(contrastive), expected_count)

            self.assertEqual(
                [r["production_case_id"] for r in gold],
                [r["production_case_id"] for r in contrastive],
            )

        all_gold = (
            datasets["gold_train"]
            + datasets["gold_loss_val"]
        )

        all_contrastive = (
            datasets["contrastive_train"]
            + datasets["contrastive_loss_val"]
        )

        self.assertEqual(
            {r["production_case_id"] for r in all_gold},
            {pid for pid, _ in self.cases},
        )

        for condition, records in (
            ("gold", all_gold),
            ("contrastive", all_contrastive),
        ):
            for record in records:
                pid = record["production_case_id"]

                with self.subTest(condition=condition, case_id=pid):
                    self.assertEqual(
                        record["condition"], condition
                    )

                    for field in ("prompt", "completion"):
                        actual = hashlib.sha256(
                            record[field].encode("utf-8")
                        ).hexdigest()

                        self.assertEqual(
                            actual,
                            record[f"{field}_sha256"],
                        )

                    self.assertTrue(
                        record["completion"].startswith("<think>\n")
                    )

                    hops = dict(self.cases)[pid]
                    expected_path = list(range(hops + 1))

                    self.assertTrue(
                        record["completion"].endswith(
                            f"Answer: {expected_path}"
                        )
                    )

                    if condition == "contrastive":
                        self.assertIn(
                            f"Incorrect Student reasoning {pid}",
                            record["prompt"],
                        )

        train_ids = {
            r["production_case_id"]
            for r in datasets["gold_train"]
        }

        easy_ids = ["PR00001", "PR00002"]

        expected_easy_train = min(
            easy_ids,
            key=lambda pid: hashlib.sha256(
                f"{NAMESPACE}|{pid}".encode("utf-8")
            ).hexdigest(),
        )

        self.assertEqual(
            train_ids,
            {expected_easy_train, "PR00003"},
        )

    def test_rejects_missing_student_trace(self):
        records = list(iter_jsonl(self.paths["g04b"]))

        records[0]["canonical_teacher_record"]["messages"][0][
            "content"
        ] = "Corrective prompt without the Student trace"

        self.write_jsonl(self.paths["g04b"], records)

        with self.assertRaises(ValueError):
            self.construct()

        self.assertFalse(
            (self.root / "generated").exists()
        )

    def test_does_not_overwrite_existing_output(self):
        output = self.root / "generated"
        output.mkdir()

        marker = output / "existing.txt"
        marker.write_text("preserve", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            self.construct()

        self.assertEqual(
            marker.read_text(encoding="utf-8"),
            "preserve",
        )


if __name__ == "__main__":
    unittest.main()
