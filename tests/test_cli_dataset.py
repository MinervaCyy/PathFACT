"""Portable end-to-end tests for the PathFACT dataset CLI."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pathfact.datasets import iter_jsonl


class TestDatasetCLI(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "generated"

        cases = [
            ("PR00001", 2),
            ("PR00002", 3),
            ("PR00003", 4),
            ("PR00004", 6),
        ]

        datasets = {
            "g00": [],
            "g05": [],
            "g04b": [],
            "prompts": [],
        }

        for pid, hops in cases:
            task = f"Shortest-path task {pid}"
            wrong = f"Frozen incorrect reasoning {pid}"

            datasets["g00"].append({
                "production_case_id": pid,
                "gold_path_hops": hops,
                "ground_truth_shortest_path": list(range(hops + 1)),
            })

            datasets["g05"].append({
                "production_case_id": pid,
                "joint_valid": True,
                "canonical_wrong_reasoning_trace": wrong,
                "gold_target": {
                    "teacher_reasoning_trace": f"Gold reasoning {pid}",
                },
                "contrastive_target": {
                    "teacher_reasoning_trace": f"Correction {pid}",
                },
            })

            datasets["g04b"].append({
                "production_case_id": pid,
                "canonical_teacher_record": {
                    "messages": [{
                        "role": "user",
                        "content": task + "\n" + wrong,
                    }],
                },
            })

            datasets["prompts"].append({
                "production_case_id": pid,
                "messages": [{
                    "role": "user",
                    "content": task,
                }],
            })

        for name, records in datasets.items():
            (self.root / f"{name}.jsonl").write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, arguments):
        return subprocess.run(
            [sys.executable, "-m", "pathfact.cli", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )

    def build_arguments(self):
        return [
            "dataset", "build-paired-sft",
            "--g00", str(self.root / "g00.jsonl"),
            "--g05", str(self.root / "g05.jsonl"),
            "--g04b", str(self.root / "g04b.jsonl"),
            "--task-prompt", str(self.root / "prompts.jsonl"),
            "--output-dir", str(self.output),
            "--namespace", "pathfact-cli-test",
            "--train-quotas", "1", "1", "0",
            "--lossval-quotas", "1", "0", "1",
        ]

    def test_version_and_help(self):
        version = self.run_cli(["--version"])

        self.assertEqual(version.returncode, 0)
        self.assertIn("PathFACT 0.1.0.dev0", version.stdout)

        help_result = self.run_cli([
            "dataset", "build-paired-sft", "--help",
        ])

        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--train-quotas", help_result.stdout)
        self.assertIn("--lossval-quotas", help_result.stdout)

    def test_build_paired_sft(self):
        result = self.run_cli(self.build_arguments())

        self.assertEqual(
            result.returncode, 0,
            msg=result.stderr,
        )
        self.assertIn(
            "Paired SFT construction completed",
            result.stdout,
        )

        expected_files = (
            "gold_train.jsonl",
            "contrastive_train.jsonl",
            "gold_loss_val.jsonl",
            "contrastive_loss_val.jsonl",
            "ranking.tsv",
            "train_ids.txt",
            "loss_val_ids.txt",
        )

        for filename in expected_files:
            self.assertTrue(
                (self.output / filename).is_file(),
                msg=filename,
            )

        for split in ("train", "loss_val"):
            gold = list(iter_jsonl(
                self.output / f"gold_{split}.jsonl"
            ))
            contrastive = list(iter_jsonl(
                self.output / f"contrastive_{split}.jsonl"
            ))

            self.assertEqual(len(gold), 2)
            self.assertEqual(len(contrastive), 2)

            self.assertEqual(
                [r["production_case_id"] for r in gold],
                [r["production_case_id"] for r in contrastive],
            )

            for record in gold:
                self.assertEqual(record["condition"], "gold")
                self.assertIn("<think>\n", record["completion"])

            for record in contrastive:
                pid = record["production_case_id"]
                self.assertEqual(
                    record["condition"], "contrastive"
                )
                self.assertIn(
                    f"Frozen incorrect reasoning {pid}",
                    record["prompt"],
                )

    def test_rejects_invalid_arguments(self):
        original = self.build_arguments()

        for option, invalid_value in (
            ("--namespace", "   "),
            ("--train-quotas", "-1"),
        ):
            with self.subTest(option=option):
                arguments = original.copy()
                index = arguments.index(option)
                arguments[index + 1] = invalid_value

                result = self.run_cli(arguments)

                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.output.exists())

    def test_rejects_output_overwrite(self):
        arguments = self.build_arguments()

        first = self.run_cli(arguments)
        self.assertEqual(
            first.returncode, 0,
            msg=first.stderr,
        )

        marker = self.output / "preserve.txt"
        marker.write_text("unchanged", encoding="utf-8")

        second = self.run_cli(arguments)

        self.assertEqual(second.returncode, 2)
        self.assertEqual(
            marker.read_text(encoding="utf-8"),
            "unchanged",
        )


if __name__ == "__main__":
    unittest.main()
