"""Portable CLI regression tests for Vanilla-SFT materialisation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pathfact.datasets.g06 import (
    sft_record,
    sha256_text,
    write_jsonl,
)


class TestVanillaCLI(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "vanilla"

        train = []
        loss_val = []

        for i in range(244):
            pid = f"PR{i:05d}"
            prompt = (
                f"<｜User｜>Shortest-path task {i}"
                f"<｜Assistant｜>"
            )
            completion = (
                f"<think>\nReasoning {i}\n</think>\n"
                f"Answer: [0, {i + 1}]"
            )

            record = sft_record(
                pid, "gold", prompt, completion
            )

            if i < 200:
                train.append(record)
            else:
                loss_val.append(record)

        self.gold_train = self.root / "gold_train.jsonl"
        self.gold_loss_val = self.root / "gold_loss_val.jsonl"

        write_jsonl(self.gold_train, train)
        write_jsonl(self.gold_loss_val, loss_val)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, arguments):
        return subprocess.run(
            [sys.executable, "-m", "pathfact.cli", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )

    def cli_arguments(self):
        return [
            "dataset",
            "materialize-vanilla",
            "--gold-train", str(self.gold_train),
            "--gold-loss-val", str(self.gold_loss_val),
            "--output-dir", str(self.output),
        ]

    def test_materialize_vanilla_cli(self):
        result = self.run_cli(self.cli_arguments())

        self.assertEqual(
            result.returncode, 0, msg=result.stderr
        )
        self.assertIn(
            "Vanilla SFT materialisation completed",
            result.stdout,
        )

        for filename, count in (
            ("answer_only_train_200_prompt_completion_v1.jsonl", 200),
            ("answer_only_loss_val_44_prompt_completion_v1.jsonl", 44),
        ):
            path = self.output / filename
            self.assertTrue(path.is_file())

            with path.open(encoding="utf-8") as f:
                records = [
                    json.loads(line)
                    for line in f if line.strip()
                ]

            self.assertEqual(len(records), count)

            for record in records:
                self.assertEqual(
                    record["condition"], "vanilla"
                )
                self.assertTrue(
                    record["completion"].startswith("Answer: [")
                )
                self.assertNotIn(
                    "<think>", record["completion"]
                )

                for field in ("prompt", "completion"):
                    self.assertEqual(
                        record[f"{field}_sha256"],
                        sha256_text(record[field]),
                    )

    def test_existing_output_is_rejected(self):
        first = self.run_cli(self.cli_arguments())
        self.assertEqual(
            first.returncode, 0, msg=first.stderr
        )

        second = self.run_cli(self.cli_arguments())

        self.assertNotEqual(second.returncode, 0)
        self.assertIn(
            "File exists",
            second.stderr,
        )


if __name__ == "__main__":
    unittest.main()
