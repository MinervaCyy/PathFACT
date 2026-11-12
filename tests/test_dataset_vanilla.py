"""Regression tests for Vanilla-SFT dataset materialisation."""

import json
import tempfile
import unittest
from pathlib import Path

from pathfact.datasets.g06 import (
    sft_record,
    sha256_text,
    write_jsonl,
)
from pathfact.datasets.vanilla import materialize_vanilla_sft


class TestVanillaSFT(unittest.TestCase):

    def make_inputs(self, root):
        train = []
        loss_val = []

        for i in range(244):
            pid = f"PR{i:05d}"
            prompt = f"<｜User｜>Test graph {i}<｜Assistant｜>"
            completion = (
                f"<think>\nExample reasoning {i}\n</think>\n"
                f"Answer: [0, {i + 1}]"
            )

            record = sft_record(
                pid, "gold", prompt, completion
            )

            if i < 200:
                train.append(record)
            else:
                loss_val.append(record)

        train_path = root / "gold_train.jsonl"
        val_path = root / "gold_loss_val.jsonl"

        write_jsonl(train_path, train)
        write_jsonl(val_path, loss_val)

        return train_path, val_path, train, loss_val

    def test_answer_only_materialisation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_path, val_path, train, val = (
                self.make_inputs(root)
            )

            outputs = materialize_vanilla_sft(
                gold_train_path=train_path,
                gold_loss_val_path=val_path,
                output_dir=root / "output",
            )

            for split, source in (
                ("train", train),
                ("loss_val", val),
            ):
                with outputs[split].open(
                    encoding="utf-8"
                ) as handle:
                    records = [
                        json.loads(line)
                        for line in handle
                    ]

                self.assertEqual(
                    len(records), len(source)
                )

                for original, generated in zip(
                    source, records
                ):
                    self.assertEqual(
                        original["production_case_id"],
                        generated["production_case_id"],
                    )

                    self.assertEqual(
                        original["prompt"],
                        generated["prompt"],
                    )

                    self.assertEqual(
                        generated["condition"], "vanilla"
                    )

                    self.assertTrue(
                        generated["completion"].startswith(
                            "Answer: ["
                        )
                    )

                    self.assertNotIn(
                        "<think>", generated["completion"]
                    )

                    self.assertEqual(
                        generated["completion_sha256"],
                        sha256_text(
                            generated["completion"]
                        ),
                    )

                    self.assertEqual(
                        generated["prompt_sha256"],
                        sha256_text(generated["prompt"]),
                    )

    def test_reject_corrupted_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            train_path, val_path, train, _ = (
                self.make_inputs(root)
            )

            train[0]["completion_sha256"] = "0" * 64
            write_jsonl(train_path, train)

            with self.assertRaisesRegex(
                ValueError, "checksum mismatch"
            ):
                materialize_vanilla_sft(
                    gold_train_path=train_path,
                    gold_loss_val_path=val_path,
                    output_dir=root / "output",
                )

            self.assertFalse(
                (root / "output").exists()
            )

    def test_reject_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_path, val_path, _, _ = (
                self.make_inputs(root)
            )

            output = root / "existing"
            output.mkdir()

            with self.assertRaises(FileExistsError):
                materialize_vanilla_sft(
                    gold_train_path=train_path,
                    gold_loss_val_path=val_path,
                    output_dir=output,
                )


if __name__ == "__main__":
    unittest.main()
