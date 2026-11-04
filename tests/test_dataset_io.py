"""Portable tests for the PathFACT JSONL reader."""

import json
import tempfile
import unittest
from pathlib import Path

from pathfact.datasets import DatasetFormatError, iter_jsonl


class TestDatasetIO(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "records.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def write_records(self, records):
        self.path.write_text(
            "".join(json.dumps(r) + "\n" for r in records),
            encoding="utf-8",
        )

    def test_preserves_nested_records(self):
        original = {
            "production_case_id": "PR00001",
            "usable_attempts": [
                {"attempt_id": "A1", "record": {"path": [1, 2]}}
            ],
        }

        self.write_records([original])

        loaded = list(iter_jsonl(
            self.path,
            required_fields=("usable_attempts",),
            unique_key="production_case_id",
        ))

        self.assertEqual(loaded, [original])

    def test_rejects_duplicate_identifiers(self):
        self.write_records([
            {"production_case_id": "PR00001"},
            {"production_case_id": "PR00001"},
        ])

        with self.assertRaises(DatasetFormatError):
            list(iter_jsonl(
                self.path,
                unique_key="production_case_id",
            ))

    def test_rejects_missing_fields(self):
        self.write_records([{"production_case_id": "PR00001"}])

        with self.assertRaises(DatasetFormatError):
            list(iter_jsonl(
                self.path,
                required_fields=("prompt", "completion"),
            ))

    def test_rejects_invalid_json_and_nonobjects(self):
        for content in ('{"broken":', '[1, 2]\n'):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")

                with self.assertRaises(DatasetFormatError):
                    list(iter_jsonl(self.path))

    def test_rejects_empty_lines(self):
        self.path.write_text(
            '{"production_case_id": "PR00001"}\n\n',
            encoding="utf-8",
        )

        with self.assertRaises(DatasetFormatError):
            list(iter_jsonl(self.path))


if __name__ == "__main__":
    unittest.main()
