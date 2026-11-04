"""Portable tests for historical response extraction."""

import unittest

from pathfact.evaluation.extraction import (
    extract_historical_message,
)


class TestHistoricalExtraction(unittest.TestCase):

    def test_final_answer_takes_priority(self):
        message = {
            "reasoning_content": (
                "Candidate paths: [0, 2] and [0, 1, 2]"
            ),
            "content": "Answer: [0, 1, 2]",
        }

        result = extract_historical_message(message)

        self.assertEqual(
            result["predicted_path"],
            [0, 1, 2],
        )
        self.assertTrue(result["answer_extracted"])
        self.assertTrue(result["reasoning_present"])
        self.assertTrue(result["final_content_present"])

    def test_reasoning_only_is_not_an_answer(self):
        message = {
            "reasoning_content": "Answer: [0, 1, 2]",
            "content": "",
        }

        result = extract_historical_message(message)

        self.assertEqual(result["predicted_path"], [])
        self.assertFalse(result["answer_extracted"])
        self.assertTrue(result["reasoning_present"])
        self.assertFalse(result["final_content_present"])

    def test_inline_think_separation(self):
        message = {
            "content": (
                "<think>\n"
                "Consider [0, 2] first.\n"
                "</think>\n"
                "Answer: [0, 1, 2]"
            )
        }

        result = extract_historical_message(message)

        self.assertEqual(
            result["reasoning_trace"],
            "Consider [0, 2] first.",
        )
        self.assertEqual(
            result["final_content"],
            "Answer: [0, 1, 2]",
        )
        self.assertEqual(
            result["predicted_path"],
            [0, 1, 2],
        )

    def test_missing_answer_and_invalid_message(self):
        result = extract_historical_message({
            "content": "Unable to determine the route."
        })

        self.assertEqual(result["predicted_path"], [])
        self.assertFalse(result["answer_extracted"])
        self.assertFalse(result["reasoning_present"])
        self.assertTrue(result["final_content_present"])

        with self.assertRaises(TypeError):
            extract_historical_message(None)


if __name__ == "__main__":
    unittest.main()
