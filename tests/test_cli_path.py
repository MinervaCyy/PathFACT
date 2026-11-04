"""Portable end-to-end tests for shortest-path verification CLI."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestPathCLI(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.graph_file = self.root / "graph.json"

        self.graph_file.write_text(
            json.dumps({
                "0": [[1, 2], [2, 10]],
                "1": [[0, 2], [2, 3]],
                "2": [[0, 10], [1, 3]],
            }),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "pathfact.cli", *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def verify(self, candidate, source=0, destination=2):
        return self.run_cli(
            "path", "verify",
            "--graph-file", str(self.graph_file),
            "--source", str(source),
            "--destination", str(destination),
            "--candidate", json.dumps(candidate),
        )

    def test_help(self):
        result = self.run_cli("path", "verify", "--help")

        self.assertEqual(result.returncode, 0)
        self.assertIn("--graph-file", result.stdout)
        self.assertIn("--candidate", result.stdout)

    def test_optimal_and_nonoptimal_paths(self):
        cases = [
            ([0, 1, 2], True, 5, ()),
            ([0, 2], False, 10, ("non_optimal",)),
        ]

        for candidate, optimal, cost, reasons in cases:
            with self.subTest(candidate=candidate):
                result = self.verify(candidate)

                self.assertEqual(result.returncode, 0, result.stderr)

                data = json.loads(result.stdout)

                self.assertTrue(data["feasible"])
                self.assertEqual(data["optimal"], optimal)
                self.assertEqual(data["path_cost"], cost)
                self.assertEqual(data["optimal_cost"], 5)
                self.assertEqual(tuple(data["reasons"]), reasons)
                self.assertEqual(data["min_optimal_hops"], 2)
                self.assertEqual(data["max_optimal_hops"], 2)

    def test_invalid_and_missing_paths(self):
        cases = [
            ([0, 9, 2], "missing_edge"),
            ([0, 1, 0, 1, 2], "repeated_node"),
            (None, "missing_path"),
            ([], "empty_path"),
        ]

        for candidate, reason in cases:
            with self.subTest(candidate=candidate):
                result = self.verify(candidate)

                self.assertEqual(result.returncode, 0, result.stderr)

                data = json.loads(result.stdout)

                self.assertFalse(data["feasible"])
                self.assertFalse(data["optimal"])
                self.assertIn(reason, data["reasons"])

    def test_unknown_endpoint(self):
        result = self.verify([0, 99], destination=99)

        self.assertEqual(result.returncode, 0, result.stderr)

        data = json.loads(result.stdout)

        self.assertFalse(data["feasible"])
        self.assertFalse(data["optimal"])
        self.assertEqual(data["reasons"], ["unknown_endpoint"])

    def test_rejects_invalid_inputs(self):
        base = [
            "path", "verify",
            "--graph-file", str(self.graph_file),
            "--source", "0",
            "--destination", "2",
            "--candidate", "[0, 1, 2]",
        ]

        # Invalid candidate JSON.
        bad_candidate = base.copy()
        bad_candidate[-1] = "[0,"

        # Missing graph file.
        missing_graph = base.copy()
        missing_graph[3] = str(self.root / "missing.json")

        # Malformed graph JSON.
        malformed_graph_file = self.root / "malformed.json"
        malformed_graph_file.write_text(
            '{"0": [',
            encoding="utf-8",
        )
        malformed_graph = base.copy()
        malformed_graph[3] = str(malformed_graph_file)

        for name, args in (
            ("invalid_candidate", bad_candidate),
            ("missing_graph", missing_graph),
            ("malformed_graph", malformed_graph),
        ):
            with self.subTest(case=name):
                result = self.run_cli(*args)

                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
