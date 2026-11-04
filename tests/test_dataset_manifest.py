"""Portable tests for PathFACT dataset release manifests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pathfact.datasets.io import DatasetFormatError
from pathfact.datasets.manifest import (
    create_release_manifest,
    verify_release_manifest,
)


class TestReleaseManifest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name) / "release"
        (self.root / "tasks").mkdir(parents=True)
        (self.root / "paired").mkdir()

        self.task_file = self.root / "tasks" / "tasks.jsonl"
        self.pair_file = self.root / "paired" / "pairs.jsonl"

        self.tasks = [
            {"production_case_id": "synthetic-1", "gold_cost": 1},
            {"production_case_id": "synthetic-2", "gold_cost": 2},
        ]

        self.pairs = [
            self.pair("synthetic-1", True),
            self.pair("synthetic-2", False),
        ]

        self.write_records(self.task_file, self.tasks)
        self.write_records(self.pair_file, self.pairs)

        self.artifacts = [
            {
                "path": "tasks/tasks.jsonl",
                "source_stage": "G00",
                "schema_version": "synthetic-task-v1",
                "source_sha256": "a" * 64,
            },
            {
                "path": "paired/pairs.jsonl",
                "source_stage": "G05",
                "schema_version": "pathfact-paired-v0.1",
                "source_sha256": {
                    "g00": "a" * 64,
                    "g05": "b" * 64,
                },
            },
        ]

    @staticmethod
    def pair(pid, audited):
        if audited:
            gold_profile = (
                "vsp_plus_deterministic_rt_factuality_audit"
            )
            contrastive_profile = (
                "vsp_plus_corrective_rt_factuality_validation"
            )
            tier = "both_reasoning_factuality_validated"
        else:
            gold_profile = "solver_verified_vsp"
            contrastive_profile = "solver_verified_vsp"
            tier = "both_vsp_only"

        return {
            "production_case_id": pid,
            "joint_valid": True,
            "validation_tier": tier,
            "teacher_targets": {
                "gold": {
                    "teacher_vsp_valid": True,
                    "validation_profile": gold_profile,
                },
                "contrastive": {
                    "teacher_vsp_valid": True,
                    "validation_profile": contrastive_profile,
                },
            },
        }

    @staticmethod
    def write_records(path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True) + "\n"
                )

    def create(self):
        return create_release_manifest(
            self.root,
            self.artifacts,
            "synthetic-test-v1",
        )

    def test_create_and_verify(self):
        path = self.create()
        manifest = json.loads(path.read_text())

        self.assertEqual(
            manifest["release_status"], "draft_not_approved"
        )
        self.assertIsNone(manifest["license"])
        self.assertEqual(len(manifest["files"]), 2)

        entries = {
            item["path"]: item
            for item in manifest["files"]
        }

        paired = entries["paired/pairs.jsonl"]

        self.assertEqual(paired["record_count"], 2)
        self.assertEqual(
            paired["validation_profile"],
            {
                "both_reasoning_factuality_validated": 1,
                "both_vsp_only": 1,
            },
        )
        self.assertEqual(
            paired["sha256"],
            hashlib.sha256(self.pair_file.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            paired["size_bytes"], self.pair_file.stat().st_size
        )
        self.assertEqual(
            paired["source_sha256"],
            {"g00": "a" * 64, "g05": "b" * 64},
        )

        result = verify_release_manifest(self.root)
        self.assertEqual(result["files_verified"], 2)
        self.assertEqual(result["status"], "PASS")

    def test_existing_manifest_protected(self):
        path = self.create()
        original = path.read_bytes()

        with self.assertRaises(FileExistsError):
            self.create()

        self.assertEqual(path.read_bytes(), original)

    def test_modified_file_detected(self):
        self.create()

        with self.task_file.open("a", encoding="utf-8") as handle:
            handle.write(
                '{"production_case_id":"synthetic-3"}\n'
            )

        with self.assertRaises(DatasetFormatError):
            verify_release_manifest(self.root)

    def test_missing_file_detected(self):
        self.create()
        self.task_file.unlink()

        with self.assertRaises(DatasetFormatError):
            verify_release_manifest(self.root)

    def test_duplicate_case_rejected(self):
        self.tasks.append(dict(self.tasks[0]))
        self.write_records(self.task_file, self.tasks)

        with self.assertRaisesRegex(
            DatasetFormatError, "duplicate production_case_id"
        ):
            self.create()

    def test_wrong_g05_tier_rejected(self):
        self.pairs[0]["validation_tier"] = "both_vsp_only"
        self.write_records(self.pair_file, self.pairs)

        with self.assertRaisesRegex(
            DatasetFormatError, "G05 validation tier mismatch"
        ):
            self.create()

    def test_parent_traversal_rejected(self):
        self.artifacts[0]["path"] = (
            "tasks/../paired/pairs.jsonl"
        )

        with self.assertRaisesRegex(
            DatasetFormatError, "Unsafe release artifact path"
        ):
            self.create()

    def test_symlink_rejected(self):
        link = self.root / "paired" / "linked.jsonl"
        link.symlink_to(self.pair_file)

        self.artifacts[1]["path"] = "paired/linked.jsonl"

        with self.assertRaisesRegex(
            DatasetFormatError, "Symlink not allowed"
        ):
            self.create()

    def test_invalid_source_hash_rejected(self):
        self.artifacts[0]["source_sha256"] = "invalid"

        with self.assertRaisesRegex(
            DatasetFormatError, "Invalid source SHA-256"
        ):
            self.create()

    def test_g05_stage_mismatch_on_create(self):
        self.artifacts[1]["source_stage"] = "G00"

        with self.assertRaisesRegex(
            DatasetFormatError, "G05 source_stage/path mismatch"
        ):
            self.create()

    def test_g05_stage_mismatch_on_verify(self):
        path = self.create()
        manifest = json.loads(path.read_text())

        paired = next(
            item for item in manifest["files"]
            if item["path"] == "paired/pairs.jsonl"
        )
        paired["source_stage"] = "G00"
        paired["validation_profile"] = None

        path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(
            DatasetFormatError, "G05 source_stage/path mismatch"
        ):
            verify_release_manifest(self.root)

    def test_manifest_cannot_self_approve(self):
        path = self.create()
        manifest = json.loads(path.read_text())

        manifest["release_status"] = "approved"
        manifest["license"] = "CC-BY-4.0"

        path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(
            DatasetFormatError, "draft-only"
        ):
            verify_release_manifest(self.root)

    def test_cli_create_verify_and_detect_tampering(self):
        import subprocess
        import sys

        specs_file = self.root.parent / "artifacts.json"
        specs_file.write_text(
            json.dumps(self.artifacts),
            encoding="utf-8",
        )

        cli = [
            sys.executable, "-m", "pathfact.cli", "dataset"
        ]

        create = subprocess.run(
            cli + [
                "manifest-create",
                "--root-dir", str(self.root),
                "--artifacts-json", str(specs_file),
                "--dataset-version", "synthetic-v1",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(create.returncode, 0, create.stderr)

        verify_command = cli + [
            "manifest-verify",
            "--root-dir", str(self.root),
        ]

        verified = subprocess.run(
            verify_command,
            capture_output=True,
            text=True,
        )

        self.assertEqual(verified.returncode, 0, verified.stderr)

        report = json.loads(verified.stdout)
        self.assertEqual(report["files_verified"], 2)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["release_status"], "draft_not_approved"
        )

        manifest_file = self.root / "manifest.json"
        manifest_before = manifest_file.read_bytes()

        self.write_records(
            self.task_file,
            self.tasks + [{
                "production_case_id": "synthetic-3",
                "gold_cost": 3,
            }],
        )

        tampered = subprocess.run(
            verify_command,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(tampered.returncode, 0)
        self.assertIn("mismatch", tampered.stderr)
        self.assertEqual(
            manifest_file.read_bytes(),
            manifest_before,
        )

    def test_manifest_path_tampering_rejected(self):
        path = self.create()
        manifest = json.loads(path.read_text())

        manifest["files"][0]["path"] = (
            "tasks/../paired/pairs.jsonl"
        )
        path.write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        with self.assertRaises(DatasetFormatError):
            verify_release_manifest(self.root)


if __name__ == "__main__":
    unittest.main()
