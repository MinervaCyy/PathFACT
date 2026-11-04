"""Portable regression tests for local candidate dataset export."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pathfact.datasets.io import DatasetFormatError, iter_jsonl
from pathfact.datasets.manifest import verify_release_manifest
from pathfact.datasets.public import (
    EVALUATION_MODEL_INPUT_FIELDS,
    EVALUATION_PROVENANCE_FIELDS,
    EVALUATION_REFERENCE_FIELDS,
    EVALUATION_RELEASE_FILES,
    G06_FIELDS,
    G06_RELEASE_FILES,
    SAMPLING_FIELDS,
    STUDENT_FIELDS,
    TASK_FIELDS,
    TEACHER_FIELDS,
    TEACHER_METADATA_FIELDS,
    adapt_g08_evaluation_view,
    adapt_g09_evaluation_view,
    export_core_release_candidate,
    export_public_pairs,
    validate_joint_links,
)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generation_settings():
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "seed": 123,
        "client_concurrency": 1,
        "request_timeout_seconds": 30,
    }


class TestPublicDataset(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

        names = (
            "g00_path", "g02_path", "g04a_path",
            "g04b_path", "g05_path",
        )

        self.paths = {
            name: self.root / (name + ".jsonl")
            for name in names
        }
        self.data = {name: [] for name in names}

        for number, audited in ((1, True), (2, False)):
            pid = f"synthetic-case-{number}"

            task = dict.fromkeys(TASK_FIELDS)
            task.update({
                "production_case_id": pid,
                "topology": {
                    "0": {"1": 1},
                    "1": {"0": 1},
                },
                "node_pair": [0, 1],
                "ground_truth_shortest_path": [0, 1],
                "gold_cost": 1,
                "num_nodes": 2,
                "num_edges": 1,
                "case_instance_sha256": digest(f"case-{number}"),
                "graph_structure_sha256": digest(f"graph-{number}"),
                "weighted_graph_sha256": digest(f"weighted-{number}"),
                "internal_request_id": "PRIVATE_INFRA_SENTINEL",
            })

            self.data["g00_path"].append(task)

            wrong = f"Synthetic incorrect reasoning {number}"
            wrong_hash = digest(wrong)

            identity_fields = (
                "case_instance_sha256",
                "graph_structure_sha256",
                "weighted_graph_sha256",
                "node_pair",
                "gold_cost",
            )

            student_meta = dict.fromkeys(STUDENT_FIELDS)
            student_meta.update({
                "model_label": "synthetic-student",
                "prompt_version": "synthetic-task-v1",
                "generation_settings": generation_settings(),
                "predicted_path": [1, 0],
                "predicted_cost": 2,
                "reasoning_present": True,
                "answer_extracted": True,
                "stopped_normally": True,
                "token_limit_truncation": False,
                "vsp_valid": False,
                "raw_message": "PRIVATE_INFRA_SENTINEL",
                "request_id": "PRIVATE_INFRA_SENTINEL",
                "served_model_name": "PRIVATE_INFRA_SENTINEL",
            })

            for field in identity_fields:
                student_meta[field] = task[field]

            self.data["g02_path"].append({
                "production_case_id": pid,
                "canonical_wrong_reasoning_trace": wrong,
                "canonical_wrong_reasoning_trace_sha256": wrong_hash,
                "canonical_student_record": student_meta,
                "selection_reason": "synthetic-test",
            })

            targets = {}

            for role in ("gold", "contrastive"):
                profile = (
                    "vsp_plus_deterministic_rt_factuality_audit"
                    if audited and role == "gold"
                    else "vsp_plus_corrective_rt_factuality_validation"
                    if audited
                    else "solver_verified_vsp"
                )

                reasoning = f"Synthetic {role} reasoning {number}"
                answer = f"Synthetic path [0, 1] {number}"

                fields = {
                    "teacher_reasoning_trace": reasoning,
                    "teacher_reasoning_trace_sha256": digest(reasoning),
                    "teacher_final_content": answer,
                    "teacher_final_content_sha256": digest(answer),
                    "teacher_vsp_valid": True,
                    "validation_profile": profile,
                }

                metadata = dict.fromkeys(TEACHER_METADATA_FIELDS)
                metadata.update({
                    "model_label": "synthetic-teacher",
                    "prompt_version": f"synthetic-{role}-v1",
                    "generation_settings": generation_settings(),
                    "raw_message": "PRIVATE_INFRA_SENTINEL",
                    "served_model_name": "PRIVATE_INFRA_SENTINEL",
                })

                for field in (
                    "node_pair", "gold_cost", "graph_structure_sha256"
                ):
                    metadata[field] = task[field]

                record = {
                    "production_case_id": pid,
                    **fields,
                    "canonical_teacher_record": metadata,
                }

                target = dict(fields)

                if role == "contrastive":
                    record.update({
                        "canonical_wrong_reasoning_trace": wrong,
                        "canonical_wrong_reasoning_trace_sha256": wrong_hash,
                        "wrong_rt_continuity_verified": True,
                    })
                    target["wrong_rt_continuity_verified"] = True

                key = "g04a_path" if role == "gold" else "g04b_path"
                self.data[key].append(record)
                targets[role] = target

            self.data["g05_path"].append({
                "production_case_id": pid,
                "canonical_wrong_reasoning_trace": wrong,
                "canonical_wrong_reasoning_trace_sha256": wrong_hash,
                "gold_target": targets["gold"],
                "contrastive_target": targets["contrastive"],
                "joint_valid": True,
            })

        self.save()
        self.make_g06()
        self.make_evaluation_views()

    def save(self):
        for name, path in self.paths.items():
            with path.open("w", encoding="utf-8") as handle:
                for record in self.data[name]:
                    handle.write(
                        json.dumps(record, sort_keys=True) + "\n"
                    )

    def make_g06(self):
        self.g06_paths = {
            name: self.root / G06_RELEASE_FILES[name]
            for name in G06_RELEASE_FILES
        }
        specifications = {
            "gold_train": ("synthetic-case-1", "gold"),
            "contrastive_train": (
                "synthetic-case-1", "contrastive"
            ),
            "gold_loss_val": ("synthetic-case-2", "gold"),
            "contrastive_loss_val": (
                "synthetic-case-2", "contrastive"
            ),
        }
        self.g06_data = {}

        for name, (pid, condition) in specifications.items():
            prompt = f"Synthetic {condition} prompt for {pid}"
            completion = f"Synthetic {condition} completion for {pid}"
            self.g06_data[name] = [{
                "completion": completion,
                "completion_sha256": digest(completion),
                "condition": condition,
                "production_case_id": pid,
                "prompt": prompt,
                "prompt_sha256": digest(prompt),
            }]

        self.save_g06()

    def save_g06(self):
        for name, path in self.g06_paths.items():
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in self.g06_data[name]:
                    self.assertEqual(set(record), set(G06_FIELDS))
                    handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

    def evaluation_task(self, pid, number):
        task = dict.fromkeys(TASK_FIELDS)
        task.update({
            "production_case_id": pid,
            "topology": {
                "0": [[1, number]],
                "1": [[0, number]],
            },
            "node_pair": [0, 1],
            "ground_truth_shortest_path": [0, 1],
            "gold_cost": number,
            "num_nodes": 2,
            "num_edges": 1,
            "topology_type": "synthetic",
            "topology_dimension": {"m": 1, "n": 2},
            "weight_distribution": "synthetic",
            "gold_path_hops": 1,
            "hop_bucket": "1-3",
            "selected_exact_hop": 1,
            "min_optimal_hops": 1,
            "max_optimal_hops": 1,
            "min_total_cost": number,
            "feasible_exact_hops": [1],
            "eligible_pair_count_total": 1,
            "eligible_pair_count_selected_exact_hop": 1,
            "graph_seed": number,
            "weight_seed": number + 10,
            "pair_seed": number + 20,
            "generation_attempt_index": 0,
            "attempts_to_accept": 1,
            "replicate": 0,
            "slot_id": f"synthetic-slot-{number}",
            "support_row": number,
            "pair_orientation_swapped": False,
            "graph_structure_sha256": digest(f"eval-graph-{pid}"),
            "weighted_graph_sha256": digest(f"eval-weighted-{pid}"),
            "case_instance_sha256": digest(f"eval-case-{pid}"),
        })
        return task

    def make_evaluation_views(self):
        self.evaluation_paths = {
            name: self.root / f"{name}.jsonl"
            for name in EVALUATION_RELEASE_FILES
        }
        tasks = {
            f"eval-{number}": self.evaluation_task(
                f"eval-{number}", number
            )
            for number in range(1, 7)
        }

        def g08(pid, source_view, difficulty):
            task = tasks[pid]
            return {
                "production_case_id": pid,
                "task_record": task,
                "g08_dataset": source_view,
                "difficulty": difficulty,
                "gold_path_hops": task["gold_path_hops"],
                "canonical_student_request_id": (
                    "PRIVATE_INFRA_SENTINEL"
                ),
                "canonical_wrong_reasoning_trace": (
                    "PRIVATE_INFRA_SENTINEL"
                ),
                "internal_file_path": "PRIVATE_INFRA_SENTINEL",
            }

        def g09(pid):
            return {
                **tasks[pid],
                "raw_inference_log": "PRIVATE_INFRA_SENTINEL",
                "internal_file_path": "PRIVATE_INFRA_SENTINEL",
            }

        self.evaluation_tasks = tasks
        self.evaluation_data = {
            "g08_balanced": [
                g08("eval-1", "balanced_300", "Easy"),
                g08("eval-2", "balanced_300", "Medium"),
            ],
            "g08_train_dist": [
                g08("eval-3", "train_distribution_300", "Easy"),
                g08("eval-4", "train_distribution_300", "Hard"),
            ],
            "g09_balanced": [
                g09("eval-1"), g09("eval-3"), g09("eval-5"),
            ],
            "g09_train_dist": [
                g09("eval-2"), g09("eval-4"), g09("eval-5"),
            ],
        }
        self.save_evaluation_views()

    def save_evaluation_views(self):
        for name, path in self.evaluation_paths.items():
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in self.evaluation_data[name]:
                    handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )

    def core_arguments(self, output):
        return {
            **self.paths,
            "g06_gold_train_path": self.g06_paths["gold_train"],
            "g06_contrastive_train_path": (
                self.g06_paths["contrastive_train"]
            ),
            "g06_gold_loss_val_path": (
                self.g06_paths["gold_loss_val"]
            ),
            "g06_contrastive_loss_val_path": (
                self.g06_paths["contrastive_loss_val"]
            ),
            "g08_balanced_path": (
                self.evaluation_paths["g08_balanced"]
            ),
            "g08_train_dist_path": (
                self.evaluation_paths["g08_train_dist"]
            ),
            "g09_balanced_path": (
                self.evaluation_paths["g09_balanced"]
            ),
            "g09_train_dist_path": (
                self.evaluation_paths["g09_train_dist"]
            ),
            "output_dir": output,
            "dataset_version": "synthetic-core-v0.1",
            "expected_g00_records": 2,
            "expected_paired_records": 2,
            "expected_train_records": 1,
            "expected_loss_val_records": 1,
            "expected_reasoning_validated_pairs": 1,
            "expected_vsp_only_pairs": 1,
            "expected_g08_balanced_records": 2,
            "expected_g08_train_dist_records": 2,
            "expected_g09_balanced_records": 3,
            "expected_g09_train_dist_records": 3,
            "expected_g09_overlap": 1,
            "expected_g08_balanced_g09_balanced_overlap": 1,
            "expected_g08_balanced_g09_train_dist_overlap": 1,
            "expected_g08_train_dist_g09_balanced_overlap": 1,
            "expected_g08_train_dist_g09_train_dist_overlap": 1,
        }

    def validate(self):
        return validate_joint_links(
            **self.paths, expected_records=2
        )

    def test_valid_linkage_and_tiers(self):
        self.assertEqual(
            self.validate(),
            {
                "case_count": 2,
                "validation_tiers": {
                    "both_reasoning_factuality_validated": 1,
                    "both_vsp_only": 1,
                },
            },
        )

    def test_export_allowlist(self):
        output = self.root / "candidate"

        paths = export_public_pairs(
            **self.paths, output_dir=output, expected_records=2
        )

        rows = list(iter_jsonl(
            paths["paired"],
            unique_key="production_case_id",
        ))

        self.assertEqual(len(rows), 2)

        for row in rows:
            self.assertEqual(set(row["task"]), set(TASK_FIELDS))
            self.assertEqual(
                set(row["student"]),
                set(STUDENT_FIELDS) | {
                    "reasoning_trace",
                    "reasoning_trace_sha256",
                    "selection_reason",
                },
            )

            self.assertEqual(
                set(row["student"]["generation_settings"]),
                set(SAMPLING_FIELDS),
            )

            for role, teacher in row["teacher_targets"].items():
                expected = (
                    set(TEACHER_FIELDS)
                    | set(TEACHER_METADATA_FIELDS)
                )

                if role == "contrastive":
                    expected.add("wrong_rt_continuity_verified")

                self.assertEqual(set(teacher), expected)
                self.assertEqual(
                    set(teacher["generation_settings"]),
                    set(SAMPLING_FIELDS),
                )

        exported_text = paths["paired"].read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE_INFRA_SENTINEL", exported_text)

        summary = json.loads(paths["summary"].read_text())
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(
            summary["release_status"],
            "local_candidate_not_approved",
        )

    def test_cli_export(self):
        output = self.root / "cli-output"

        command = [
            sys.executable, "-m", "pathfact.cli",
            "dataset", "export-public",
        ]

        for name, path in self.paths.items():
            command.extend([
                "--" + name.removesuffix("_path"), str(path)
            ])

        command.extend([
            "--output-dir", str(output),
            "--expected-records", "2",
        ])

        result = subprocess.run(
            command, capture_output=True, text=True
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "paired.jsonl").is_file())

    def test_missing_upstream_record(self):
        self.data["g04b_path"].pop()
        self.save()

        with self.assertRaises(DatasetFormatError):
            self.validate()

    def test_student_trace_mismatch(self):
        self.data["g05_path"][0][
            "canonical_wrong_reasoning_trace"
        ] = "Changed reasoning"
        self.save()

        with self.assertRaises(DatasetFormatError):
            self.validate()

    def test_correct_student_answer_rejected(self):
        student = self.data["g02_path"][0]["canonical_student_record"]
        student["vsp_valid"] = True
        self.save()

        with self.assertRaisesRegex(
            DatasetFormatError, "Student vsp_valid"
        ):
            self.validate()

    def test_truncated_student_rejected(self):
        student = self.data["g02_path"][0]["canonical_student_record"]
        student["token_limit_truncation"] = True
        self.save()

        with self.assertRaisesRegex(
            DatasetFormatError, "Student token_limit_truncation"
        ):
            self.validate()

    def test_missing_answer_extraction_rejected(self):
        student = self.data["g02_path"][0]["canonical_student_record"]
        student["answer_extracted"] = False
        self.save()

        with self.assertRaisesRegex(
            DatasetFormatError, "Student answer_extracted"
        ):
            self.validate()

    def test_empty_student_reasoning_rejected(self):
        for name in ("g02_path", "g04b_path", "g05_path"):
            record = self.data[name][0]
            record["canonical_wrong_reasoning_trace"] = ""
            record["canonical_wrong_reasoning_trace_sha256"] = digest("")
        self.save()

        with self.assertRaisesRegex(
            DatasetFormatError, "empty Student reasoning trace"
        ):
            self.validate()

    def test_teacher_text_hash_mismatch(self):
        self.data["g04a_path"][0][
            "teacher_reasoning_trace_sha256"
        ] = "0" * 64
        self.data["g05_path"][0]["gold_target"][
            "teacher_reasoning_trace_sha256"
        ] = "0" * 64
        self.save()

        with self.assertRaisesRegex(
            DatasetFormatError, "SHA-256 mismatch"
        ):
            self.validate()

    def test_invalid_vsp(self):
        self.data["g04a_path"][0]["teacher_vsp_valid"] = False
        self.data["g05_path"][0]["gold_target"][
            "teacher_vsp_valid"
        ] = False
        self.save()

        with self.assertRaises(DatasetFormatError):
            self.validate()

    def test_unknown_validation_profile(self):
        self.data["g04a_path"][0][
            "validation_profile"
        ] = "unrecognized"
        self.data["g05_path"][0]["gold_target"][
            "validation_profile"
        ] = "unrecognized"
        self.save()

        with self.assertRaises(DatasetFormatError):
            self.validate()

    def test_duplicate_case(self):
        self.data["g05_path"].append(
            dict(self.data["g05_path"][0])
        )
        self.save()

        with self.assertRaisesRegex(
            DatasetFormatError, "duplicate production_case_id"
        ):
            self.validate()

    def test_export_failure_cleans_partial_output(self):
        # The second record is malformed only for export.
        # Linkage validation succeeds, then export must clean up.
        student = self.data["g02_path"][1]["canonical_student_record"]
        student["generation_settings"].pop("top_p")
        self.save()

        output = self.root / "incomplete-candidate"
        before = set(self.root.iterdir())

        with self.assertRaisesRegex(
            DatasetFormatError, "missing fields"
        ):
            export_public_pairs(
                **self.paths,
                output_dir=output,
                expected_records=2,
            )

        self.assertFalse(output.exists())
        self.assertEqual(set(self.root.iterdir()), before)

    def test_export_is_byte_deterministic(self):
        first = export_public_pairs(
            **self.paths,
            output_dir=self.root / "candidate-a",
            expected_records=2,
        )

        second = export_public_pairs(
            **self.paths,
            output_dir=self.root / "candidate-b",
            expected_records=2,
        )

        for name in ("paired", "summary"):
            self.assertEqual(
                first[name].read_bytes(),
                second[name].read_bytes(),
            )

    def test_export_summary_hashes(self):
        output = self.root / "hashed-candidate"

        outputs = export_public_pairs(
            **self.paths,
            output_dir=output,
            expected_records=2,
        )

        summary = json.loads(
            outputs["summary"].read_text(encoding="utf-8")
        )

        expected_inputs = {
            name.removesuffix("_path"):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in self.paths.items()
        }

        self.assertEqual(
            summary["input_sha256"],
            expected_inputs,
        )

        self.assertEqual(
            summary["paired_sha256"],
            hashlib.sha256(
                outputs["paired"].read_bytes()
            ).hexdigest(),
        )

    def test_input_change_after_validation(self):
        from unittest.mock import patch

        original_validate = validate_joint_links

        def validate_then_modify(**kwargs):
            result = original_validate(**kwargs)

            # Modify only synthetic test data after validation.
            self.data["g00_path"][0][
                "post_validation_test_marker"
            ] = "changed"

            self.save()
            return result

        output = self.root / "changed-source-candidate"

        with patch(
            "pathfact.datasets.public.validate_joint_links",
            side_effect=validate_then_modify,
        ):
            with self.assertRaisesRegex(
                DatasetFormatError,
                "Source files changed during export: g00",
            ):
                export_public_pairs(
                    **self.paths,
                    output_dir=output,
                    expected_records=2,
                )

        self.assertFalse(output.exists())
        self.assertFalse(
            list(self.root.glob(".pathfact-candidate-*"))
        )

    def test_existing_output_is_preserved(self):
        output = self.root / "candidate"
        output.mkdir()

        sentinel = output / "existing.txt"
        sentinel.write_text("Do not overwrite", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            export_public_pairs(
                **self.paths,
                output_dir=output,
                expected_records=2,
            )

        self.assertEqual(sentinel.read_text(), "Do not overwrite")

    def test_g08_evaluation_adapter_allowlist(self):
        adapted = adapt_g08_evaluation_view(
            self.evaluation_paths["g08_balanced"],
            view_name="balanced",
            expected_records=2,
        )
        self.assertEqual(adapted["record_count"], 2)

        for record in adapted["records"]:
            self.assertEqual(
                set(record),
                {
                    "schema_version", "production_case_id", "view",
                    "model_input", "reference", "provenance",
                },
            )
            self.assertEqual(
                set(record["model_input"]),
                set(EVALUATION_MODEL_INPUT_FIELDS),
            )
            self.assertEqual(
                set(record["reference"]),
                set(EVALUATION_REFERENCE_FIELDS),
            )
            self.assertEqual(
                set(record["provenance"]),
                set(EVALUATION_PROVENANCE_FIELDS),
            )
            self.assertEqual(record["view"]["source_stage"], "G08")
            self.assertEqual(record["view"]["name"], "balanced")
            self.assertIsInstance(
                record["view"]["source_difficulty"], str
            )

        self.assertNotIn(
            "PRIVATE_INFRA_SENTINEL",
            json.dumps(adapted["records"], sort_keys=True),
        )

    def test_g09_evaluation_adapter_allowlist(self):
        adapted = adapt_g09_evaluation_view(
            self.evaluation_paths["g09_balanced"],
            view_name="balanced",
            expected_records=3,
        )
        self.assertEqual(adapted["record_count"], 3)

        for record in adapted["records"]:
            self.assertEqual(record["view"], {
                "source_stage": "G09",
                "name": "balanced",
            })
            self.assertEqual(
                set(record["model_input"]),
                set(EVALUATION_MODEL_INPUT_FIELDS),
            )
            self.assertEqual(
                set(record["reference"]),
                set(EVALUATION_REFERENCE_FIELDS),
            )
            self.assertEqual(
                set(record["provenance"]),
                set(EVALUATION_PROVENANCE_FIELDS),
            )

        self.assertNotIn(
            "PRIVATE_INFRA_SENTINEL",
            json.dumps(adapted["records"], sort_keys=True),
        )

    def test_evaluation_adapters_reject_missing_and_duplicate_ids(self):
        self.evaluation_data["g08_balanced"][0].pop("task_record")
        self.save_evaluation_views()

        with self.assertRaisesRegex(DatasetFormatError, "missing fields"):
            adapt_g08_evaluation_view(
                self.evaluation_paths["g08_balanced"],
                view_name="balanced",
                expected_records=2,
            )

        self.make_evaluation_views()
        self.evaluation_data["g09_balanced"].append(
            dict(self.evaluation_data["g09_balanced"][0])
        )
        self.save_evaluation_views()

        with self.assertRaisesRegex(
            DatasetFormatError, "duplicate production_case_id"
        ):
            adapt_g09_evaluation_view(
                self.evaluation_paths["g09_balanced"],
                view_name="balanced",
                expected_records=4,
            )

    def test_core_candidate_layout_and_integrity(self):
        output = self.root / "core-candidate"
        source_bytes = {
            **{
                name: path.read_bytes()
                for name, path in self.paths.items()
            },
            **{
                f"g06_{name}": path.read_bytes()
                for name, path in self.g06_paths.items()
            },
            **{
                name: path.read_bytes()
                for name, path in self.evaluation_paths.items()
            },
        }
        g06_source_bytes = {
            name: path.read_bytes()
            for name, path in self.g06_paths.items()
        }

        outputs = export_core_release_candidate(
            **self.core_arguments(output)
        )

        self.assertEqual(
            verify_release_manifest(output),
            {
                "files_verified": 10,
                "status": "PASS",
                "release_status": "draft_not_approved",
            },
        )

        task_rows = list(iter_jsonl(
            outputs["tasks"], unique_key="production_case_id"
        ))
        self.assertEqual(len(task_rows), 2)
        self.assertTrue(all(
            set(row) == set(TASK_FIELDS) for row in task_rows
        ))

        paired_rows = list(iter_jsonl(
            outputs["paired"], unique_key="production_case_id"
        ))
        self.assertEqual(len(paired_rows), 2)
        self.assertEqual(
            {row["validation_tier"] for row in paired_rows},
            {
                "both_reasoning_factuality_validated",
                "both_vsp_only",
            },
        )

        source_pairs = {
            row["production_case_id"]: row
            for row in self.data["g05_path"]
        }

        for row in paired_rows:
            source = source_pairs[row["production_case_id"]]
            self.assertEqual(
                row["student"]["reasoning_trace"],
                source["canonical_wrong_reasoning_trace"],
            )

            for condition in ("gold", "contrastive"):
                target = row["teacher_targets"][condition]
                source_target = source[f"{condition}_target"]

                for field in (
                    "teacher_reasoning_trace",
                    "teacher_final_content",
                ):
                    self.assertEqual(
                        target[field], source_target[field]
                    )

        public_text = (
            outputs["tasks"].read_text(encoding="utf-8")
            + outputs["paired"].read_text(encoding="utf-8")
            + "".join(
                outputs[name].read_text(encoding="utf-8")
                for name in EVALUATION_RELEASE_FILES
            )
        )
        self.assertNotIn("PRIVATE_INFRA_SENTINEL", public_text)

        for name, original in g06_source_bytes.items():
            self.assertEqual(outputs[name].read_bytes(), original)
            self.assertEqual(self.g06_paths[name].read_bytes(), original)

        for name, path in self.paths.items():
            self.assertEqual(path.read_bytes(), source_bytes[name])

        for name, path in self.g06_paths.items():
            self.assertEqual(
                path.read_bytes(), source_bytes[f"g06_{name}"]
            )

        for name, path in self.evaluation_paths.items():
            self.assertEqual(path.read_bytes(), source_bytes[name])

        evaluation_rows = {
            name: list(iter_jsonl(
                outputs[name], unique_key="production_case_id"
            ))
            for name in EVALUATION_RELEASE_FILES
        }
        self.assertEqual(
            {name: len(rows) for name, rows in evaluation_rows.items()},
            {
                "g08_balanced": 2,
                "g08_train_dist": 2,
                "g09_balanced": 3,
                "g09_train_dist": 3,
            },
        )
        self.assertEqual(
            sum(len(rows) for rows in evaluation_rows.values()), 10
        )

        manifest = json.loads(
            outputs["manifest"].read_text(encoding="utf-8")
        )
        evaluation_entries = {
            entry["path"]: entry
            for entry in manifest["files"]
            if entry["path"].startswith("evaluation/")
        }
        self.assertEqual(len(evaluation_entries), 4)
        expected_evaluation_counts = {
            "g08_balanced": 2,
            "g08_train_dist": 2,
            "g09_balanced": 3,
            "g09_train_dist": 3,
        }

        for name, filename in EVALUATION_RELEASE_FILES.items():
            entry = evaluation_entries[f"evaluation/{filename}"]
            self.assertEqual(
                entry["source_stage"],
                "G08" if name.startswith("g08") else "G09",
            )
            self.assertEqual(
                entry["schema_version"],
                "pathfact-evaluation-v0.1",
            )
            self.assertEqual(
                entry["sha256"],
                hashlib.sha256(outputs[name].read_bytes()).hexdigest(),
            )
            self.assertEqual(
                entry["source_sha256"],
                hashlib.sha256(
                    self.evaluation_paths[name].read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                entry["record_count"],
                expected_evaluation_counts[name],
            )
            self.assertEqual(
                entry["size_bytes"], outputs[name].stat().st_size
            )

        summary = json.loads(
            outputs["summary"].read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["release_status"],
            "local_candidate_not_approved",
        )
        self.assertIsNone(summary["license"])
        self.assertEqual(summary["g00"]["record_count"], 2)
        self.assertEqual(summary["g05"]["case_count"], 2)
        self.assertEqual(summary["g06"]["unique_case_count"], 2)
        self.assertEqual(
            summary["evaluation"]["overlaps"],
            {
                "g09_balanced_train_dist": 1,
                "g08_balanced_g09_balanced": 1,
                "g08_balanced_g09_train_dist": 1,
                "g08_train_dist_g09_balanced": 1,
                "g08_train_dist_g09_train_dist": 1,
            },
        )
        self.assertEqual(
            summary["evaluation"]["g06_overlap_count"], 0
        )
        self.assertEqual(
            summary["g05"]["validation_tiers"],
            {
                "both_reasoning_factuality_validated": 1,
                "both_vsp_only": 1,
            },
        )

    def test_core_candidate_is_byte_deterministic(self):
        first = self.root / "core-a"
        second = self.root / "core-b"

        export_core_release_candidate(
            **self.core_arguments(first)
        )
        export_core_release_candidate(
            **self.core_arguments(second)
        )

        first_files = sorted(
            path.relative_to(first)
            for path in first.rglob("*")
            if path.is_file()
        )
        second_files = sorted(
            path.relative_to(second)
            for path in second.rglob("*")
            if path.is_file()
        )
        self.assertEqual(first_files, second_files)

        for relative in first_files:
            self.assertEqual(
                (first / relative).read_bytes(),
                (second / relative).read_bytes(),
            )

    def test_core_candidate_rejects_bad_g06_hash(self):
        record = self.g06_data["gold_train"][0]
        record["prompt_sha256"] = "0" * 64
        self.save_g06()
        output = self.root / "bad-g06"

        with self.assertRaisesRegex(
            DatasetFormatError, "text SHA-256 mismatch"
        ):
            export_core_release_candidate(
                **self.core_arguments(output)
            )

        self.assertFalse(output.exists())
        self.assertFalse(list(
            self.root.glob(".pathfact-core-candidate-*")
        ))

    def test_core_candidate_enforces_validation_tier_counts(self):
        arguments = self.core_arguments(self.root / "wrong-tiers")
        arguments["expected_reasoning_validated_pairs"] = 2
        arguments["expected_vsp_only_pairs"] = 0

        with self.assertRaisesRegex(
            DatasetFormatError,
            "Unexpected G05 validation tier counts",
        ):
            export_core_release_candidate(**arguments)

    def test_core_candidate_rejects_unexpected_evaluation_overlaps(self):
        self.evaluation_data["g09_train_dist"][-1] = {
            **self.evaluation_tasks["eval-6"],
            "raw_inference_log": "PRIVATE_INFRA_SENTINEL",
        }
        self.save_evaluation_views()

        with self.assertRaisesRegex(
            DatasetFormatError,
            "Unexpected evaluation view overlaps",
        ):
            export_core_release_candidate(
                **self.core_arguments(self.root / "wrong-overlap")
            )

    def test_core_candidate_rejects_g08_view_overlap(self):
        source = self.evaluation_data["g08_balanced"][0]
        replacement = dict(source)
        replacement["g08_dataset"] = "train_distribution_300"
        self.evaluation_data["g08_train_dist"][0] = replacement
        self.save_evaluation_views()

        with self.assertRaisesRegex(
            DatasetFormatError, "G08 evaluation views must be disjoint"
        ):
            export_core_release_candidate(
                **self.core_arguments(self.root / "g08-overlap")
            )

    def test_core_candidate_rejects_inconsistent_eval_reference(self):
        self.evaluation_data["g09_balanced"][0]["gold_cost"] = 999
        self.save_evaluation_views()

        with self.assertRaisesRegex(
            DatasetFormatError,
            "Inconsistent evaluation task/reference",
        ):
            export_core_release_candidate(
                **self.core_arguments(self.root / "bad-reference")
            )

    def test_core_candidate_rejects_g06_evaluation_overlap(self):
        task = self.evaluation_task("synthetic-case-1", 7)
        g08 = self.evaluation_data["g08_balanced"][0]
        g08["production_case_id"] = "synthetic-case-1"
        g08["task_record"] = task
        g08["gold_path_hops"] = task["gold_path_hops"]
        self.evaluation_data["g09_balanced"][0] = {
            **task,
            "raw_inference_log": "PRIVATE_INFRA_SENTINEL",
        }
        self.save_evaluation_views()

        with self.assertRaisesRegex(
            DatasetFormatError,
            "Evaluation views overlap G06 Train/LossVal",
        ):
            export_core_release_candidate(
                **self.core_arguments(self.root / "g06-overlap")
            )

    def test_core_candidate_preserves_existing_output(self):
        output = self.root / "existing-core"
        output.mkdir()
        sentinel = output / "existing.txt"
        sentinel.write_text("Do not overwrite", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            export_core_release_candidate(
                **self.core_arguments(output)
            )

        self.assertEqual(sentinel.read_text(), "Do not overwrite")

    def test_cli_core_candidate(self):
        output = self.root / "cli-core"
        command = [
            sys.executable, "-m", "pathfact.cli",
            "dataset", "export-core-candidate",
        ]

        for name, path in self.paths.items():
            command.extend([
                "--" + name.removesuffix("_path"), str(path)
            ])

        for name, flag in (
            ("gold_train", "--g06-gold-train"),
            ("contrastive_train", "--g06-contrastive-train"),
            ("gold_loss_val", "--g06-gold-loss-val"),
            (
                "contrastive_loss_val",
                "--g06-contrastive-loss-val",
            ),
        ):
            command.extend([flag, str(self.g06_paths[name])])

        for name, flag in (
            ("g08_balanced", "--g08-balanced"),
            ("g08_train_dist", "--g08-train-dist"),
            ("g09_balanced", "--g09-balanced"),
            ("g09_train_dist", "--g09-train-dist"),
        ):
            command.extend([flag, str(self.evaluation_paths[name])])

        command.extend([
            "--output-dir", str(output),
            "--dataset-version", "synthetic-cli-v0.1",
            "--expected-g00-records", "2",
            "--expected-paired-records", "2",
            "--expected-train-records", "1",
            "--expected-loss-val-records", "1",
            "--expected-reasoning-validated-pairs", "1",
            "--expected-vsp-only-pairs", "1",
            "--expected-g08-balanced-records", "2",
            "--expected-g08-train-dist-records", "2",
            "--expected-g09-balanced-records", "3",
            "--expected-g09-train-dist-records", "3",
            "--expected-g09-overlap", "1",
            "--expected-g08-balanced-g09-balanced-overlap", "1",
            "--expected-g08-balanced-g09-train-dist-overlap", "1",
            "--expected-g08-train-dist-g09-balanced-overlap", "1",
            "--expected-g08-train-dist-g09-train-dist-overlap", "1",
        ])

        result = subprocess.run(
            command, capture_output=True, text=True
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Redistribution has not been approved.", result.stdout
        )
        self.assertEqual(
            verify_release_manifest(output)["files_verified"], 10
        )


if __name__ == "__main__":
    unittest.main()
