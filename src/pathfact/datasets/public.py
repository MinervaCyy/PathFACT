"""Read-only cross-stage validation for candidate public paired datasets."""

import hashlib
from collections import Counter

from pathfact.datasets.io import DatasetFormatError, iter_jsonl


TEACHER_FIELDS = (
    "teacher_reasoning_trace",
    "teacher_reasoning_trace_sha256",
    "teacher_final_content",
    "teacher_final_content_sha256",
    "teacher_vsp_valid",
    "validation_profile",
)


def _select(record, fields, label):
    if not isinstance(record, dict):
        raise DatasetFormatError(f"{label}: expected an object")
    missing = sorted(set(fields) - record.keys())
    if missing:
        raise DatasetFormatError(f"{label}: missing fields {missing}")
    return {name: record[name] for name in fields}


def _index(path, fields, nested=None, nested_fields=()):
    result = {}
    for row in iter_jsonl(path, unique_key="production_case_id"):
        pid = row["production_case_id"]
        item = _select(row, fields, f"{path}:{pid}")
        if nested:
            item[nested] = _select(
                row.get(nested), nested_fields, f"{path}:{pid}:{nested}"
            )
        result[pid] = item
    return result


def _equal(left, right, fields, label):
    for name in fields:
        if left.get(name) != right.get(name):
            raise DatasetFormatError(f"{label}: mismatch in {name}")


def _hash_text(text, digest, label):
    if not isinstance(text, str) or not isinstance(digest, str):
        raise DatasetFormatError(f"{label}: expected text and SHA-256")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
        raise DatasetFormatError(f"{label}: text SHA-256 mismatch")


def _sha256_file(path):
    from pathlib import Path

    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _check_student_failure(student, label):
    """Require a complete, normally terminated Student failure."""
    expected = {
        "vsp_valid": False,
        "answer_extracted": True,
        "reasoning_present": True,
        "stopped_normally": True,
        "token_limit_truncation": False,
    }

    for field, value in expected.items():
        if student.get(field) is not value:
            raise DatasetFormatError(
                f"{label}: Student {field} must be {value}"
            )

    if not isinstance(student.get("predicted_path"), list):
        raise DatasetFormatError(
            f"{label}: Student predicted_path must be a list"
        )


def _teacher(source, target, task, label):
    target = _select(target, TEACHER_FIELDS, label)
    _equal(source, target, TEACHER_FIELDS, label)
    _equal(
        task, source["canonical_teacher_record"],
        ("node_pair", "gold_cost", "graph_structure_sha256"), label,
    )
    if target["teacher_vsp_valid"] is not True:
        raise DatasetFormatError(f"{label}: Teacher VSP not true")

    for field, digest in (
        ("teacher_reasoning_trace", "teacher_reasoning_trace_sha256"),
        ("teacher_final_content", "teacher_final_content_sha256"),
    ):
        _hash_text(target[field], target[digest], f"{label}:{field}")

    return target["validation_profile"]


def validate_joint_links(
    *, g00_path, g02_path, g04a_path, g04b_path, g05_path,
    expected_records=None,
):
    """Validate upstream identity, source hashes, and evidence tiers."""
    tasks = _index(
        g00_path,
        ("production_case_id", "case_instance_sha256",
         "graph_structure_sha256", "weighted_graph_sha256",
         "node_pair", "gold_cost"),
    )

    students = _index(
        g02_path,
        ("production_case_id", "canonical_wrong_reasoning_trace",
         "canonical_wrong_reasoning_trace_sha256"),
        "canonical_student_record",
        ("case_instance_sha256", "graph_structure_sha256",
         "weighted_graph_sha256", "node_pair", "gold_cost",
         "vsp_valid", "answer_extracted", "reasoning_present",
         "stopped_normally", "token_limit_truncation",
         "predicted_path"),
    )

    teacher_meta = ("node_pair", "gold_cost", "graph_structure_sha256")

    gold = _index(
        g04a_path, ("production_case_id", *TEACHER_FIELDS),
        "canonical_teacher_record", teacher_meta,
    )

    contrastive = _index(
        g04b_path,
        ("production_case_id", *TEACHER_FIELDS,
         "canonical_wrong_reasoning_trace",
         "canonical_wrong_reasoning_trace_sha256",
         "wrong_rt_continuity_verified"),
        "canonical_teacher_record", teacher_meta,
    )

    tiers = Counter()

    for row in iter_jsonl(
        g05_path,
        unique_key="production_case_id",
        required_fields=(
            "gold_target", "contrastive_target", "joint_valid",
            "canonical_wrong_reasoning_trace",
            "canonical_wrong_reasoning_trace_sha256",
        ),
    ):
        pid = row["production_case_id"]

        try:
            task, student, g, c = (
                source[pid]
                for source in (tasks, students, gold, contrastive)
            )
        except KeyError as exc:
            raise DatasetFormatError(
                f"{pid}: missing upstream record"
            ) from exc

        if row["joint_valid"] is not True:
            raise DatasetFormatError(f"{pid}: joint_valid is not true")

        trace_fields = (
            "canonical_wrong_reasoning_trace",
            "canonical_wrong_reasoning_trace_sha256",
        )

        _equal(student, row, trace_fields, f"{pid}:student")
        _equal(c, row, trace_fields, f"{pid}:contrastive")

        _check_student_failure(
            student["canonical_student_record"],
            f"{pid}:student",
        )

        trace = row["canonical_wrong_reasoning_trace"]
        if not isinstance(trace, str) or not trace.strip():
            raise DatasetFormatError(
                f"{pid}: empty Student reasoning trace"
            )

        _hash_text(
            row[trace_fields[0]],
            row[trace_fields[1]],
            f"{pid}:student",
        )

        _equal(
            task, student["canonical_student_record"],
            ("case_instance_sha256", "graph_structure_sha256",
             "weighted_graph_sha256", "node_pair", "gold_cost"),
            pid,
        )

        if (
            c["wrong_rt_continuity_verified"] is not True
            or not isinstance(row["contrastive_target"], dict)
            or row["contrastive_target"].get(
                "wrong_rt_continuity_verified"
            ) is not True
        ):
            raise DatasetFormatError(
                f"{pid}: contrastive continuity not true"
            )

        profiles = (
            _teacher(g, row["gold_target"], task, f"{pid}:gold"),
            _teacher(
                c, row["contrastive_target"], task,
                f"{pid}:contrastive",
            ),
        )

        if profiles == (
            "vsp_plus_deterministic_rt_factuality_audit",
            "vsp_plus_corrective_rt_factuality_validation",
        ):
            tiers["both_reasoning_factuality_validated"] += 1
        elif profiles == (
            "solver_verified_vsp",
            "solver_verified_vsp",
        ):
            tiers["both_vsp_only"] += 1
        else:
            raise DatasetFormatError(
                f"{pid}: unrecognized validation tier"
            )

    total = sum(tiers.values())

    if expected_records is not None and total != expected_records:
        raise DatasetFormatError(
            f"Expected {expected_records} cases, got {total}"
        )

    return {
        "case_count": total,
        "validation_tiers": dict(sorted(tiers.items())),
    }


# Explicit field allowlists for local release candidates.
TASK_FIELDS = (
    "production_case_id", "topology", "node_pair",
    "ground_truth_shortest_path", "gold_cost",
    "num_nodes", "num_edges", "topology_type",
    "topology_dimension", "weight_distribution",
    "gold_path_hops", "hop_bucket", "selected_exact_hop",
    "min_optimal_hops", "max_optimal_hops", "min_total_cost",
    "feasible_exact_hops", "eligible_pair_count_total",
    "eligible_pair_count_selected_exact_hop",
    "graph_seed", "weight_seed", "pair_seed",
    "generation_attempt_index", "attempts_to_accept",
    "replicate", "slot_id", "support_row",
    "pair_orientation_swapped",
    "graph_structure_sha256", "weighted_graph_sha256",
    "case_instance_sha256",
)

SAMPLING_FIELDS = ("temperature", "top_p", "max_tokens", "seed")

STUDENT_FIELDS = (
    "predicted_path", "predicted_cost", "final_content",
    "failure_category", "error_type", "syntactic_valid",
    "path_continuity", "no_repeated_nodes",
    "source_target_consistent", "vsp_valid",
    "answer_extracted", "reasoning_present",
    "stopped_normally", "token_limit_truncation",
    "model_label", "prompt_version", "prompt_sha256",
    "tokenizer_protocol", "generation_settings",
)

TEACHER_METADATA_FIELDS = (
    "model_label", "prompt_version", "prompt_sha256",
    "tokenizer_protocol", "generation_settings",
    "predicted_path", "predicted_cost",
    "stopped_normally", "token_limit_truncation",
)

G06_FIELDS = (
    "production_case_id", "condition", "prompt", "prompt_sha256",
    "completion", "completion_sha256",
)

G06_RELEASE_FILES = {
    "gold_train": "gold_train_200_prompt_completion_v1.jsonl",
    "contrastive_train": (
        "contrastive_train_200_prompt_completion_v1.jsonl"
    ),
    "gold_loss_val": "gold_loss_val_44_prompt_completion_v1.jsonl",
    "contrastive_loss_val": (
        "contrastive_loss_val_44_prompt_completion_v1.jsonl"
    ),
}

EVALUATION_MODEL_INPUT_FIELDS = (
    "topology", "node_pair",
)

EVALUATION_REFERENCE_FIELDS = (
    "ground_truth_shortest_path", "gold_cost",
)

EVALUATION_PROVENANCE_FIELDS = (
    "num_nodes", "num_edges", "topology_type",
    "topology_dimension", "weight_distribution",
    "gold_path_hops", "hop_bucket", "selected_exact_hop",
    "min_optimal_hops", "max_optimal_hops", "min_total_cost",
    "feasible_exact_hops", "eligible_pair_count_total",
    "eligible_pair_count_selected_exact_hop",
    "graph_seed", "weight_seed", "pair_seed",
    "generation_attempt_index", "attempts_to_accept",
    "replicate", "slot_id", "support_row",
    "pair_orientation_swapped",
    "graph_structure_sha256", "weighted_graph_sha256",
    "case_instance_sha256",
)

EVALUATION_RELEASE_FILES = {
    "g08_balanced": "g08_balanced_300.jsonl",
    "g08_train_dist": "g08_train_distribution_300.jsonl",
    "g09_balanced": "g09_balanced_846.jsonl",
    "g09_train_dist": "g09_train_distribution_1600.jsonl",
}

G08_SOURCE_VIEWS = {
    "balanced": "balanced_300",
    "train_distribution": "train_distribution_300",
}


def _public_metadata(metadata):
    """Select scientific provenance without service metadata."""
    result = dict(metadata)
    result["generation_settings"] = _select(
        metadata["generation_settings"],
        SAMPLING_FIELDS,
        "generation_settings",
    )
    return result


def export_public_pairs(
    *, g00_path, g02_path, g04a_path, g04b_path,
    g05_path, output_dir, expected_records=None,
):
    """Create a private, validated candidate dataset.

    Does not grant or establish redistribution permission.
    The destination must not already exist.
    """
    import json
    import os
    import shutil
    import tempfile
    from pathlib import Path

    output = Path(output_dir)

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")

    if not output.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory not found: {output.parent}"
        )

    input_paths = {
        "g00": Path(g00_path),
        "g02": Path(g02_path),
        "g04a": Path(g04a_path),
        "g04b": Path(g04b_path),
        "g05": Path(g05_path),
    }

    input_sha256 = {
        name: _sha256_file(source)
        for name, source in input_paths.items()
    }

    # Validate every paired record before creating output files.
    report = validate_joint_links(
        g00_path=g00_path,
        g02_path=g02_path,
        g04a_path=g04a_path,
        g04b_path=g04b_path,
        g05_path=g05_path,
        expected_records=expected_records,
    )

    tasks = _index(g00_path, TASK_FIELDS)

    students = _index(
        g02_path,
        ("production_case_id", "selection_reason",
         "canonical_wrong_reasoning_trace",
         "canonical_wrong_reasoning_trace_sha256"),
        "canonical_student_record", STUDENT_FIELDS,
    )

    teachers = {}
    for condition, path in (
        ("gold", g04a_path),
        ("contrastive", g04b_path),
    ):
        teachers[condition] = _index(
            path,
            ("production_case_id", *TEACHER_FIELDS),
            "canonical_teacher_record",
            TEACHER_METADATA_FIELDS,
        )

    staging = Path(tempfile.mkdtemp(
        prefix=".pathfact-candidate-",
        dir=output.parent,
    ))

    try:
        os.chmod(staging, 0o700)
        count = 0

        with (staging / "paired.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in iter_jsonl(
                g05_path, unique_key="production_case_id"
            ):
                pid = row["production_case_id"]
                student_source = students[pid]

                student = _public_metadata(
                    student_source["canonical_student_record"]
                )
                student.update({
                    "reasoning_trace":
                        row["canonical_wrong_reasoning_trace"],
                    "reasoning_trace_sha256":
                        row["canonical_wrong_reasoning_trace_sha256"],
                    "selection_reason":
                        student_source["selection_reason"],
                })

                targets = {}
                for condition in ("gold", "contrastive"):
                    source = teachers[condition][pid]
                    metadata = _public_metadata(
                        source["canonical_teacher_record"]
                    )
                    target = row[f"{condition}_target"]

                    targets[condition] = {
                        **_select(
                            target, TEACHER_FIELDS,
                            f"{pid}:{condition}"
                        ),
                        **metadata,
                    }

                targets["contrastive"][
                    "wrong_rt_continuity_verified"
                ] = row["contrastive_target"][
                    "wrong_rt_continuity_verified"
                ]

                profiles = (
                    targets["gold"]["validation_profile"],
                    targets["contrastive"]["validation_profile"],
                )

                tier = (
                    "both_reasoning_factuality_validated"
                    if profiles == (
                        "vsp_plus_deterministic_rt_factuality_audit",
                        "vsp_plus_corrective_rt_factuality_validation",
                    )
                    else "both_vsp_only"
                )

                record = {
                    "schema_version": "pathfact-paired-v0.1",
                    "production_case_id": pid,
                    "task": tasks[pid],
                    "student": student,
                    "teacher_targets": targets,
                    "joint_valid": row["joint_valid"],
                    "validation_tier": tier,
                }

                handle.write(
                    json.dumps(
                        record, sort_keys=True, ensure_ascii=False
                    ) + "\n"
                )
                count += 1

        if count != report["case_count"]:
            raise DatasetFormatError(
                "Record count changed during export"
            )

        changed = [
            name
            for name, source in input_paths.items()
            if _sha256_file(source) != input_sha256[name]
        ]

        if changed:
            raise DatasetFormatError(
                "Source files changed during export: "
                + ", ".join(changed)
            )

        summary = {
            "schema_version": "pathfact-paired-v0.1",
            "release_status": "local_candidate_not_approved",
            "input_sha256": input_sha256,
            "paired_sha256": _sha256_file(
                staging / "paired.jsonl"
            ),
            **report,
        }

        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for path in staging.iterdir():
            os.chmod(path, 0o600)

        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"Output appeared during export: {output}"
            )

        os.rename(staging, output)

    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "paired": output / "paired.jsonl",
        "summary": output / "summary.json",
    }


def _write_public_tasks(source_path, destination, expected_records):
    """Write the established G00 allowlist in source-record order."""
    import json

    source_sha256 = _sha256_file(source_path)
    count = 0

    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in iter_jsonl(
            source_path, unique_key="production_case_id"
        ):
            selected = _select(row, TASK_FIELDS, str(source_path))
            handle.write(
                json.dumps(
                    selected, sort_keys=True, ensure_ascii=False
                )
                + "\n"
            )
            count += 1

    if count != expected_records:
        raise DatasetFormatError(
            f"Expected {expected_records} G00 cases, got {count}"
        )

    if _sha256_file(source_path) != source_sha256:
        raise DatasetFormatError("G00 source changed during export")

    return {
        "record_count": count,
        "source_sha256": source_sha256,
        "sha256": _sha256_file(destination),
    }


def _validate_g06_file(path, condition, expected_records):
    """Validate the frozen six-field G06 prompt/completion contract."""
    identifiers = set()
    source_sha256 = _sha256_file(path)

    for row in iter_jsonl(
        path,
        required_fields=G06_FIELDS,
        unique_key="production_case_id",
    ):
        unexpected = sorted(set(row) - set(G06_FIELDS))

        if unexpected:
            raise DatasetFormatError(
                f"{path}: unexpected G06 fields {unexpected}"
            )

        pid = row["production_case_id"]
        identifiers.add(pid)

        if row["condition"] != condition:
            raise DatasetFormatError(
                f"{path}:{pid}: expected condition {condition}"
            )

        for field in ("prompt", "completion"):
            _hash_text(
                row[field],
                row[f"{field}_sha256"],
                f"{path}:{pid}:{field}",
            )

    if len(identifiers) != expected_records:
        raise DatasetFormatError(
            f"Expected {expected_records} records in {path}, "
            f"got {len(identifiers)}"
        )

    if _sha256_file(path) != source_sha256:
        raise DatasetFormatError(
            f"G06 source changed during validation: {path}"
        )

    return identifiers, source_sha256


def validate_g06_release_files(
    *,
    gold_train_path,
    contrastive_train_path,
    gold_loss_val_path,
    contrastive_loss_val_path,
    expected_train_records=200,
    expected_loss_val_records=44,
    expected_case_ids=None,
):
    """Validate the four frozen G06 JSONL files without rewriting them."""
    paths = {
        "gold_train": gold_train_path,
        "contrastive_train": contrastive_train_path,
        "gold_loss_val": gold_loss_val_path,
        "contrastive_loss_val": contrastive_loss_val_path,
    }
    conditions = {
        "gold_train": "gold",
        "contrastive_train": "contrastive",
        "gold_loss_val": "gold",
        "contrastive_loss_val": "contrastive",
    }
    expected = {
        "gold_train": expected_train_records,
        "contrastive_train": expected_train_records,
        "gold_loss_val": expected_loss_val_records,
        "contrastive_loss_val": expected_loss_val_records,
    }

    identifiers = {}
    source_sha256 = {}

    for name, path in paths.items():
        identifiers[name], source_sha256[name] = _validate_g06_file(
            path, conditions[name], expected[name]
        )

    if identifiers["gold_train"] != identifiers["contrastive_train"]:
        raise DatasetFormatError("G06 Train Gold/Contrastive IDs differ")

    if identifiers["gold_loss_val"] != identifiers["contrastive_loss_val"]:
        raise DatasetFormatError("G06 LossVal Gold/Contrastive IDs differ")

    train_ids = identifiers["gold_train"]
    loss_val_ids = identifiers["gold_loss_val"]

    if train_ids & loss_val_ids:
        raise DatasetFormatError("G06 Train and LossVal IDs overlap")

    all_ids = train_ids | loss_val_ids

    if expected_case_ids is not None and all_ids != set(expected_case_ids):
        raise DatasetFormatError("G06 case IDs do not match G05")

    return {
        "train_case_count": len(train_ids),
        "loss_val_case_count": len(loss_val_ids),
        "unique_case_count": len(all_ids),
        "source_sha256": source_sha256,
    }


def _public_evaluation_record(
    task, *, source_stage, view_name, source_difficulty
):
    pid = task["production_case_id"]
    view = {
        "source_stage": source_stage,
        "name": view_name,
    }

    if source_difficulty is not None:
        view["source_difficulty"] = source_difficulty

    return {
        "schema_version": "pathfact-evaluation-v0.1",
        "production_case_id": pid,
        "view": view,
        "model_input": _select(
            task,
            EVALUATION_MODEL_INPUT_FIELDS,
            f"{source_stage}:{view_name}:{pid}:model_input",
        ),
        "reference": _select(
            task,
            EVALUATION_REFERENCE_FIELDS,
            f"{source_stage}:{view_name}:{pid}:reference",
        ),
        "provenance": _select(
            task,
            EVALUATION_PROVENANCE_FIELDS,
            f"{source_stage}:{view_name}:{pid}:provenance",
        ),
    }


def adapt_g08_evaluation_view(
    source_path, *, view_name, expected_records
):
    """Adapt a nested G08 view through explicit public allowlists."""
    expected_source_view = G08_SOURCE_VIEWS.get(view_name)

    if expected_source_view is None:
        raise DatasetFormatError(f"Unsupported G08 view: {view_name}")

    source_sha256 = _sha256_file(source_path)
    records = []

    for row in iter_jsonl(
        source_path, unique_key="production_case_id"
    ):
        selected = _select(
            row,
            (
                "production_case_id", "task_record", "g08_dataset",
                "difficulty", "gold_path_hops",
            ),
            str(source_path),
        )
        pid = selected["production_case_id"]
        task = _select(
            selected["task_record"],
            TASK_FIELDS,
            f"{source_path}:{pid}:task_record",
        )

        if task["production_case_id"] != pid:
            raise DatasetFormatError(
                f"{source_path}:{pid}: nested case ID mismatch"
            )

        if selected["g08_dataset"] != expected_source_view:
            raise DatasetFormatError(
                f"{source_path}:{pid}: unexpected G08 source view"
            )

        if selected["gold_path_hops"] != task["gold_path_hops"]:
            raise DatasetFormatError(
                f"{source_path}:{pid}: G08 hop metadata mismatch"
            )

        difficulty = selected["difficulty"]

        if not isinstance(difficulty, str) or not difficulty:
            raise DatasetFormatError(
                f"{source_path}:{pid}: invalid G08 difficulty"
            )

        records.append(_public_evaluation_record(
            task,
            source_stage="G08",
            view_name=view_name,
            source_difficulty=difficulty,
        ))

    if len(records) != expected_records:
        raise DatasetFormatError(
            f"Expected {expected_records} G08 {view_name} records, "
            f"got {len(records)}"
        )

    if _sha256_file(source_path) != source_sha256:
        raise DatasetFormatError(
            f"G08 source changed during adaptation: {source_path}"
        )

    return {
        "records": records,
        "record_count": len(records),
        "source_sha256": source_sha256,
    }


def adapt_g09_evaluation_view(
    source_path, *, view_name, expected_records
):
    """Adapt a top-level G09 view through explicit public allowlists."""
    if view_name not in ("balanced", "train_distribution"):
        raise DatasetFormatError(f"Unsupported G09 view: {view_name}")

    source_sha256 = _sha256_file(source_path)
    records = []

    for row in iter_jsonl(
        source_path, unique_key="production_case_id"
    ):
        task = _select(row, TASK_FIELDS, str(source_path))
        records.append(_public_evaluation_record(
            task,
            source_stage="G09",
            view_name=view_name,
            source_difficulty=None,
        ))

    if len(records) != expected_records:
        raise DatasetFormatError(
            f"Expected {expected_records} G09 {view_name} records, "
            f"got {len(records)}"
        )

    if _sha256_file(source_path) != source_sha256:
        raise DatasetFormatError(
            f"G09 source changed during adaptation: {source_path}"
        )

    return {
        "records": records,
        "record_count": len(records),
        "source_sha256": source_sha256,
    }


def _evaluation_task_payload(record):
    return {
        "model_input": record["model_input"],
        "reference": record["reference"],
        "provenance": record["provenance"],
    }


def validate_evaluation_view_relationships(
    views,
    *,
    g06_case_ids,
    expected_g09_overlap=617,
    expected_g08_balanced_g09_balanced_overlap=164,
    expected_g08_balanced_g09_train_dist_overlap=221,
    expected_g08_train_dist_g09_balanced_overlap=110,
    expected_g08_train_dist_g09_train_dist_overlap=265,
):
    """Validate frozen G08/G09 identity, overlap and reference rules."""
    expected_names = set(EVALUATION_RELEASE_FILES)

    if set(views) != expected_names:
        raise DatasetFormatError("Expected all four evaluation views")

    indexed = {}

    for name, adapted in views.items():
        records = adapted.get("records")

        if not isinstance(records, list):
            raise DatasetFormatError(f"Malformed adapted view: {name}")

        indexed[name] = {
            row["production_case_id"]: row
            for row in records
        }

        if len(indexed[name]) != len(records):
            raise DatasetFormatError(
                f"Duplicate adapted case ID in {name}"
            )

    names = tuple(EVALUATION_RELEASE_FILES)

    for index, left_name in enumerate(names):
        left = indexed[left_name]

        for right_name in names[index + 1:]:
            right = indexed[right_name]

            for pid in left.keys() & right.keys():
                if (
                    _evaluation_task_payload(left[pid])
                    != _evaluation_task_payload(right[pid])
                ):
                    raise DatasetFormatError(
                        f"Inconsistent evaluation task/reference: {pid}"
                    )

    identifiers = {
        name: set(records)
        for name, records in indexed.items()
    }

    if identifiers["g08_balanced"] & identifiers["g08_train_dist"]:
        raise DatasetFormatError("G08 evaluation views must be disjoint")

    overlaps = {
        "g09_balanced_train_dist": len(
            identifiers["g09_balanced"]
            & identifiers["g09_train_dist"]
        ),
        "g08_balanced_g09_balanced": len(
            identifiers["g08_balanced"]
            & identifiers["g09_balanced"]
        ),
        "g08_balanced_g09_train_dist": len(
            identifiers["g08_balanced"]
            & identifiers["g09_train_dist"]
        ),
        "g08_train_dist_g09_balanced": len(
            identifiers["g08_train_dist"]
            & identifiers["g09_balanced"]
        ),
        "g08_train_dist_g09_train_dist": len(
            identifiers["g08_train_dist"]
            & identifiers["g09_train_dist"]
        ),
    }
    expected_overlaps = {
        "g09_balanced_train_dist": expected_g09_overlap,
        "g08_balanced_g09_balanced": (
            expected_g08_balanced_g09_balanced_overlap
        ),
        "g08_balanced_g09_train_dist": (
            expected_g08_balanced_g09_train_dist_overlap
        ),
        "g08_train_dist_g09_balanced": (
            expected_g08_train_dist_g09_balanced_overlap
        ),
        "g08_train_dist_g09_train_dist": (
            expected_g08_train_dist_g09_train_dist_overlap
        ),
    }

    if overlaps != expected_overlaps:
        raise DatasetFormatError(
            "Unexpected evaluation view overlaps: "
            f"expected {expected_overlaps}, got {overlaps}"
        )

    evaluation_ids = set().union(*identifiers.values())
    g06_overlap = evaluation_ids & set(g06_case_ids)

    if g06_overlap:
        raise DatasetFormatError(
            "Evaluation views overlap G06 Train/LossVal"
        )

    return {
        "view_counts": {
            name: len(values)
            for name, values in identifiers.items()
        },
        "overlaps": overlaps,
        "g08_views_disjoint": True,
        "g06_overlap_count": 0,
        "source_sha256": {
            name: views[name]["source_sha256"]
            for name in names
        },
    }


def _write_evaluation_records(destination, records):
    import json

    with destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record, sort_keys=True, ensure_ascii=False
                )
                + "\n"
            )

    return _sha256_file(destination)


def export_core_release_candidate(
    *,
    g00_path,
    g02_path,
    g04a_path,
    g04b_path,
    g05_path,
    g06_gold_train_path,
    g06_contrastive_train_path,
    g06_gold_loss_val_path,
    g06_contrastive_loss_val_path,
    g08_balanced_path,
    g08_train_dist_path,
    g09_balanced_path,
    g09_train_dist_path,
    output_dir,
    dataset_version,
    expected_g00_records=2200,
    expected_paired_records=244,
    expected_train_records=200,
    expected_loss_val_records=44,
    expected_reasoning_validated_pairs=141,
    expected_vsp_only_pairs=103,
    expected_g08_balanced_records=300,
    expected_g08_train_dist_records=300,
    expected_g09_balanced_records=846,
    expected_g09_train_dist_records=1600,
    expected_g09_overlap=617,
    expected_g08_balanced_g09_balanced_overlap=164,
    expected_g08_balanced_g09_train_dist_overlap=221,
    expected_g08_train_dist_g09_balanced_overlap=110,
    expected_g08_train_dist_g09_train_dist_overlap=265,
):
    """Build a local-only G00/G05/G06/G08/G09 release candidate.

    G00 and G05 are exported through explicit public field allowlists.
    The four validated G06 files are copied byte-for-byte. Frozen inputs
    are never modified. G08/G09 are normalized through explicit
    evaluation allowlists. The destination must not already exist.
    """
    import json
    import os
    import shutil
    import tempfile
    from pathlib import Path

    from pathfact.datasets.manifest import (
        create_release_manifest,
        verify_release_manifest,
    )

    output = Path(output_dir)

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")

    if not output.parent.is_dir():
        raise FileNotFoundError(
            f"Output parent directory not found: {output.parent}"
        )

    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise DatasetFormatError("Dataset version is required")

    input_paths = {
        "g00": Path(g00_path),
        "g02": Path(g02_path),
        "g04a": Path(g04a_path),
        "g04b": Path(g04b_path),
        "g05": Path(g05_path),
        "g06_gold_train": Path(g06_gold_train_path),
        "g06_contrastive_train": Path(g06_contrastive_train_path),
        "g06_gold_loss_val": Path(g06_gold_loss_val_path),
        "g06_contrastive_loss_val": Path(
            g06_contrastive_loss_val_path
        ),
        "g08_balanced": Path(g08_balanced_path),
        "g08_train_dist": Path(g08_train_dist_path),
        "g09_balanced": Path(g09_balanced_path),
        "g09_train_dist": Path(g09_train_dist_path),
    }
    input_sha256 = {
        name: _sha256_file(path)
        for name, path in input_paths.items()
    }

    pair_report = validate_joint_links(
        g00_path=g00_path,
        g02_path=g02_path,
        g04a_path=g04a_path,
        g04b_path=g04b_path,
        g05_path=g05_path,
        expected_records=expected_paired_records,
    )
    expected_tiers = {
        "both_reasoning_factuality_validated": (
            expected_reasoning_validated_pairs
        ),
        "both_vsp_only": expected_vsp_only_pairs,
    }
    observed_tiers = {
        name: pair_report["validation_tiers"].get(name, 0)
        for name in expected_tiers
    }

    if observed_tiers != expected_tiers:
        raise DatasetFormatError(
            "Unexpected G05 validation tier counts: "
            f"expected {expected_tiers}, "
            f"got {observed_tiers}"
        )

    g05_ids = {
        row["production_case_id"]
        for row in iter_jsonl(
            g05_path, unique_key="production_case_id"
        )
    }
    g06_report = validate_g06_release_files(
        gold_train_path=g06_gold_train_path,
        contrastive_train_path=g06_contrastive_train_path,
        gold_loss_val_path=g06_gold_loss_val_path,
        contrastive_loss_val_path=g06_contrastive_loss_val_path,
        expected_train_records=expected_train_records,
        expected_loss_val_records=expected_loss_val_records,
        expected_case_ids=g05_ids,
    )
    g06_case_ids = set()

    for path in (g06_gold_train_path, g06_gold_loss_val_path):
        g06_case_ids.update(
            row["production_case_id"]
            for row in iter_jsonl(
                path, unique_key="production_case_id"
            )
        )

    evaluation_views = {
        "g08_balanced": adapt_g08_evaluation_view(
            g08_balanced_path,
            view_name="balanced",
            expected_records=expected_g08_balanced_records,
        ),
        "g08_train_dist": adapt_g08_evaluation_view(
            g08_train_dist_path,
            view_name="train_distribution",
            expected_records=expected_g08_train_dist_records,
        ),
        "g09_balanced": adapt_g09_evaluation_view(
            g09_balanced_path,
            view_name="balanced",
            expected_records=expected_g09_balanced_records,
        ),
        "g09_train_dist": adapt_g09_evaluation_view(
            g09_train_dist_path,
            view_name="train_distribution",
            expected_records=expected_g09_train_dist_records,
        ),
    }
    evaluation_report = validate_evaluation_view_relationships(
        evaluation_views,
        g06_case_ids=g06_case_ids,
        expected_g09_overlap=expected_g09_overlap,
        expected_g08_balanced_g09_balanced_overlap=(
            expected_g08_balanced_g09_balanced_overlap
        ),
        expected_g08_balanced_g09_train_dist_overlap=(
            expected_g08_balanced_g09_train_dist_overlap
        ),
        expected_g08_train_dist_g09_balanced_overlap=(
            expected_g08_train_dist_g09_balanced_overlap
        ),
        expected_g08_train_dist_g09_train_dist_overlap=(
            expected_g08_train_dist_g09_train_dist_overlap
        ),
    )

    staging = Path(tempfile.mkdtemp(
        prefix=".pathfact-core-candidate-",
        dir=output.parent,
    ))

    try:
        os.chmod(staging, 0o700)
        tasks_dir = staging / "tasks"
        paired_dir = staging / "paired"
        training_dir = staging / "training"
        evaluation_dir = staging / "evaluation"
        metadata_dir = staging / "metadata"

        for directory in (
            tasks_dir, paired_dir, training_dir,
            evaluation_dir, metadata_dir,
        ):
            directory.mkdir()

        task_path = tasks_dir / "g00_tasks.jsonl"
        task_report = _write_public_tasks(
            g00_path, task_path, expected_g00_records
        )

        pair_candidate = staging / ".paired-candidate"
        pair_outputs = export_public_pairs(
            g00_path=g00_path,
            g02_path=g02_path,
            g04a_path=g04a_path,
            g04b_path=g04b_path,
            g05_path=g05_path,
            output_dir=pair_candidate,
            expected_records=expected_paired_records,
        )
        pair_summary = json.loads(
            pair_outputs["summary"].read_text(encoding="utf-8")
        )
        paired_path = paired_dir / "g05_paired.jsonl"
        os.rename(pair_outputs["paired"], paired_path)
        pair_outputs["summary"].unlink()
        pair_candidate.rmdir()

        g06_sources = {
            "gold_train": Path(g06_gold_train_path),
            "contrastive_train": Path(g06_contrastive_train_path),
            "gold_loss_val": Path(g06_gold_loss_val_path),
            "contrastive_loss_val": Path(
                g06_contrastive_loss_val_path
            ),
        }
        training_paths = {}

        for name, source in g06_sources.items():
            destination = training_dir / G06_RELEASE_FILES[name]
            shutil.copyfile(source, destination)

            if _sha256_file(destination) != input_sha256[f"g06_{name}"]:
                raise DatasetFormatError(
                    f"G06 byte preservation failed: {name}"
                )

            training_paths[name] = destination

        evaluation_paths = {}
        exported_evaluation_sha256 = {}

        for name, adapted in evaluation_views.items():
            destination = (
                evaluation_dir / EVALUATION_RELEASE_FILES[name]
            )
            exported_evaluation_sha256[name] = (
                _write_evaluation_records(
                    destination, adapted["records"]
                )
            )
            evaluation_paths[name] = destination

        artifacts = [
            {
                "path": "tasks/g00_tasks.jsonl",
                "source_stage": "G00",
                "schema_version": "pathfact-g00-public-v0.1",
                "source_sha256": input_sha256["g00"],
            },
            {
                "path": "paired/g05_paired.jsonl",
                "source_stage": "G05",
                "schema_version": "pathfact-paired-v0.1",
                "source_sha256": {
                    name: input_sha256[name]
                    for name in ("g00", "g02", "g04a", "g04b", "g05")
                },
            },
        ]

        for name, destination in training_paths.items():
            artifacts.append({
                "path": destination.relative_to(staging).as_posix(),
                "source_stage": "G06",
                "schema_version": "pathfact-g06-prompt-completion-v1",
                "source_sha256": input_sha256[f"g06_{name}"],
            })

        for name, destination in evaluation_paths.items():
            artifacts.append({
                "path": destination.relative_to(staging).as_posix(),
                "source_stage": "G08" if name.startswith("g08") else "G09",
                "schema_version": "pathfact-evaluation-v0.1",
                "source_sha256": input_sha256[name],
            })

        manifest_path = create_release_manifest(
            staging, artifacts, dataset_version
        )
        manifest_report = verify_release_manifest(staging)

        summary = {
            "schema_version": "pathfact-core-candidate-v0.2",
            "dataset_version": dataset_version,
            "release_status": "local_candidate_not_approved",
            "license": None,
            "g00": task_report,
            "g05": pair_report,
            "g06": g06_report,
            "evaluation": {
                **evaluation_report,
                "export_sha256": exported_evaluation_sha256,
            },
            "input_sha256": input_sha256,
            "paired_sha256": pair_summary["paired_sha256"],
            "manifest_sha256": _sha256_file(manifest_path),
            "manifest_verification": manifest_report,
        }
        summary_path = metadata_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        changed = [
            name
            for name, source in input_paths.items()
            if _sha256_file(source) != input_sha256[name]
        ]

        if changed:
            raise DatasetFormatError(
                "Source files changed during core export: "
                + ", ".join(changed)
            )

        for path in staging.rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)

        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"Output appeared during export: {output}"
            )

        os.rename(staging, output)

    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "root": output,
        "manifest": output / "manifest.json",
        "summary": output / "metadata" / "summary.json",
        "tasks": output / "tasks" / "g00_tasks.jsonl",
        "paired": output / "paired" / "g05_paired.jsonl",
        **{
            name: output / "training" / filename
            for name, filename in G06_RELEASE_FILES.items()
        },
        **{
            name: output / "evaluation" / filename
            for name, filename in EVALUATION_RELEASE_FILES.items()
        },
    }
