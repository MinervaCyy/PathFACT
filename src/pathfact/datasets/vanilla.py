"""CPU-only materialisation of answer-only Vanilla-SFT datasets.

Uses validated G06 Gold records as input.
Preserves case order and task-only prompts.
Does not perform independent solver verification.
"""

import ast
import re
from pathlib import Path

from .g06 import sft_record, sha256_text, write_jsonl
from .io import iter_jsonl


ANSWER_PATTERN = re.compile(
    r"Answer:\s*(\[[^\r\n]*\])\s*\Z"
)


def _convert_gold_records(path, expected_count):
    """Convert frozen G06 Gold records to pure answer-only targets."""

    records = []

    for source in iter_jsonl(
        path,
        required_fields=(
            "production_case_id",
            "condition",
            "prompt",
            "prompt_sha256",
            "completion",
            "completion_sha256",
        ),
        unique_key="production_case_id",
    ):
        pid = source["production_case_id"]

        if source["condition"] != "gold":
            raise ValueError(f"Expected Gold condition: {pid}")

        if (
            sha256_text(source["prompt"])
            != source["prompt_sha256"]
            or sha256_text(source["completion"])
            != source["completion_sha256"]
        ):
            raise ValueError(f"Source checksum mismatch: {pid}")

        match = ANSWER_PATTERN.search(source["completion"])

        if match is None:
            raise ValueError(f"Missing final answer: {pid}")

        try:
            path_nodes = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid answer path: {pid}") from exc

        if (
            not isinstance(path_nodes, list)
            or not path_nodes
            or any(
                not isinstance(node, int)
                or isinstance(node, bool)
                for node in path_nodes
            )
        ):
            raise ValueError(f"Invalid path nodes: {pid}")

        completion = "Answer: " + str(path_nodes)

        records.append(
            sft_record(
                pid,
                "vanilla",
                source["prompt"],
                completion,
            )
        )

    if len(records) != expected_count:
        raise ValueError(
            f"Expected {expected_count} records, got {len(records)}"
        )

    return records


def materialize_vanilla_sft(
    *,
    gold_train_path,
    gold_loss_val_path,
    output_dir,
):
    """Materialise frozen-aligned Train-200 and LossVal-44.

    Input files remain unchanged.
    The output directory must not already exist.
    """

    train = _convert_gold_records(gold_train_path, 200)
    loss_val = _convert_gold_records(gold_loss_val_path, 44)

    train_ids = {
        record["production_case_id"] for record in train
    }
    val_ids = {
        record["production_case_id"] for record in loss_val
    }

    if train_ids & val_ids:
        raise ValueError("Training and validation cases overlap")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    filenames = {
        "train": "answer_only_train_200_prompt_completion_v1.jsonl",
        "loss_val": "answer_only_loss_val_44_prompt_completion_v1.jsonl",
    }

    paths = {}

    for split, records in (
        ("train", train),
        ("loss_val", loss_val),
    ):
        path = output / filenames[split]
        write_jsonl(path, records)
        paths[split] = path

    return paths


def validate_vanilla_release_files(
    *,
    vanilla_train_path,
    vanilla_loss_val_path,
    gold_train_path,
    gold_loss_val_path,
    expected_train_records=200,
    expected_loss_val_records=44,
):
    """Validate Vanilla files against frozen G06 Gold supervision.

    Checks exact schema, record order, prompts, answer-only
    completions and their internal SHA256 hashes.

    This is not an independent graph-solver verification.
    """

    splits = (
        (
            "train",
            vanilla_train_path,
            gold_train_path,
            expected_train_records,
        ),
        (
            "loss_val",
            vanilla_loss_val_path,
            gold_loss_val_path,
            expected_loss_val_records,
        ),
    )

    report = {}
    all_ids = set()

    for name, vanilla_path, gold_path, expected_count in splits:
        expected = _convert_gold_records(
            gold_path, expected_count
        )

        actual = list(
            iter_jsonl(
                vanilla_path,
                required_fields=(
                    "production_case_id",
                    "condition",
                    "prompt",
                    "prompt_sha256",
                    "completion",
                    "completion_sha256",
                ),
                unique_key="production_case_id",
            )
        )

        if len(actual) != expected_count:
            raise ValueError(
                f"Vanilla {name}: expected {expected_count} "
                f"records, got {len(actual)}"
            )

        for index, (observed, reference) in enumerate(
            zip(actual, expected), start=1
        ):
            if set(observed) != set(reference):
                raise ValueError(
                    f"Vanilla {name}: unexpected fields "
                    f"at row {index}"
                )

            for field, expected_value in reference.items():
                if observed[field] != expected_value:
                    raise ValueError(
                        f"Vanilla {name}: mismatch in "
                        f"{field} at row {index}"
                    )

        ids = {
            record["production_case_id"]
            for record in actual
        }

        if ids & all_ids:
            raise ValueError(
                "Vanilla Train and LossVal cases overlap"
            )

        all_ids.update(ids)
        report[name] = len(actual)

    report["total"] = len(all_ids)
    report["status"] = "PASS"

    return report
