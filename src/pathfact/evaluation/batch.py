"""CPU-only batch re-scoring of saved shortest-path responses.

Historical VSP scoring is preserved. Completion status is reported
separately and does not gate VSP validity.

This module scores already-extracted candidate paths. Raw-response
answer extraction is a separate operation.
"""

from collections import Counter
from pathlib import Path

from pathfact.datasets import iter_jsonl

from .historical_vsp import verify_path


def score_saved_record(record):
    """Recompute historical VSP fields for one saved response."""

    if not isinstance(record, dict):
        raise ValueError("Expected a response record object")

    case_id = record.get("case_id")

    if case_id is None:
        case_id = record.get("production_case_id")

    if case_id is None or case_id == "":
        raise ValueError("Missing case identifier")

    if "predicted_path" not in record:
        raise ValueError(
            f"{case_id}: missing predicted_path"
        )

    predicted_path = record["predicted_path"]

    if (
        not isinstance(predicted_path, list)
        or any(not isinstance(node, int) for node in predicted_path)
    ):
        raise ValueError(
            f"{case_id}: predicted_path must be a list of integers"
        )

    verification = verify_path(record, predicted_path)

    return {
        "case_id": case_id,
        "request_id": record.get("request_id"),
        "status": record.get("status"),
        "finish_reason": record.get("finish_reason"),
        "stopped_normally": record.get("stopped_normally"),
        "token_limit_truncation": record.get(
            "token_limit_truncation"
        ),
        "answer_extracted": record.get("answer_extracted"),
        "predicted_path": predicted_path,
        **verification,
    }


def iter_scored_jsonl(path: str | Path):
    """Stream historical VSP results from saved JSONL responses."""

    for record in iter_jsonl(path):
        yield score_saved_record(record)


def summarize_scored_records(
    records,
    *,
    expected_records=None,
):
    """Summarize recomputed scores over all supplied records.

    Truncated and failed requests remain in the denominator.
    expected_records, when supplied, checks dataset completeness.
    """

    total = 0
    valid = 0
    extracted = 0
    stopped = 0
    truncated = 0

    case_ids = set()
    request_ids = set()

    statuses = Counter()
    finish_reasons = Counter()
    failure_categories = Counter()

    for record in records:
        total += 1

        case_ids.add(str(record["case_id"]))

        request_id = record.get("request_id")
        if request_id is not None:
            request_ids.add(str(request_id))

        statuses[str(record.get("status"))] += 1
        finish_reasons[str(record.get("finish_reason"))] += 1

        failure_categories[
            record["failure_category"]
        ] += 1

        if record["vsp_valid"] is True:
            valid += 1

        if record.get("answer_extracted") is True:
            extracted += 1

        if record.get("stopped_normally") is True:
            stopped += 1

        if record.get("token_limit_truncation") is True:
            truncated += 1

    if expected_records is not None and total != expected_records:
        raise ValueError(
            f"Expected {expected_records} records, found {total}"
        )

    return {
        "records_present": total,
        "unique_case_ids": len(case_ids),
        "unique_request_ids": len(request_ids),
        "vsp_valid": valid,
        "vsp_rate": valid / total if total else None,
        "answer_extracted": extracted,
        "stopped_normally": stopped,
        "token_limit_truncation": truncated,
        "status_counts": dict(sorted(statuses.items())),
        "finish_reason_counts": dict(
            sorted(finish_reasons.items())
        ),
        "failure_categories": dict(
            sorted(failure_categories.items())
        ),
    }


def summarize_saved_jsonl(path, *, expected_records=None):
    """Re-score and summarize a saved evaluation file."""

    return summarize_scored_records(
        iter_scored_jsonl(path),
        expected_records=expected_records,
    )
