"""Read-only JSONL dataset utilities for PathFACT."""

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


class DatasetFormatError(ValueError):
    """Invalid JSONL record structure or identifier."""


def iter_jsonl(
    path: str | Path,
    *,
    required_fields: Iterable[str] = (),
    unique_key: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream JSONL records without modifying their contents.

    Empty lines, malformed JSON and non-object records are rejected.
    Required fields and unique string identifiers are optional checks.
    """

    source = Path(path)
    required = set(required_fields)

    if unique_key is not None:
        required.add(unique_key)

    seen = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{source}:{line_number}"

            if not line.strip():
                raise DatasetFormatError(
                    f"{location}: empty JSONL line"
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(
                    f"{location}: invalid JSON: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise DatasetFormatError(
                    f"{location}: expected a JSON object"
                )

            missing = sorted(required - record.keys())

            if missing:
                raise DatasetFormatError(
                    f"{location}: missing fields: {missing}"
                )

            if unique_key is not None:
                identifier = record[unique_key]

                if not isinstance(identifier, str) or not identifier:
                    raise DatasetFormatError(
                        f"{location}: invalid {unique_key}"
                    )

                if identifier in seen:
                    raise DatasetFormatError(
                        f"{location}: duplicate {unique_key}: "
                        f"{identifier}"
                    )

                seen.add(identifier)

            yield record
