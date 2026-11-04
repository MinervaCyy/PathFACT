"""CPU-only evaluation of saved, already-extracted candidate paths.

No model inference is performed. Historical VSP scoring is used,
and inference completion status does not gate path validity.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .batch import (
    iter_scored_jsonl,
    summarize_scored_records,
)


def evaluate_saved_jsonl(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    expected_records: int | None = None,
) -> dict[str, Any]:
    """Re-score saved responses and write portable evaluation outputs.

    Output files:
      per_case.jsonl: independently recomputed historical VSP results
      summary.json: aggregate scores and completion-status counts

    Existing output directories are never overwritten. Incomplete
    results are discarded when evaluation raises an exception.
    """

    if expected_records is not None:
        if (
            isinstance(expected_records, bool)
            or not isinstance(expected_records, int)
            or expected_records < 1
        ):
            raise ValueError(
                "expected_records must be a positive integer"
            )

    source = Path(input_path)
    target = Path(output_dir)

    if not source.is_file():
        raise FileNotFoundError(
            f"Input JSONL file not found: {source}"
        )

    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"Output path already exists: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
    )

    try:
        scored_path = staging / "per_case.jsonl"

        with scored_path.open("w", encoding="utf-8") as output:

            def scored_records():
                for result in iter_scored_jsonl(source):
                    output.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    yield result

            summary = summarize_scored_records(
                scored_records(),
                expected_records=expected_records,
            )

        if summary["records_present"] == 0:
            raise ValueError(
                "Input JSONL contains no evaluation records"
            )

        report = {
            "format_version": 1,
            "scoring_policy": "historical_vsp",
            "input_mode": "saved_predicted_path",
            "expected_records": expected_records,
            **summary,
        }

        (staging / "summary.json").write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Output path already exists: {target}"
            )

        staging.rename(target)
        return report

    finally:
        if staging.exists():
            shutil.rmtree(staging)
