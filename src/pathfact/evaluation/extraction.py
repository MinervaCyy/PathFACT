"""Historical G08/G09 model-response extraction.

This module reconstructs fields from successful model messages.
Request errors are handled separately by the evaluation workflow.

No model inference is performed.
"""

from __future__ import annotations

from typing import Any

from .historical_vsp import (
    extract_path,
    split_reasoning_and_content,
)


def extract_historical_message(
    message: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the historical successful-response fields.

    Preserves the G08/G09 extraction sequence exactly.
    In particular, a separate reasoning trace is not searched
    for an answer when final content has no extractable path.
    """

    if not isinstance(message, dict):
        raise TypeError("Model message must be a dictionary")

    reasoning_trace, final_content = (
        split_reasoning_and_content(message)
    )

    predicted_path = extract_path(final_content)

    # Historical fallback for messages without a separate
    # reasoning trace.
    if not predicted_path and not reasoning_trace:
        predicted_path = extract_path(
            str(message.get("content") or "")
        )

    return {
        "reasoning_trace": reasoning_trace,
        "final_content": final_content,
        "reasoning_present": bool(
            reasoning_trace.strip()
        ),
        "final_content_present": bool(
            final_content.strip()
        ),
        "predicted_path": predicted_path,
        "answer_extracted": bool(predicted_path),
    }
