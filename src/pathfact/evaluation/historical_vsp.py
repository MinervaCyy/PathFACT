"""Historical VSP-compatible answer extraction and scoring.

Functions extracted from the fingerprint-verified historical
evaluate_deepseek_zeroshot.py implementation.

The scientific scoring logic is preserved without modification.
This module performs no model inference or GPU operations.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any


def parse_maybe_literal(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def normalise_topology(value: Any) -> dict[int, list[tuple[int, float]]]:
    topology = parse_maybe_literal(value)
    if not isinstance(topology, dict):
        raise TypeError(f"Topology must be a dict, got {type(topology).__name__}")

    normalised: dict[int, list[tuple[int, float]]] = {}
    for raw_node, raw_neighbours in topology.items():
        node = int(raw_node)
        neighbours: list[tuple[int, float]] = []
        for item in raw_neighbours:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Invalid adjacency item for node {node}: {item!r}")
            neighbour, weight = item
            neighbours.append((int(neighbour), float(weight)))
        normalised[node] = neighbours
    return normalised


def normalise_node_pair(value: Any) -> list[int]:
    pair = parse_maybe_literal(value)
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f"Invalid node_pair: {value!r}")
    return [int(pair[0]), int(pair[1])]


def extract_path(text: str) -> list[int]:
    if not text:
        return []

    answer_matches = list(
        re.finditer(
            r"Answer:\s*\[([\d,\s-]+)\]",
            text,
            flags=re.IGNORECASE,
        )
    )
    candidates = answer_matches

    if not candidates:
        candidates = list(re.finditer(r"\[([\d,\s-]+)\]", text))

    if not candidates:
        return []

    try:
        values = [
            int(value.strip())
            for value in candidates[-1].group(1).split(",")
            if value.strip()
        ]
    except ValueError:
        return []
    return values


def split_reasoning_and_content(message: dict[str, Any]) -> tuple[str, str]:
    reasoning = (
        message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )
    content = message.get("content") or ""

    # Compatibility fallback for vLLM builds without a reasoning parser.
    if not reasoning and "</think>" in content:
        reasoning_part, final_part = content.split("</think>", 1)
        reasoning = reasoning_part.removeprefix("<think>").strip()
        content = final_part.strip()

    return str(reasoning), str(content)


def edge_weight(
    topology: dict[int, list[tuple[int, float]]],
    source: int,
    target: int,
) -> float | None:
    for neighbour, weight in topology.get(source, []):
        if neighbour == target:
            return weight
    return None


def verify_path(record: dict[str, Any], predicted_path: list[int]) -> dict[str, Any]:
    topology = normalise_topology(record["topology"])
    source, target = normalise_node_pair(record["node_pair"])
    gold_cost = float(record["gold_cost"])

    syntactic_valid = bool(predicted_path) and all(
        isinstance(node, int) for node in predicted_path
    )
    source_target_consistent = (
        syntactic_valid
        and predicted_path[0] == source
        and predicted_path[-1] == target
    )
    no_repeated_nodes = len(predicted_path) == len(set(predicted_path))

    path_continuity = syntactic_valid
    predicted_cost = 0.0
    missing_edges: list[list[int]] = []
    edge_checks: list[dict[str, Any]] = []

    for u, v in zip(predicted_path, predicted_path[1:]):
        weight = edge_weight(topology, u, v)
        if weight is None:
            path_continuity = False
            missing_edges.append([u, v])
            edge_checks.append({"source": u, "target": v, "exists": False})
        else:
            predicted_cost += weight
            edge_checks.append(
                {
                    "source": u,
                    "target": v,
                    "exists": True,
                    "weight": weight,
                }
            )

    optimal_cost = path_continuity and math.isclose(
        predicted_cost,
        gold_cost,
        rel_tol=0.0,
        abs_tol=1e-9,
    )

    vsp_valid = all(
        (
            syntactic_valid,
            source_target_consistent,
            no_repeated_nodes,
            path_continuity,
            optimal_cost,
        )
    )

    if not syntactic_valid:
        failure_category = "no_path_extracted"
    elif not source_target_consistent:
        failure_category = "wrong_source_or_destination"
    elif not no_repeated_nodes:
        failure_category = "repeated_nodes"
    elif not path_continuity:
        failure_category = "nonexistent_edge"
    elif not optimal_cost:
        failure_category = "continuous_but_suboptimal"
    else:
        failure_category = "valid"

    return {
        "syntactic_valid": syntactic_valid,
        "source_target_consistent": source_target_consistent,
        "no_repeated_nodes": no_repeated_nodes,
        "path_continuity": path_continuity,
        "optimal_cost": optimal_cost,
        "vsp_valid": vsp_valid,
        "failure_category": failure_category,
        "predicted_cost": predicted_cost if path_continuity else None,
        "gold_cost": gold_cost,
        "missing_edges": missing_edges,
        "edge_checks": edge_checks,
    }
