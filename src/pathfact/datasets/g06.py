"""CPU-only construction of paired Gold/Contrastive SFT datasets.

Frozen-compatible G06 defaults are retained for reproducibility.
This is a new PathFACT implementation, not the historical script.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .io import iter_jsonl


FROZEN_NAMESPACE = (
    "contrastivelimo-joint-valid-244-train200-lossval44-v1-20260828"
)

FROZEN_TRAIN_QUOTAS = {
    "Easy": 110,
    "Medium": 75,
    "Hard": 15,
}

FROZEN_LOSSVAL_QUOTAS = {
    "Easy": 24,
    "Medium": 16,
    "Hard": 4,
}


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def difficulty(hops):
    hops = int(hops)
    if 1 <= hops <= 3:
        return "Easy"
    if 4 <= hops <= 5:
        return "Medium"
    if 6 <= hops <= 8:
        return "Hard"
    raise ValueError(f"Unexpected shortest-path hop count: {hops}")


def rank_sha(pid, namespace=FROZEN_NAMESPACE):
    return sha256_text(f"{namespace}|{pid}")


def serialize_prompt(messages):
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
    ):
        raise ValueError("Expected exactly one user message")

    return (
        "<\uff5cUser\uff5c>"
        + messages[0]["content"]
        + "<\uff5cAssistant\uff5c>"
    )


def serialize_completion(reasoning, verified_path):
    return (
        "<think>\n"
        + str(reasoning).strip()
        + "\n</think>\n"
        + "Answer: "
        + str([int(node) for node in verified_path])
    )


def sft_record(pid, condition, prompt, completion):
    return {
        "completion": completion,
        "completion_sha256": sha256_text(completion),
        "condition": condition,
        "production_case_id": pid,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
    }


def read_by_case(path):
    return {
        record["production_case_id"]: record
        for record in iter_jsonl(
            path,
            unique_key="production_case_id",
        )
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def materialize_g06(
    *,
    g00_path,
    g05_path,
    g04b_path,
    task_prompt_paths,
    output_dir,
    namespace=FROZEN_NAMESPACE,
    train_quotas=None,
    lossval_quotas=None,
    output_profile="frozen_g06",
    naming_mode="method",
):
    """Construct paired SFT records from verified upstream data.

    The default settings reproduce the historical G06 split.
    Input files are read-only; output_dir must not already exist. The
    default frozen filenames use current method names; select
    naming_mode="legacy" to reproduce the pre-v0.3 names.

    Returns a dictionary mapping output names to paths.
    """

    train_quotas = dict(
        FROZEN_TRAIN_QUOTAS
        if train_quotas is None else train_quotas
    )
    lossval_quotas = dict(
        FROZEN_LOSSVAL_QUOTAS
        if lossval_quotas is None else lossval_quotas
    )

    tiers = ("Easy", "Medium", "Hard")

    if (
        set(train_quotas) != set(tiers)
        or set(lossval_quotas) != set(tiers)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in (
                list(train_quotas.values())
                + list(lossval_quotas.values())
            )
        )
    ):
        raise ValueError("Invalid difficulty-stratified quotas")

    if output_profile not in ("frozen_g06", "generic"):
        raise ValueError("Unknown output profile")

    if naming_mode not in ("method", "legacy"):
        raise ValueError("Unknown G06 filename naming mode")

    if output_profile == "frozen_g06" and (
        namespace != FROZEN_NAMESPACE
        or train_quotas != FROZEN_TRAIN_QUOTAS
        or lossval_quotas != FROZEN_LOSSVAL_QUOTAS
    ):
        raise ValueError(
            "Use materialize_paired_sft for configurable settings"
        )

    g00 = read_by_case(g00_path)
    g05 = read_by_case(g05_path)
    g04b = read_by_case(g04b_path)

    missing_g00 = set(g05) - set(g00)
    missing_g04b = set(g05) - set(g04b)

    if missing_g00 or missing_g04b:
        raise ValueError(
            "Missing upstream cases: "
            f"G00={sorted(missing_g00)[:3]}, "
            f"G04b={sorted(missing_g04b)[:3]}"
        )

    task_prompts = {}

    for source in task_prompt_paths:
        for record in iter_jsonl(
            source,
            required_fields=("messages",),
            unique_key="production_case_id",
        ):
            pid = record["production_case_id"]

            if pid in task_prompts:
                raise ValueError(
                    f"Duplicate task-only prompt case: {pid}"
                )

            task_prompts[pid] = record["messages"]

    missing_prompts = set(g05) - set(task_prompts)

    if missing_prompts:
        raise ValueError(
            "Missing task-only prompts: "
            f"{sorted(missing_prompts)[:3]}"
        )

    by_difficulty = defaultdict(list)
    ranked = []

    for pid, pair in g05.items():
        if pair.get("joint_valid") is not True:
            raise ValueError(f"Not joint-valid: {pid}")

        diff = difficulty(g00[pid]["gold_path_hops"])
        digest = rank_sha(pid, namespace)

        item = {
            "production_case_id": pid,
            "difficulty": diff,
            "rank_sha256": digest,
        }

        by_difficulty[diff].append(item)
        ranked.append(item)

    split_by_pid = {}

    for diff in tiers:
        rows = sorted(
            by_difficulty[diff],
            key=lambda item: (
                item["rank_sha256"],
                item["production_case_id"],
            ),
        )

        train_count = train_quotas[diff]
        val_count = lossval_quotas[diff]

        if len(rows) != train_count + val_count:
            raise ValueError(
                f"{diff}: {len(rows)} cases, but quotas require "
                f"{train_count + val_count}"
            )

        for item in rows[:train_count]:
            split_by_pid[item["production_case_id"]] = "train"

        for item in rows[train_count:]:
            split_by_pid[item["production_case_id"]] = "loss_val"

    ranked.sort(
        key=lambda item: (
            item["rank_sha256"],
            item["production_case_id"],
        )
    )

    ordered_train = [
        item["production_case_id"]
        for item in ranked
        if split_by_pid[item["production_case_id"]] == "train"
    ]

    ordered_val = [
        item["production_case_id"]
        for item in ranked
        if split_by_pid[item["production_case_id"]] == "loss_val"
    ]

    datasets = {
        "gold_train": [],
        "contrastive_train": [],
        "gold_loss_val": [],
        "contrastive_loss_val": [],
    }

    for item in ranked:
        pid = item["production_case_id"]
        pair = g05[pid]
        verified_path = g00[pid]["ground_truth_shortest_path"]

        gold_prompt = serialize_prompt(task_prompts[pid])

        gold_completion = serialize_completion(
            pair["gold_target"]["teacher_reasoning_trace"],
            verified_path,
        )

        contrastive_source = g04b[pid][
            "canonical_teacher_record"
        ]

        messages = contrastive_source["messages"]
        wrong_trace = pair["canonical_wrong_reasoning_trace"]

        if (
            not isinstance(wrong_trace, str)
            or not wrong_trace
            or wrong_trace not in messages[0]["content"]
        ):
            raise ValueError(
                f"Incorrect Student trace continuity: {pid}"
            )

        contrastive_prompt = serialize_prompt(messages)

        contrastive_completion = serialize_completion(
            pair["contrastive_target"]["teacher_reasoning_trace"],
            verified_path,
        )

        split = split_by_pid[pid]

        datasets[f"gold_{split}"].append(
            sft_record(
                pid,
                "gold",
                gold_prompt,
                gold_completion,
            )
        )

        datasets[f"contrastive_{split}"].append(
            sft_record(
                pid,
                "contrastive",
                contrastive_prompt,
                contrastive_completion,
            )
        )

    output = Path(output_dir)

    # Never silently overwrite frozen or previously generated data.
    output.mkdir(parents=True, exist_ok=False)

    filenames = {
        "gold_train": (
            "limo_train_200_prompt_completion_v1.jsonl"
        ),
        "contrastive_train": (
            "contrastive_limo_train_200_prompt_completion_v1.jsonl"
        ),
        "gold_loss_val": (
            "limo_loss_val_44_prompt_completion_v1.jsonl"
        ),
        "contrastive_loss_val": (
            "contrastive_limo_loss_val_44_prompt_completion_v1.jsonl"
        ),
    }

    if naming_mode == "legacy":
        filenames = {
            "gold_train": "gold_train_200_prompt_completion_v1.jsonl",
            "contrastive_train": (
                "contrastive_train_200_prompt_completion_v1.jsonl"
            ),
            "gold_loss_val": "gold_loss_val_44_prompt_completion_v1.jsonl",
            "contrastive_loss_val": (
                "contrastive_loss_val_44_prompt_completion_v1.jsonl"
            ),
        }

    if output_profile == "generic":
        filenames = {
            "gold_train": "gold_train.jsonl",
            "contrastive_train": "contrastive_train.jsonl",
            "gold_loss_val": "gold_loss_val.jsonl",
            "contrastive_loss_val": "contrastive_loss_val.jsonl",
        }
        ranking_filename = "ranking.tsv"
        train_filename = "train_ids.txt"
        val_filename = "loss_val_ids.txt"
    else:
        ranking_filename = (
            "G06_JOINT_VALID_244_TRAIN_LOSSVAL_RANKING_V1.tsv"
        )
        train_filename = "G06_TRAIN_200_CASE_IDS_V1.txt"
        val_filename = "G06_LOSS_VAL_44_CASE_IDS_V1.txt"

    paths = {}

    for name, filename in filenames.items():
        path = output / filename
        write_jsonl(path, datasets[name])
        paths[name] = path

    ranking_path = output / ranking_filename

    ranking_path.write_text(
        "production_case_id\tdifficulty\tsplit\trank_sha256\n"
        + "".join(
            f"{item['production_case_id']}\t"
            f"{item['difficulty']}\t"
            f"{split_by_pid[item['production_case_id']]}\t"
            f"{item['rank_sha256']}\n"
            for item in ranked
        ),
        encoding="utf-8",
    )

    paths["ranking"] = ranking_path

    train_path = output / train_filename
    train_path.write_text(
        "\n".join(ordered_train) + "\n",
        encoding="utf-8",
    )
    paths["train_ids"] = train_path

    val_path = output / val_filename
    val_path.write_text(
        "\n".join(ordered_val) + "\n",
        encoding="utf-8",
    )
    paths["loss_val_ids"] = val_path

    return paths


def materialize_paired_sft(
    *,
    g00_path,
    g05_path,
    g04b_path,
    task_prompt_paths,
    output_dir,
    namespace,
    train_quotas,
    lossval_quotas,
):
    """Construct configurable paired SFT datasets.

    Uses the G06-style upstream schema and difficulty definitions,
    but does not imply historical G06 split sizes or filenames.

    Inputs must already have appropriate scientific validation.
    """

    return materialize_g06(
        g00_path=g00_path,
        g05_path=g05_path,
        g04b_path=g04b_path,
        task_prompt_paths=task_prompt_paths,
        output_dir=output_dir,
        namespace=namespace,
        train_quotas=train_quotas,
        lossval_quotas=lossval_quotas,
        output_profile="generic",
    )
