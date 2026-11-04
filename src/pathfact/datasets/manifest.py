"""CPU-only integrity manifests for local PathFACT dataset candidates."""

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from pathfact.datasets.io import DatasetFormatError, iter_jsonl


SCHEMA = "pathfact-release-manifest-v0.1"

RELEASE_DIRS = {
    "tasks", "student", "teacher", "paired",
    "training", "evaluation", "counterfactual", "metadata",
}

VALID_G05_PROFILES = {
    (
        "vsp_plus_deterministic_rt_factuality_audit",
        "vsp_plus_corrective_rt_factuality_validation",
    ): "both_reasoning_factuality_validated",
    (
        "solver_verified_vsp",
        "solver_verified_vsp",
    ): "both_vsp_only",
}


def _root(directory):
    root = Path(directory)

    if not root.is_dir() or root.is_symlink():
        raise DatasetFormatError("Invalid dataset root directory")

    return root


def _safe_file(root, relative):
    if not isinstance(relative, str):
        raise DatasetFormatError("Artifact path must be a string")

    parts = relative.split("/")

    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or ":" in relative
        or any(part in ("", ".", "..") for part in parts)
        or parts[0] not in RELEASE_DIRS
    ):
        raise DatasetFormatError(
            f"Unsafe release artifact path: {relative}"
        )

    path = root

    for part in parts:
        path = path / part
        if path.is_symlink():
            raise DatasetFormatError(
                f"Symlink not allowed: {relative}"
            )

    if path.suffix != ".jsonl" or not path.is_file():
        raise DatasetFormatError(
            f"Expected an existing JSONL file: {relative}"
        )

    return path


def _sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _check_source_hashes(value):
    if value is None:
        return None

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, dict) and value:
        if not all(isinstance(k, str) and k for k in value):
            raise DatasetFormatError("Invalid source hash keys")
        values = list(value.values())
    else:
        raise DatasetFormatError("Invalid source_sha256")

    for digest in values:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise DatasetFormatError("Invalid source SHA-256 value")

    return value


def _g05_tier(row):
    try:
        gold = row["teacher_targets"]["gold"]
        contrastive = row["teacher_targets"]["contrastive"]

        if (
            row["joint_valid"] is not True
            or gold["teacher_vsp_valid"] is not True
            or contrastive["teacher_vsp_valid"] is not True
        ):
            raise DatasetFormatError("G05 validity flag mismatch")

        profiles = (
            gold["validation_profile"],
            contrastive["validation_profile"],
        )

        tier = VALID_G05_PROFILES.get(profiles)

        if tier is None or row["validation_tier"] != tier:
            raise DatasetFormatError("G05 validation tier mismatch")

        return tier

    except (KeyError, TypeError) as exc:
        raise DatasetFormatError(
            "Malformed G05 validation metadata"
        ) from exc


def _check_stage_location(relative, stage):
    """Prevent paired data from bypassing G05 checks."""
    is_paired = relative.split("/", 1)[0] == "paired"
    is_g05 = stage == "G05"

    if is_paired != is_g05:
        raise DatasetFormatError(
            "G05 source_stage/path mismatch"
        )


def _inspect(path, stage):
    before = _sha256(path)
    size_before = path.stat().st_size

    count = 0
    tiers = Counter()

    for row in iter_jsonl(
        path, unique_key="production_case_id"
    ):
        count += 1

        if stage == "G05":
            tiers[_g05_tier(row)] += 1

    after = _sha256(path)
    size_after = path.stat().st_size

    if before != after or size_before != size_after:
        raise DatasetFormatError(
            f"Artifact changed during inspection: {path.name}"
        )

    return {
        "record_count": count,
        "size_bytes": size_after,
        "sha256": after,
        "validation_profile": (
            dict(sorted(tiers.items())) if stage == "G05"
            else None
        ),
    }


def create_release_manifest(root_dir, artifacts, dataset_version):
    """Create a draft manifest for explicitly selected JSONL artifacts."""
    root = _root(root_dir)
    destination = root / "manifest.json"

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise DatasetFormatError("Dataset version is required")

    if not isinstance(artifacts, list) or not artifacts:
        raise DatasetFormatError("Explicit artifact list is required")

    entries = []
    seen = set()

    for spec in artifacts:
        if not isinstance(spec, dict):
            raise DatasetFormatError("Artifact specification must be an object")

        relative = spec.get("path")
        stage = spec.get("source_stage")
        version = spec.get("schema_version")

        if not isinstance(stage, str) or not stage:
            raise DatasetFormatError("Missing source_stage")

        if not isinstance(version, str) or not version:
            raise DatasetFormatError("Missing artifact schema_version")

        path = _safe_file(root, relative)
        _check_stage_location(relative, stage)

        if relative in seen:
            raise DatasetFormatError("Duplicate manifest artifact path")

        seen.add(relative)

        entries.append({
            "path": relative,
            "source_stage": stage,
            "schema_version": version,
            "source_sha256": _check_source_hashes(
                spec.get("source_sha256")
            ),
            **_inspect(path, stage),
        })

    manifest = {
        "dataset_name": "PathFACT",
        "dataset_version": dataset_version,
        "schema_version": SCHEMA,
        "release_status": "draft_not_approved",
        "license": None,
        "files": sorted(entries, key=lambda item: item["path"]),
    }

    fd, temporary = tempfile.mkstemp(
        prefix=".pathfact-manifest-",
        suffix=".tmp",
        dir=root,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        # Atomic create without overwriting an existing manifest.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)

    return destination


def verify_release_manifest(root_dir):
    """Verify released-file bytes, JSONL counts, and recorded G05 tiers."""
    root = _root(root_dir)
    source = root / "manifest.json"

    if source.is_symlink() or not source.is_file():
        raise DatasetFormatError("Missing or unsafe manifest.json")

    manifest = json.loads(source.read_text(encoding="utf-8"))

    if (
        not isinstance(manifest, dict)
        or manifest.get("dataset_name") != "PathFACT"
        or manifest.get("schema_version") != SCHEMA
        or not isinstance(manifest.get("files"), list)
    ):
        raise DatasetFormatError("Invalid release manifest")

    if (
        manifest.get("release_status") != "draft_not_approved"
        or manifest.get("license") is not None
    ):
        raise DatasetFormatError(
            "Manifest v0.1 is draft-only; "
            "integrity verification is not release approval"
        )

    seen = set()

    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise DatasetFormatError("Invalid artifact entry")

        relative = entry.get("path")

        if relative in seen:
            raise DatasetFormatError("Duplicate manifest path")

        seen.add(relative)

        _check_source_hashes(entry.get("source_sha256"))

        path = _safe_file(root, relative)
        stage = entry.get("source_stage")
        _check_stage_location(relative, stage)
        actual = _inspect(path, stage)

        for key, observed in actual.items():
            if (
                key not in entry
                or type(entry[key]) is not type(observed)
                or entry[key] != observed
            ):
                raise DatasetFormatError(
                    f"{relative}: mismatch in {key}"
                )

    return {
        "files_verified": len(seen),
        "status": "PASS",
        "release_status": manifest.get("release_status"),
    }
