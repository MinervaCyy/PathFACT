"""Command-line entry point for PathFACT."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pathfact import __version__


def nonnegative_int(value):
    """Parse a non-negative integer for dataset quotas."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected an integer"
        ) from exc

    if number < 0:
        raise argparse.ArgumentTypeError(
            "Quota must be non-negative"
        )

    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pathfact",
        description=(
            "PathFACT: A reusable framework for "
            "failure-informed LLM shortest-path reasoning."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"PathFACT {__version__}",
    )

    commands = parser.add_subparsers(dest="command")

    dataset_parser = commands.add_parser(
        "dataset",
        help="Dataset construction and verification",
    )

    dataset_commands = dataset_parser.add_subparsers(
        dest="dataset_command"
    )

    build_parser = dataset_commands.add_parser(
        "build-paired-sft",
        help="Construct configurable paired Gold/Contrastive SFT data",
        description=(
            "Construct paired SFT datasets using G06-style "
            "validated upstream records and configurable splits."
        ),
    )

    build_parser.add_argument(
        "--g00",
        type=Path,
        required=True,
        help="JSONL containing shortest-path ground truth",
    )

    build_parser.add_argument(
        "--g05",
        type=Path,
        required=True,
        help="JSONL containing paired validated Teacher targets",
    )

    build_parser.add_argument(
        "--g04b",
        type=Path,
        required=True,
        help="JSONL containing canonical Contrastive Teacher inputs",
    )

    build_parser.add_argument(
        "--task-prompt",
        type=Path,
        action="append",
        required=True,
        help="Task-only prompt JSONL; repeat for multiple files",
    )

    build_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; must not already exist",
    )

    build_parser.add_argument(
        "--namespace",
        required=True,
        help="Namespace for deterministic case ranking",
    )

    build_parser.add_argument(
        "--train-quotas",
        type=nonnegative_int,
        nargs=3,
        metavar=("EASY", "MEDIUM", "HARD"),
        required=True,
        help="Training case quotas by difficulty",
    )

    build_parser.add_argument(
        "--lossval-quotas",
        type=nonnegative_int,
        nargs=3,
        metavar=("EASY", "MEDIUM", "HARD"),
        required=True,
        help="Loss-validation case quotas by difficulty",
    )


    vanilla_parser = dataset_commands.add_parser(
        "materialize-vanilla",
        help="Materialize answer-only Vanilla-SFT datasets",
    )

    vanilla_parser.add_argument(
        "--gold-train",
        type=Path,
        required=True,
        help="Validated G06 Gold Train-200 JSONL",
    )

    vanilla_parser.add_argument(
        "--gold-loss-val",
        type=Path,
        required=True,
        help="Validated G06 Gold LossVal-44 JSONL",
    )

    vanilla_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; must not already exist",
    )

    export_parser = dataset_commands.add_parser(
        "export-public",
        help="Create a local candidate dataset; not approved for release",
    )

    for flag, description in (
        ("--g00", "Canonical graph task JSONL"),
        ("--g02", "Canonical Student failure JSONL"),
        ("--g04a", "Gold Teacher JSONL"),
        ("--g04b", "Contrastive Teacher JSONL"),
        ("--g05", "Joint-valid paired Teacher JSONL"),
    ):
        export_parser.add_argument(
            flag, type=Path, required=True, help=description
        )

    export_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New local candidate output directory",
    )

    export_parser.add_argument(
        "--expected-records",
        type=nonnegative_int,
        default=None,
        help="Optional exact expected case count",
    )


    core_export_parser = dataset_commands.add_parser(
        "export-core-candidate",
        help="Build a local G00/G05/G06/G08/G09 release candidate",
        description=(
            "Build a local-only core dataset candidate containing "
            "allowlisted G00 tasks, allowlisted G05 paired records, "
            "byte-preserved G06 JSONL files, and normalized G08/G09 "
            "evaluation views."
        ),
    )

    for flag, description in (
        ("--g00", "Canonical G00 task JSONL"),
        ("--g02", "Canonical G02 Student failure JSONL"),
        ("--g04a", "Canonical G04a Gold Teacher JSONL"),
        ("--g04b", "Canonical G04b Contrastive Teacher JSONL"),
        ("--g05", "Canonical G05 joint-valid JSONL"),
        ("--g06-gold-train", "Frozen G06 Gold Train JSONL"),
        (
            "--g06-contrastive-train",
            "Frozen G06 Contrastive Train JSONL",
        ),
        ("--g06-gold-loss-val", "Frozen G06 Gold LossVal JSONL"),
        (
            "--g06-contrastive-loss-val",
            "Frozen G06 Contrastive LossVal JSONL",
        ),
        ("--g08-balanced", "Frozen G08 Balanced JSONL"),
        ("--g08-train-dist", "Frozen G08 TrainDist JSONL"),
        ("--g09-balanced", "Frozen G09 Balanced JSONL"),
        ("--g09-train-dist", "Frozen G09 TrainDist JSONL"),
    ):
        core_export_parser.add_argument(
            flag, type=Path, required=True, help=description
        )

    core_export_parser.add_argument(
        "--g06-vanilla-train",
        type=Path,
        default=None,
        help="Optional validated Vanilla-SFT Train-200 JSONL",
    )

    core_export_parser.add_argument(
        "--g06-vanilla-loss-val",
        type=Path,
        default=None,
        help="Optional validated Vanilla-SFT LossVal-44 JSONL",
    )

    core_export_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New local candidate directory; must not already exist",
    )
    core_export_parser.add_argument(
        "--dataset-version",
        required=True,
        help="Local candidate version identifier",
    )

    for flag, default, description in (
        ("--expected-g00-records", 2200, "Expected G00 records"),
        ("--expected-paired-records", 244, "Expected G05 records"),
        ("--expected-train-records", 200, "Expected records per Train file"),
        (
            "--expected-loss-val-records",
            44,
            "Expected records per LossVal file",
        ),
        (
            "--expected-reasoning-validated-pairs",
            141,
            "Expected two-sided reasoning-validated G05 pairs",
        ),
        (
            "--expected-vsp-only-pairs",
            103,
            "Expected VSP-only G05 pairs",
        ),
        (
            "--expected-g08-balanced-records",
            300,
            "Expected G08 Balanced records",
        ),
        (
            "--expected-g08-train-dist-records",
            300,
            "Expected G08 TrainDist records",
        ),
        (
            "--expected-g09-balanced-records",
            846,
            "Expected G09 Balanced records",
        ),
        (
            "--expected-g09-train-dist-records",
            1600,
            "Expected G09 TrainDist records",
        ),
        (
            "--expected-g09-overlap",
            617,
            "Expected G09 Balanced/TrainDist overlap",
        ),
        (
            "--expected-g08-balanced-g09-balanced-overlap",
            164,
            "Expected G08 Balanced/G09 Balanced overlap",
        ),
        (
            "--expected-g08-balanced-g09-train-dist-overlap",
            221,
            "Expected G08 Balanced/G09 TrainDist overlap",
        ),
        (
            "--expected-g08-train-dist-g09-balanced-overlap",
            110,
            "Expected G08 TrainDist/G09 Balanced overlap",
        ),
        (
            "--expected-g08-train-dist-g09-train-dist-overlap",
            265,
            "Expected G08 TrainDist/G09 TrainDist overlap",
        ),
    ):
        core_export_parser.add_argument(
            flag,
            type=nonnegative_int,
            default=default,
            help=description,
        )

    derived_export_parser = dataset_commands.add_parser(
        "export-derived-candidate",
        help="Build a local draft from verified counterfactual and ranking resources",
        description=(
            "Normalize frozen counterfactual triples, G06 ranking eligibility, "
            "and Boundary-Local records into a private local draft candidate."
        ),
    )
    for flag, description in (
        ("--counterfactual", "Finalized Original/Stay/Switch JSONL"),
        ("--g09-balanced", "Core G09 Balanced evaluation JSONL"),
        ("--g09-train-dist", "Core G09 TrainDist evaluation JSONL"),
        ("--ranking", "Frozen G06 ranking source JSONL"),
        ("--eligibility", "Frozen G06 eligibility JSONL"),
        ("--core-g06-train", "Core G06 Gold Train JSONL"),
        ("--boundary-local", "Frozen Boundary-Local strict-162 JSONL"),
        ("--boundary-protocol", "Frozen Boundary-Local protocol JSON"),
        ("--counterfactual-script", "Frozen counterfactual input construction script"),
        ("--ranking-script", "Frozen G06 ranking construction script"),
        ("--boundary-script", "Frozen Boundary-Local construction script"),
        ("--scoring-report", "October 15 unified scoring report JSON"),
    ):
        derived_export_parser.add_argument(
            flag, type=Path, required=True, help=description
        )
    derived_export_parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New private local candidate directory; must not exist",
    )
    derived_export_parser.add_argument(
        "--dataset-version", required=True,
        help="Draft candidate version identifier",
    )


    manifest_create_parser = dataset_commands.add_parser(
        "manifest-create",
        help="Create a draft dataset integrity manifest",
    )
    manifest_create_parser.add_argument(
        "--root-dir", type=Path, required=True,
        help="Existing local dataset release directory",
    )
    manifest_create_parser.add_argument(
        "--artifacts-json", type=Path, required=True,
        help="JSON file listing explicitly selected artifacts",
    )
    manifest_create_parser.add_argument(
        "--dataset-version", required=True,
        help="Version identifier for the candidate dataset",
    )

    manifest_verify_parser = dataset_commands.add_parser(
        "manifest-verify",
        help="Verify a local dataset integrity manifest",
    )
    manifest_verify_parser.add_argument(
        "--root-dir", type=Path, required=True,
        help="Directory containing manifest.json and dataset files",
    )

    path_parser = commands.add_parser(
        "path",
        help="Shortest-path verification",
    )

    path_commands = path_parser.add_subparsers(
        dest="path_command"
    )

    verify_parser = path_commands.add_parser(
        "verify",
        help="Verify a candidate weighted shortest path",
    )

    verify_parser.add_argument(
        "--graph-file",
        type=Path,
        required=True,
        help="JSON file containing weighted adjacency lists",
    )

    verify_parser.add_argument(
        "--source",
        type=int,
        required=True,
        help="Source node identifier",
    )

    verify_parser.add_argument(
        "--destination",
        type=int,
        required=True,
        help="Destination node identifier",
    )

    verify_parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate path as a JSON list, or null",
    )

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="CPU-only offline evaluation of saved model outputs",
    )

    evaluate_commands = evaluate_parser.add_subparsers(
        dest="evaluate_command"
    )

    saved_parser = evaluate_commands.add_parser(
        "saved",
        help="Re-score saved candidate paths using historical VSP",
    )

    saved_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL file containing saved predicted_path values",
    )

    saved_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for per-case scores and summary",
    )

    saved_parser.add_argument(
        "--expected-records",
        type=nonnegative_int,
        default=None,
        help="Require exactly this many input records",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "path":
        if args.path_command is None:
            path_parser.print_help()
            return 0

        from pathfact.graphs import graph_from_topology_dict
        from pathfact.validation import validate_candidate_path

        try:
            graph_data = json.loads(
                args.graph_file.read_text(encoding="utf-8")
            )

            if not isinstance(graph_data, dict):
                raise ValueError(
                    "Graph JSON must contain an adjacency object"
                )

            candidate = json.loads(args.candidate)

            graph = graph_from_topology_dict(graph_data)

            result = validate_candidate_path(
                graph,
                args.source,
                args.destination,
                candidate,
            )

        except (OSError, ValueError, TypeError, KeyError) as exc:
            verify_parser.error(str(exc))

        print(json.dumps(asdict(result), sort_keys=True))
        return 0

    if args.command == "evaluate":
        if args.evaluate_command is None:
            evaluate_parser.print_help()
            return 0

        from pathfact.evaluation.saved import (
            evaluate_saved_jsonl,
        )

        try:
            report = evaluate_saved_jsonl(
                input_path=args.input,
                output_dir=args.output_dir,
                expected_records=args.expected_records,
            )

        except (OSError, ValueError, TypeError, KeyError) as exc:
            saved_parser.error(str(exc))

        print(json.dumps(report, sort_keys=True))
        return 0

    if args.dataset_command is None:
        dataset_parser.print_help()
        return 0

    if args.dataset_command == "build-paired-sft":
        if not args.namespace.strip():
            build_parser.error(
                "--namespace must not be empty"
            )

        from pathfact.datasets.g06 import materialize_paired_sft

        tiers = ("Easy", "Medium", "Hard")

        try:
            outputs = materialize_paired_sft(
                g00_path=args.g00,
                g05_path=args.g05,
                g04b_path=args.g04b,
                task_prompt_paths=args.task_prompt,
                output_dir=args.output_dir,
                namespace=args.namespace,
                train_quotas=dict(zip(
                    tiers, args.train_quotas
                )),
                lossval_quotas=dict(zip(
                    tiers, args.lossval_quotas
                )),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            build_parser.error(str(exc))

        print("Paired SFT construction completed.")

        for name, path in outputs.items():
            print(f"{name}: {path}")

        return 0


    if args.dataset_command == "materialize-vanilla":
        from pathfact.datasets.vanilla import materialize_vanilla_sft

        try:
            outputs = materialize_vanilla_sft(
                gold_train_path=args.gold_train,
                gold_loss_val_path=args.gold_loss_val,
                output_dir=args.output_dir,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            vanilla_parser.error(str(exc))

        print("Vanilla SFT materialisation completed.")

        for name, path in outputs.items():
            print(f"{name}: {path}")

        return 0

    if args.dataset_command == "export-public":
        from pathfact.datasets.public import export_public_pairs

        try:
            outputs = export_public_pairs(
                g00_path=args.g00,
                g02_path=args.g02,
                g04a_path=args.g04a,
                g04b_path=args.g04b,
                g05_path=args.g05,
                output_dir=args.output_dir,
                expected_records=args.expected_records,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            export_parser.error(str(exc))

        print("Local candidate export completed.")
        print("Redistribution has not been approved.")

        for name, output_path in outputs.items():
            print(f"{name}: {output_path}")

        return 0


    if args.dataset_command == "export-core-candidate":
        from pathfact.datasets.public import (
            export_core_release_candidate,
        )

        try:
            outputs = export_core_release_candidate(
                g00_path=args.g00,
                g02_path=args.g02,
                g04a_path=args.g04a,
                g04b_path=args.g04b,
                g05_path=args.g05,
                g06_gold_train_path=args.g06_gold_train,
                g06_contrastive_train_path=(
                    args.g06_contrastive_train
                ),
                g06_gold_loss_val_path=args.g06_gold_loss_val,
                g06_contrastive_loss_val_path=(
                    args.g06_contrastive_loss_val
                ),
                g06_vanilla_train_path=args.g06_vanilla_train,
                g06_vanilla_loss_val_path=(
                    args.g06_vanilla_loss_val
                ),
                g08_balanced_path=args.g08_balanced,
                g08_train_dist_path=args.g08_train_dist,
                g09_balanced_path=args.g09_balanced,
                g09_train_dist_path=args.g09_train_dist,
                output_dir=args.output_dir,
                dataset_version=args.dataset_version,
                expected_g00_records=args.expected_g00_records,
                expected_paired_records=args.expected_paired_records,
                expected_train_records=args.expected_train_records,
                expected_loss_val_records=(
                    args.expected_loss_val_records
                ),
                expected_reasoning_validated_pairs=(
                    args.expected_reasoning_validated_pairs
                ),
                expected_vsp_only_pairs=(
                    args.expected_vsp_only_pairs
                ),
                expected_g08_balanced_records=(
                    args.expected_g08_balanced_records
                ),
                expected_g08_train_dist_records=(
                    args.expected_g08_train_dist_records
                ),
                expected_g09_balanced_records=(
                    args.expected_g09_balanced_records
                ),
                expected_g09_train_dist_records=(
                    args.expected_g09_train_dist_records
                ),
                expected_g09_overlap=args.expected_g09_overlap,
                expected_g08_balanced_g09_balanced_overlap=(
                    args.expected_g08_balanced_g09_balanced_overlap
                ),
                expected_g08_balanced_g09_train_dist_overlap=(
                    args.expected_g08_balanced_g09_train_dist_overlap
                ),
                expected_g08_train_dist_g09_balanced_overlap=(
                    args.expected_g08_train_dist_g09_balanced_overlap
                ),
                expected_g08_train_dist_g09_train_dist_overlap=(
                    args.expected_g08_train_dist_g09_train_dist_overlap
                ),
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            core_export_parser.error(str(exc))

        print("Local core release candidate completed.")
        print("Redistribution has not been approved.")

        for name, output_path in outputs.items():
            print(f"{name}: {output_path}")

        return 0

    if args.dataset_command == "export-derived-candidate":
        from pathfact.datasets.derived import export_derived_candidate

        try:
            outputs = export_derived_candidate(
                counterfactual_path=args.counterfactual,
                g09_balanced_path=args.g09_balanced,
                g09_train_dist_path=args.g09_train_dist,
                ranking_path=args.ranking,
                eligibility_path=args.eligibility,
                core_g06_train_path=args.core_g06_train,
                boundary_path=args.boundary_local,
                boundary_protocol_path=args.boundary_protocol,
                counterfactual_script_path=args.counterfactual_script,
                ranking_script_path=args.ranking_script,
                boundary_script_path=args.boundary_script,
                scoring_report_path=args.scoring_report,
                output_dir=args.output_dir,
                dataset_version=args.dataset_version,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            derived_export_parser.error(str(exc))

        print("Local derived candidate completed.")
        print("Redistribution has not been approved.")
        for name, output_path in outputs.items():
            print(f"{name}: {output_path}")
        return 0


    if args.dataset_command == "manifest-create":
        from pathfact.datasets.manifest import create_release_manifest

        try:
            artifacts = json.loads(
                args.artifacts_json.read_text(encoding="utf-8")
            )
            destination = create_release_manifest(
                root_dir=args.root_dir,
                artifacts=artifacts,
                dataset_version=args.dataset_version,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            manifest_create_parser.error(str(exc))

        print("Draft manifest created:", destination)
        print("This manifest does not authorize data publication.")
        return 0

    if args.dataset_command == "manifest-verify":
        from pathfact.datasets.manifest import verify_release_manifest

        try:
            report = verify_release_manifest(args.root_dir)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            manifest_verify_parser.error(str(exc))

        print(json.dumps(report, sort_keys=True))
        return 0

    parser.error("Unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
