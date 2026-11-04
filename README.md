# PathFACT

**PathFACT: A Reusable Framework and Dataset for Failure-Informed LLM Shortest-Path Reasoning**

PathFACT provides weighted shortest-path tasks, student failure traces,
teacher corrections, training examples, and evaluation resources.
Start with the [Getting Started guide](GETTING_STARTED.md), then use the
[Dataset Map](DATASET_MAP.md) to choose a resource and the
[Data Dictionary](DATA_DICTIONARY.md) to interpret its fields. The guides
describe the 13 JSONL files at the fixed public revision linked below. This
GitHub README is the software project introduction; the separate
[Hugging Face Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT)
describes the published dataset, configuration files, and license scope.

PathFACT is a research project developing a reusable framework and curated dataset suite for studying failure-informed large language model reasoning on weighted shortest-path problems.

## Project scope

The project scope includes:

- Configurable weighted graph generation and exact shortest-path solving.
- Student failure selection and Student-Teacher data construction.
- Deterministic path verification and dataset validation.
- Derived shortest-path dataset resources, including the verified counterfactual export. StepPref and model-response preference construction are not implemented.
- Offline evaluation of saved model responses.

## Execution environments

The public framework is designed for a CPU-only Python environment.

Researchers provide their own environments for model inference, GPU training, and model-dependent candidate-answer likelihood scoring.

No CUDA, ROCm, PyTorch, or HPC environment is required for the CPU core.

## Research terminology

**PathFACT** is the overall framework and dataset project.

**ContrastiveLIMO** refers specifically to one historical failure-conditioned supervised fine-tuning method.

## Dataset Availability

PathFACT v0.1 is publicly available on [Hugging Face](https://huggingface.co/datasets/MinervaCYY/PathFACT) and contains 10 Core JSONL resources and 3 Derived JSONL resources. The fixed dataset release revision is [`3d20d452e1efe21c80d55185f5176ca3fc5e4f9f`](https://huggingface.co/datasets/MinervaCYY/PathFACT/tree/3d20d452e1efe21c80d55185f5176ca3fc5e4f9f).

The software is licensed under MIT (see `LICENSE`). The dataset is licensed under CC BY 4.0 within the documented licensing scope; third-party rights and exclusions are described in the Dataset Card. Dataset provenance, validation profiles, and scientific limitations are also described in the [Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT).

## Development status

The public software is licensed under MIT (see `LICENSE`). The public dataset release and its licensing scope are described in Dataset Availability above. Implemented software capabilities are listed below.

Historical research records and frozen experimental results are preserved separately.

## Implemented and verified capabilities

The current CPU-only Python implementation includes:

- Deterministic weighted graph generation and canonical shortest-path solving.
- Independent shortest-path verification, including feasibility and optimality.
- Streaming JSONL dataset handling and validation.
- Local candidate export of validated Student/Gold/Contrastive paired datasets using explicit field allowlists.
- Local core release-candidate assembly for G00/G05/G06 and the four completed
  G08/G09 evaluation views.
- Configurable construction of paired Gold and Contrastive SFT datasets from supplied historical records.
- Historical path-verifier scoring of saved candidate paths.
- Historical extraction of shortest-path answers from saved model messages.
- Offline evaluation summaries and per-case scoring outputs.

Automated regression coverage includes portable synthetic-data tests and optional frozen-reference tests.
Verified references include 2,200 G00 cases, 10,338 G08/G09 saved evaluation records, and 244 G05 paired cases.
Among the G05 pairs, 141 have additional two-sided reasoning factuality validation; 103 have solver-verified final-path checks only.
These validation profiles are preserved from historical records; the exporter does not independently re-audit Teacher reasoning.

Student inference, Teacher generation, GPU training, and model-dependent scoring are not provided by the CPU-only package.
The local derived-candidate builder supports the verified 354 G09
Original/Stay/Switch triples, the 200-case G06 ranking-eligibility partition,
and the 162-case G06 Boundary-Local resource, for exactly 3 JSONL resources. It excludes model responses and
StepPref, and emits only a private draft candidate without publication approval.
See `docs/DERIVED_DATASET_RELEASE_BUILDER.md` for the interface and schemas.

## Installation

PathFACT requires Python 3.12 or newer.

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e .

## Command-line interfaces

Inspect the supported commands:

    pathfact --help
    pathfact path verify --help
    pathfact dataset build-paired-sft --help
    pathfact dataset export-public --help
    pathfact dataset export-core-candidate --help
    pathfact dataset export-derived-candidate --help
    pathfact evaluate saved --help

The dataset builder requires source records supplied by the researcher.
The saved-path evaluator requires previously extracted candidate paths.

### Local candidate dataset export

The candidate exporter accepts canonical G00, G02, G04a, G04b, and G05 JSONL files.

Example:

    pathfact dataset export-public --g00 G00.jsonl --g02 G02.jsonl --g04a G04a.jsonl --g04b G04b.jsonl --g05 G05.jsonl --output-dir candidate_dataset

An optional --expected-records argument enforces an exact case count.

The output directory must not already exist. Successful exports produce
paired.jsonl and summary.json.

The exporter checks Student failure eligibility, cross-stage identities,
text hashes, and recorded Teacher validation profiles. It excludes selected
internal request and service metadata through explicit field allowlists.

This is a local candidate export, not an approved public dataset release.
Source text still requires content and redistribution-rights review.

### Core dataset release candidate

The core candidate builder assembles allowlisted G00 tasks, the validated G05
paired dataset, byte-preserved G06 Train/LossVal JSONL files, and normalized
G08/G09 Balanced and TrainDist views, for exactly 10 JSONL resources. It also creates a draft integrity
manifest and a local-candidate summary.

    pathfact dataset export-core-candidate --help

The command requires canonical G00/G02/G04a/G04b/G05 inputs, all four G06
JSONL files, G08 Balanced/TrainDist and G09 Balanced/TrainDist. Its destination
must not already exist. It excludes G08 Diagnostic-117, G09 Pool-1956 and
derived datasets, and it does not approve or publish the candidate.

See [Core Dataset Release Builder](docs/CORE_DATASET_RELEASE_BUILDER.md).

See the [Data Dictionary](DATA_DICTIONARY.md) for released fields and the
[Hugging Face Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT)
for release configurations and dataset license scope.

For evaluation inputs, outputs, scoring rules, and limitations, see [Offline Evaluation](docs/EVALUATION.md).

## Testing

Run the portable test suite:

    python -m unittest discover -s tests -v

Frozen-source regression checks require separately supplied historical reference datasets and are skipped when their reference environment variables are not configured. The optional variables are `PATHFACT_PIPELINE_REFERENCE`, `PATHFACT_G00_REFERENCE`, and `PATHFACT_G06_REFERENCE`. The supplied latest result was 113 total, 106 passed, and 7 skipped. A fresh audit run without those references also discovered 113 tests, with 103 passed and 10 skipped: seven checks gated by `PATHFACT_PIPELINE_REFERENCE`, two by `PATHFACT_G00_REFERENCE`, and one by `PATHFACT_G06_REFERENCE`. The three additional skips explain the difference from the supplied result; skipped tests are not passes.

## Release boundary

The software is MIT licensed. The public PathFACT v0.1 dataset is licensed under CC BY 4.0 only within the scope documented in its Dataset Card; this does not grant rights to third-party content outside that scope. Researchers supply their own model inference and GPU training environments. Historical Bunya implementation is internal provenance, not a public runtime dependency. Dataset provenance and scientific limitations are documented in the Dataset Card.

## Source provenance



Historical source fingerprints, implementation mappings, and verified reproducibility evidence are documented in [Source Provenance](docs/SOURCE_PROVENANCE.md).

### Dataset integrity manifests

PathFACT supports CPU-only dataset integrity manifests.

Create: pathfact dataset manifest-create --root-dir candidate_release --artifacts-json artifacts.json --dataset-version local-draft-v0.1

Verify: pathfact dataset manifest-verify --root-dir candidate_release

Verification checks SHA-256, byte sizes, JSONL counts and G05 validation tiers. Source hashes are recorded as provenance metadata, not independently reverified. The local candidate manifests created by these commands remain `draft_not_approved`; verifying them does not authorize publication. This is separate from the already published Hugging Face v0.1 package and its fixed release revision above.

See [Dataset Licensing](docs/DATASET_LICENSE.md) for the dataset's limited
CC BY 4.0 scope and exclusions.
