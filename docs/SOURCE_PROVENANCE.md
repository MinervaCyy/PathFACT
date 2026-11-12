# PathFACT Source Provenance

This document records verified historical source fingerprints, implementation mappings, and reproducibility evidence.

**Current dataset release (2026-11-04):** The 13-resource PathFACT v0.1 dataset (10 Core, 3 Derived) is publicly available on Hugging Face at revision [`3d20d452e1efe21c80d55185f5176ca3fc5e4f9f`](https://huggingface.co/datasets/MinervaCYY/PathFACT/tree/3d20d452e1efe21c80d55185f5176ca3fc5e4f9f). Its limited CC BY 4.0 scope is separate from the software MIT licence and excludes third-party material outside publisher authority.

Historical-status note: statements in the release-boundary section about permissions and licensing record the status when this provenance document was prepared. The subsequent v0.1 release package records researcher authorization for its scoped publication. This does not establish institutional approval or rights in excluded material. RC1 manifests remain historical and unchanged.

Historical research scripts are provenance references, not dependencies of the public execution environment.

## 1. Historical source fingerprints

### Graph generation and shortest-path solving

Original file: generate_region_conditioned_pilot.py
SHA-256: c2f17a4140f889823f2c91fc8e7a0606390e0ed9254e080bf929e8aa15b87ed1

Related PathFACT modules:
- src/pathfact/graphs/canonical.py
- src/pathfact/solver/canonical.py
- src/pathfact/validation/canonical.py

Historical scientific logic was extracted and adapted for a portable CPU-only package.
Reproducibility is established through regression tests, not source-file byte identity.

### Historical VSP evaluation

Original file: evaluate_deepseek_zeroshot.py
SHA-256: 1196512e57d5dcca84fdff5c7a75d8e67f1d3346340ea1b683a4b4e9b7d94b75

Related PathFACT modules:
- src/pathfact/evaluation/historical_vsp.py
- src/pathfact/evaluation/extraction.py
- src/pathfact/evaluation/batch.py
- src/pathfact/evaluation/saved.py

The historical VSP functions preserve their original scoring semantics.
The other evaluation modules provide reusable CPU-only extraction and offline evaluation interfaces.

## 2. Verified reproducibility evidence

- G00: 2,200 frozen graph-generation and shortest-path cases verified.
- G06: seven historical dataset outputs reproduced byte-for-byte.
- G05: 244/244 joint-valid paired cases passed cross-stage Student/Teacher provenance validation.
- G05 validation tiers: 141 pairs have additional two-sided reasoning factuality validation profiles; 103 pairs have VSP-only profiles.
- Canonical pipeline freeze: all 86 recorded SHA-256 entries verified.
- G09 independent freeze manifest: all 12 artifact SHA-256 entries verified.
- G02/G04: 1,779 historical Student and Teacher VSP records matched.
- G08/G09: 10,338 saved evaluation records matched across 16 runs.
- G08/G09: historical answer extraction reproduced for all 10,338 records.
- G08 Balanced-300 Base task-only: 42 valid predictions reproduced.
- Historical provenance regression snapshot: 82 tests passed with historical references configured; zero failures and zero skipped tests. This predates the release-candidate builder tests and is not the current suite total.
- Earlier release-preparation audit (before Vanilla-SFT expansion): 113 tests discovered, 103 passed and 10 frozen-source checks skipped because reference datasets were not configured. Seven require `PATHFACT_PIPELINE_REFERENCE`, two require `PATHFACT_G00_REFERENCE`, and one requires `PATHFACT_G06_REFERENCE`. These optional frozen datasets are not required for ordinary CPU-only use.
- Local CPU regression on 2026-11-12, following the Vanilla-SFT extension: 122 tests discovered, 112 passed, 10 frozen-reference tests skipped, and zero failures. The skipped checks do not constitute verification against the unavailable historical reference files.

The historical reference datasets are not required for ordinary public CPU-only execution.

## 3. Release boundaries

- Original research scripts remain internal provenance references.
- Public execution does not require the original research infrastructure.
- Model inference and GPU training are supplied by researchers.
- Historical experimental results must remain unchanged.
- Historical technical and licensing records should be read in their dated context; they do not replace the current release scope stated in the Hugging Face Dataset Card.

Source fingerprints and regression tests establish traceability and observed equivalence; they do not establish redistribution rights.

## 4. Vanilla-SFT provenance

### Historical research records

The original internal Stage C Vanilla-SFT resources contain 200
training records and 44 loss-validation records. They preserve
answer-only prompt/completion content associated with the historical
Vanilla-SFT research condition.

The original files are:

- `answer_only_train_200_prompt_completion_v1.jsonl`
- `answer_only_loss_val_44_prompt_completion_v1.jsonl`

A subsequent audit identified an internal checksum discrepancy:
the `completion_sha256` fields retained hashes of earlier
empty-think completions rather than the visible answer-only
completion strings.

The original research files remain unchanged. Their file-level
SHA-256 values and their historical content must not be confused
with the corrected public candidate files.

### Reconstructed public Vanilla-SFT resources

The CPU-only PathFACT builder derives Vanilla-SFT records from
the frozen G06 Gold Train-200 and LossVal-44 resources.

It preserves the Gold case IDs, record ordering and task-only
prompts, while converting the final path to an answer-only
completion of the form `Answer: [node, ...]`.

The six-field prompt/completion records are regenerated with
SHA-256 values calculated from their actual text.

The expanded Core exporter validates Vanilla against Gold,
including schema, record counts, case IDs, ordering, prompts,
answers and hashes. It requires disjoint Train/LossVal case
sets and copies the validated files byte-for-byte into the
candidate, recording their hashes in the integrity manifest.

This is a reconstruction from previously validated supervision,
not an independent shortest-path solver run. The reconstructed
files are not claimed to be byte-identical to the historical
Stage C files.

### Historical model-training evidence boundary

The available historical records support the existence of the
Vanilla-SFT experimental condition and its answer-only
supervision. They do not establish a verified exact input-file
SHA-256 mapping to a particular historical trained checkpoint.

No such checkpoint-to-dataset-hash claim is made.

### Release boundary

The published PathFACT v0.1 suite contains 10 Core and 3 Derived
JSONL resources. The optional expansion adds two Vanilla-SFT
Core resources, yielding a proposed 12 Core plus 3 Derived suite.

The expansion does not alter the existing published resources.
The expanded export remains a local candidate until its actual
release artifacts, provenance metadata and redistribution scope
have been reviewed.
