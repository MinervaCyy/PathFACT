# PathFACT Core Dataset Release Candidate Builder

Status: implemented local-candidate operation. This builder creates new local candidates; it is separate from the already published PathFACT v0.1 dataset.

## Scope

The core builder assembles only the first PathFACT release phase:

- G00: allowlisted shortest-path task records.
- G05: allowlisted paired Student, Gold Teacher and Contrastive Teacher records.
- G06: the four frozen Gold/Contrastive Train and LossVal JSONL files.
- G08: normalized Balanced-300 and TrainDist-300 evaluation views.
- G09: normalized Balanced-846 and TrainDist-1600 evaluation views.

G02, G04a and G04b are validation inputs for the G05 export. They are not
exported as standalone files. G08 Diagnostic-117, G09 Pool-1956 and derived
datasets are outside this operation.

## Command

    pathfact dataset export-core-candidate \
      --g00 G00.jsonl \
      --g02 G02.jsonl \
      --g04a G04A.jsonl \
      --g04b G04B.jsonl \
      --g05 G05.jsonl \
      --g06-gold-train gold_train_200_prompt_completion_v1.jsonl \
      --g06-contrastive-train contrastive_train_200_prompt_completion_v1.jsonl \
      --g06-gold-loss-val gold_loss_val_44_prompt_completion_v1.jsonl \
      --g06-contrastive-loss-val contrastive_loss_val_44_prompt_completion_v1.jsonl \
      --g08-balanced G08_BALANCED_300_V1.jsonl \
      --g08-train-dist G08_TRAIN_DISTRIBUTION_300_V1.jsonl \
      --g09-balanced G09_BALANCED_846_V1.jsonl \
      --g09-train-dist G09_TRAIN_DISTRIBUTION_1600_V1.jsonl \
      --dataset-version local-core-v0.1 \
      --output-dir pathfact_core_candidate

The default expected counts are G00=2,200, G05=244, Train=200 per condition
and LossVal=44 per condition. The default G05 tier counts are 141 two-sided
reasoning-factuality-profiled pairs and 103 VSP-only pairs. Count options are
configurable to support portable synthetic validation; a canonical release
candidate should use the defaults.

Evaluation defaults require 300/300 G08 records and 846/1,600 G09 records.
They also require disjoint G08 views; a 617-case G09 overlap; cross-view
overlaps of 164/221 for G08 Balanced against G09 Balanced/TrainDist; and
110/265 for G08 TrainDist against G09 Balanced/TrainDist. The overlap counts
are configurable only to support smaller synthetic fixtures.

The destination parent must exist and the destination itself must not exist.
The builder stages its output before renaming it into place and does not
overwrite an existing directory.

## Candidate layout (10-Core compatible mode)

    manifest.json
    metadata/summary.json
    tasks/g00_tasks.jsonl
    paired/g05_paired.jsonl
    training/gold_train_200_prompt_completion_v1.jsonl
    training/contrastive_train_200_prompt_completion_v1.jsonl
    training/gold_loss_val_44_prompt_completion_v1.jsonl
    training/contrastive_loss_val_44_prompt_completion_v1.jsonl
    evaluation/g08_balanced_300.jsonl
    evaluation/g08_train_distribution_300.jsonl
    evaluation/g09_balanced_846.jsonl
    evaluation/g09_train_distribution_1600.jsonl

`metadata/summary.json` is marked `local_candidate_not_approved` with a null
licence. Manifest v0.1 remains `draft_not_approved` with a null licence.
Neither status authorizes redistribution.

## Optional Vanilla-SFT expansion (12-Core candidate)

The command and layout above remain supported without modification. They
produce the original 10-Core candidate, containing four G06 training files.

For an expanded candidate, provide both additional G06 inputs:

    --g06-vanilla-train answer_only_train_200_prompt_completion_v1.jsonl
    --g06-vanilla-loss-val answer_only_loss_val_44_prompt_completion_v1.jsonl

Both options must be supplied together. The expanded candidate contains
12 Core JSONL resources, adding these files under `training/`:

    training/answer_only_train_200_prompt_completion_v1.jsonl
    training/answer_only_loss_val_44_prompt_completion_v1.jsonl

The three Derived resources are built separately. Together, the proposed
expanded suite would contain 12 Core plus 3 Derived JSONL resources.
This expansion does not modify the already published v0.1 resources.

### Vanilla-SFT materialisation

Vanilla-SFT uses the same frozen G06 Gold source problems and task-only
prompts as the ordinary Teacher supervision. Its target contains only the
final shortest-path answer, in the form `Answer: [node, ...]`, without a
reasoning trace.

Generate the release-ready Vanilla files from frozen G06 Gold inputs:

    pathfact dataset materialize-vanilla       --gold-train gold_train_200_prompt_completion_v1.jsonl       --gold-loss-val gold_loss_val_44_prompt_completion_v1.jsonl       --output-dir vanilla_g06

The operation requires 200 Train records and 44 LossVal records. It
recomputes the completion hashes from the actual answer-only text.

The expanded Core exporter validates Vanilla against the Gold records,
including exact field schema, case IDs and ordering, prompts, final
answers, internal hashes and Train/LossVal disjointness. It then copies
the validated Vanilla files byte-for-byte, verifies their file hashes
and includes them in the release manifest.

The Gold records originate from the validated G06 workflow. Vanilla
materialisation does not independently run a shortest-path solver.

### Historical Vanilla provenance

The original internal Stage C Vanilla files contain genuine historical
answer-only prompt/completion records, but their `completion_sha256`
fields retain hashes associated with earlier empty-think completions.
Those hashes do not match the visible answer-only completion strings.

The original research files must remain unchanged. For the expanded
public candidate, regenerate Vanilla from the frozen Gold G06 inputs
and use the corrected, internally consistent files.

This reconstruction does not by itself establish the exact input-file
hash used to train a historical Vanilla model checkpoint. No such
checkpoint-to-file identification is claimed here.

### Expanded export integrity

With both Vanilla inputs, the manifest verifies 12 Core JSONL files
rather than 10. `metadata/summary.json` includes a Vanilla validation
report under `g06.vanilla`.

Both export modes remain local-candidate operations. Their manifests
and summaries do not constitute redistribution approval.

## Validation and preservation

The builder:

- uses the established G00 field allowlist;
- reuses the G05 cross-stage identity, text-hash, Student-failure and Teacher
  validation-profile checks;
- preserves the G05 validation tier without independently re-auditing Teacher
  reasoning;
- requires each G06 record to match the established six-field frozen schema;
- verifies G06 prompt and completion SHA-256 values;
- requires matching Gold/Contrastive case IDs, disjoint Train/LossVal IDs and
  exact agreement between the G06 case union and G05;
- copies each G06 JSONL byte-for-byte and verifies the copied SHA-256;
- adapts nested G08 and top-level G09 sources through separate allowlists;
- separates model-facing topology/node-pair inputs from reference path/cost
  labels and task provenance;
- preserves source-stage and view membership, plus G08 source difficulty;
- rejects duplicate view IDs, inconsistent shared task/reference data,
  unexpected overlaps and any G06 evaluation overlap;
- records source and exported hashes in the summary and draft manifest;
- detects source changes during the operation; and
- leaves all supplied source files unchanged.

For the canonical data, the required G05 tier counts are 141 pairs with
two-sided reasoning-factuality validation profiles and 103 VSP-only pairs.

## Limitations

- This operation creates a local candidate; it does not publish data.
- Content, third-party terms, attribution and exact file-level release review
  remain required before publication.
- Derived-data release builders are not implemented here.
- G06 ranking and ID-list metadata are not included in this first builder.
- Manifest v0.1 is an integrity manifest, not a release-approval mechanism.
