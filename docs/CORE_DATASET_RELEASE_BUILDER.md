# PathFACT Core Dataset Release Candidate Builder

Status: implemented local-candidate operation. This builder creates new
local candidates. The expanded 12-Core candidate supports the published
PathFACT v0.2 suite; the original 10-Core mode preserves compatibility with
the historical v0.1 layout.

## Scope

The core builder assembles only the first PathFACT release phase:

- S00: allowlisted shortest-path task records.
- S05: allowlisted paired Student, Gold Teacher and Contrastive Teacher records.
- S06: four frozen Gold/Contrastive Teacher-reasoning files and, when both
  optional inputs are supplied, two Vanilla-SFT answer-only files.
- S08: normalized Balanced-300 and TrainDist-300 evaluation views.
- S09: normalized Balanced-846 and TrainDist-1600 evaluation views.

S02, S04a and S04b are validation inputs for the S05 export. They are not
exported as standalone files. S08 Diagnostic-117, S09 Pool-1956 and derived
datasets are outside this operation.

## Command

    pathfact dataset export-core-candidate \
      --g00 S00.jsonl \
      --g02 S02.jsonl \
      --g04a S04a.jsonl \
      --g04b S04b.jsonl \
      --g05 S05.jsonl \
      --g06-gold-train gold_train_200_prompt_completion_v1.jsonl \
      --g06-contrastive-train contrastive_train_200_prompt_completion_v1.jsonl \
      --g06-gold-loss-val gold_loss_val_44_prompt_completion_v1.jsonl \
      --g06-contrastive-loss-val contrastive_loss_val_44_prompt_completion_v1.jsonl \
      --g08-balanced G08_BALANCED_300_V1.jsonl \
      --g08-train-dist G08_TRAIN_DISTRIBUTION_300_V1.jsonl \
      --g09-balanced G09_BALANCED_846_V1.jsonl \
      --g09-train-dist G09_TRAIN_DISTRIBUTION_1600_V1.jsonl \
      --dataset-version local-core-v0.3 \
      --output-dir pathfact_core_candidate

The public stages use S00–S09. For released artifact compatibility, the exporter retains its historical G-based CLI option names, file names, Hub configurations, and machine-readable source-stage values; these identify artifacts, not current stage terminology. The exporter defaults to method-based v0.3 filenames. Pass
`--naming-mode legacy` to reproduce the v0.1/v0.2 filenames.

The default expected counts are S00=2,200, S05=244, Train=200 per condition
and LossVal=44 per condition. The default S05 tier counts are 141 two-sided
reasoning-factuality-profiled pairs and 103 VSP-only pairs. Count options are
configurable to support portable synthetic validation; a canonical release
candidate should use the defaults.

Evaluation defaults require 300/300 S08 records and 846/1,600 S09 records.
They also require disjoint S08 views; a 617-case S09 overlap; cross-view
overlaps of 164/221 for S08 Balanced against S09 Balanced/TrainDist; and
110/265 for S08 TrainDist against S09 Balanced/TrainDist. The overlap counts
are configurable only to support smaller synthetic fixtures.

The destination parent must exist and the destination itself must not exist.
The builder stages its output before renaming it into place and does not
overwrite an existing directory.

## Candidate layout (default v0.3 method names; 10 Core)

    manifest.json
    metadata/summary.json
    tasks/g00_tasks.jsonl
    paired/g05_paired.jsonl
    training/limo_train_200_prompt_completion_v1.jsonl
    training/contrastive_limo_train_200_prompt_completion_v1.jsonl
    training/limo_loss_val_44_prompt_completion_v1.jsonl
    training/contrastive_limo_loss_val_44_prompt_completion_v1.jsonl
    evaluation/g08_balanced_300.jsonl
    evaluation/g08_train_distribution_300.jsonl
    evaluation/g09_balanced_846.jsonl
    evaluation/g09_train_distribution_1600.jsonl

`metadata/summary.json` is marked `local_candidate_not_approved` with a null
licence. Manifest v0.1 remains `draft_not_approved` with a null licence.
Neither status authorizes redistribution.

## Optional Vanilla-SFT expansion (12-Core candidate; introduced in v0.2)

Both modes produce a 10-Core candidate when no Vanilla inputs are supplied.
The optional inputs add the two Vanilla-SFT resources, producing 12 Core.

For an expanded candidate, provide both additional S06 inputs:

    --g06-vanilla-train answer_only_train_200_prompt_completion_v1.jsonl
    --g06-vanilla-loss-val answer_only_loss_val_44_prompt_completion_v1.jsonl

Both options must be supplied together. The expanded candidate contains
12 Core JSONL resources, adding these files under `training/`:

    training/vanilla_sft_train_200_prompt_completion_v1.jsonl
    training/vanilla_sft_loss_val_44_prompt_completion_v1.jsonl

The three Derived resources are built separately. The published v0.3 suite
contains 12 Core plus 3 Derived JSONL resources. Its six S06 files are
byte-identical to their v0.2 counterparts, with the method-based names below.

### SFT method and data identifiers

| Method | Identifier | S06 supervision |
|---|---|---|
| Vanilla-SFT | Record condition `vanilla` | Final answer only. |
| LIMO | `gold` | Ordinary Teacher reasoning and verified answer. |
| ContrastiveLIMO | `contrastive` | Failure-conditioned corrective Teacher reasoning and verified answer. |

The method names are distinct from the preserved dataset condition values.
Existing Hub configuration names and record values are not renamed.

## Legacy naming mode (v0.1/v0.2)

Use `--naming-mode legacy` with `export-core-candidate` to reproduce these
historical paths:

    training/gold_train_200_prompt_completion_v1.jsonl
    training/contrastive_train_200_prompt_completion_v1.jsonl
    training/gold_loss_val_44_prompt_completion_v1.jsonl
    training/contrastive_loss_val_44_prompt_completion_v1.jsonl
    training/answer_only_train_200_prompt_completion_v1.jsonl
    training/answer_only_loss_val_44_prompt_completion_v1.jsonl

The same option is available for `materialize-vanilla`. The Hub tags
`pathfact-v0.1` and `pathfact-v0.2` pin historical layouts; local legacy
exports retain the corresponding filenames without altering historical data.

### Vanilla-SFT materialisation

Vanilla-SFT uses the same frozen S06 Gold source problems and task-only
prompts as the ordinary Teacher supervision. Its target contains only the
final shortest-path answer, in the form `Answer: [node, ...]`, without a
reasoning trace.

Generate the release-ready Vanilla files from frozen S06 Gold inputs:

    pathfact dataset materialize-vanilla \
      --gold-train gold_train_200_prompt_completion_v1.jsonl \
      --gold-loss-val gold_loss_val_44_prompt_completion_v1.jsonl \
      --output-dir vanilla_g06

The command writes the `vanilla_sft_*` paths by default. Add
`--naming-mode legacy` to write the historical `answer_only_*` names.

The operation requires 200 Train records and 44 LossVal records. It
recomputes the completion hashes from the actual answer-only text.

The expanded Core exporter validates Vanilla against the Gold records,
including exact field schema, case IDs and ordering, prompts, final
answers, internal hashes and Train/LossVal disjointness. It then copies
the validated Vanilla files byte-for-byte, verifies their file hashes
and includes them in the release manifest.

The Gold records originate from the validated S06 workflow. Vanilla
materialisation does not independently run a shortest-path solver.

### Historical Vanilla provenance

The original internal Stage C Vanilla files contain genuine historical
answer-only prompt/completion records, but their `completion_sha256`
fields retain hashes associated with earlier empty-think completions.
Those hashes do not match the visible answer-only completion strings.

The original research files must remain unchanged. For the expanded
public candidate, regenerate Vanilla from the frozen Gold S06 inputs
and use the corrected, internally consistent files.

This reconstruction does not by itself establish the exact input-file
hash used to train a historical Vanilla model checkpoint. No such
checkpoint-to-file identification is claimed here.

### Expanded export integrity

With both Vanilla inputs, the manifest verifies 12 Core JSONL files
rather than 10. `metadata/summary.json` includes a Vanilla validation
report under `S06.vanilla`.

Both export modes remain local-candidate operations. Their manifests
and summaries do not constitute redistribution approval.

## Validation and preservation

The builder:

- uses the established S00 field allowlist;
- reuses the S05 cross-stage identity, text-hash, Student-failure and Teacher
  validation-profile checks;
- preserves the S05 validation tier without independently re-auditing Teacher
  reasoning;
- requires each S06 record to match the established six-field frozen schema;
- verifies S06 prompt and completion SHA-256 values;
- requires matching Gold/Contrastive case IDs, disjoint Train/LossVal IDs and
  exact agreement between the S06 case union and S05;
- copies each S06 JSONL byte-for-byte and verifies the copied SHA-256;
- adapts nested S08 and top-level S09 sources through separate allowlists;
- separates model-facing topology/node-pair inputs from reference path/cost
  labels and task provenance;
- preserves source-stage and view membership, plus S08 source difficulty;
- rejects duplicate view IDs, inconsistent shared task/reference data,
  unexpected overlaps and any S06 evaluation overlap;
- records source and exported hashes in the summary and draft manifest;
- detects source changes during the operation; and
- leaves all supplied source files unchanged.

For the canonical data, the required S05 tier counts are 141 pairs with
two-sided reasoning-factuality validation profiles and 103 VSP-only pairs.

## Limitations

- This operation creates a local candidate; it does not publish data.
- Content, third-party terms, attribution and exact file-level release review
  remain required before publication.
- Derived-data release builders are not implemented here.
- S06 ranking and ID-list metadata are not included in this first builder.
- Manifest v0.1 is an integrity manifest, not a release-approval mechanism.
