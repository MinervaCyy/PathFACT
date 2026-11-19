# PathFACT Dataset Map

The published v0.3 suite contains twelve Core and three Derived JSONL
resources (15 total). The prior v0.2 suite also contained 15 resources; v0.3
renames the six S06 SFT paths without changing file bytes. The 13-resource
v0.1 suite (10 Core, 3 Derived) and v0.2 naming layout remain historical.
Public stage names use S00–S09. Retained paths and machine-readable identifiers such as `g00_tasks.jsonl` and `g05_paired.jsonl` preserve the released artifact names. Counts below were checked against the v0.3 upload manifest and released
records. This file is the GitHub repository guide; the Hugging Face
[Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT) is a
separate publication document with license scope and release disclosures.
Join records across stages with
`production_case_id` where present.

## Core resources (12)

| Resource | Rows | Contents and intended use | Boundary / relationship |
|---|---:|---|---|
| `tasks/g00_tasks.jsonl` | 2,200 | Graph tasks, endpoints, shortest-path reference labels, and generator provenance. | Task/reference records; contains no Student responses. |
| `paired/g05_paired.jsonl` | 244 | Selected genuine Student failures paired with Gold Teacher targets and Contrastive corrective targets. | Join to S00 using case identity. 141 pairs have additional reasoning factuality profiles; 103 have final-path checks only. |
| `training/limo_train_200_prompt_completion_v1.jsonl` | 200 | LIMO SFT with ordinary Teacher reasoning. | S06 Train; condition is `gold`; disjoint from LossVal. |
| `training/contrastive_limo_train_200_prompt_completion_v1.jsonl` | 200 | ContrastiveLIMO SFT with failure-conditioned corrective Teacher reasoning. | Same 200 case identities as LIMO Train; condition is `contrastive`. |
| `training/limo_loss_val_44_prompt_completion_v1.jsonl` | 44 | LIMO loss-validation examples with ordinary Teacher reasoning. | S06 LossVal; condition is `gold`; disjoint from Train. |
| `training/contrastive_limo_loss_val_44_prompt_completion_v1.jsonl` | 44 | ContrastiveLIMO loss-validation examples. | Same 44 case identities as LIMO LossVal; condition is `contrastive`. |
| `training/vanilla_sft_train_200_prompt_completion_v1.jsonl` | 200 | Vanilla-SFT final-answer-only examples. | S06 Train; condition is `vanilla`; shares LIMO Train case IDs and order. |
| `training/vanilla_sft_loss_val_44_prompt_completion_v1.jsonl` | 44 | Vanilla-SFT final-answer-only loss-validation examples. | S06 LossVal; condition is `vanilla`; shares LIMO LossVal case IDs and order. |
| `evaluation/g08_balanced_300.jsonl` | 300 | Balanced S08 evaluation view, normalized to model input, reference, and provenance. | Disjoint from S08 TrainDist; overlaps S09. |
| `evaluation/g08_train_distribution_300.jsonl` | 300 | S08 evaluation sampled to the training distribution. | Disjoint from S08 Balanced; overlaps S09. |
| `evaluation/g09_balanced_846.jsonl` | 846 | Larger balanced S09 evaluation view. | Shares 617 cases with S09 TrainDist. |
| `evaluation/g09_train_distribution_1600.jsonl` | 1,600 | Larger S09 training-distribution evaluation view. | Shares 617 cases with S09 Balanced. |

S02, S04a, and S04b are validation inputs used to construct/check S05, not
additional standalone Core release files. S01, S03, and S07 are intermediate
pipeline stages. S08 Diagnostic-117 and S09 Pool-1956 are not among the twelve
Core files.

## SFT methods and dataset identifiers

| Method | Dataset identifier | Published S06 files | Supervision |
|---|---|---|---|
| Vanilla-SFT | `vanilla` | `vanilla_sft_train_200_prompt_completion_v1.jsonl`; `vanilla_sft_loss_val_44_prompt_completion_v1.jsonl` | Final answer only, with no reasoning trace. |
| LIMO | `gold` | `limo_train_200_prompt_completion_v1.jsonl`; `limo_loss_val_44_prompt_completion_v1.jsonl` | Ordinary Teacher reasoning and verified answer. |
| ContrastiveLIMO | `contrastive` | `contrastive_limo_train_200_prompt_completion_v1.jsonl`; `contrastive_limo_loss_val_44_prompt_completion_v1.jsonl` | Failure-conditioned corrective Teacher reasoning and verified answer. |

The method names are distinct from condition identifiers. Hub configuration
IDs and record values (`vanilla`, `gold`, `contrastive`) are preserved.

## v0.3 S06 filename migration

The six v0.2 paths remain available at the pinned `pathfact-v0.2` Hugging
Face revision. The v0.3 paths below contain byte-identical files:

| v0.2 path | v0.3 path |
|---|---|
| `training/answer_only_train_200_prompt_completion_v1.jsonl` | `training/vanilla_sft_train_200_prompt_completion_v1.jsonl` |
| `training/answer_only_loss_val_44_prompt_completion_v1.jsonl` | `training/vanilla_sft_loss_val_44_prompt_completion_v1.jsonl` |
| `training/gold_train_200_prompt_completion_v1.jsonl` | `training/limo_train_200_prompt_completion_v1.jsonl` |
| `training/gold_loss_val_44_prompt_completion_v1.jsonl` | `training/limo_loss_val_44_prompt_completion_v1.jsonl` |
| `training/contrastive_train_200_prompt_completion_v1.jsonl` | `training/contrastive_limo_train_200_prompt_completion_v1.jsonl` |
| `training/contrastive_loss_val_44_prompt_completion_v1.jsonl` | `training/contrastive_limo_loss_val_44_prompt_completion_v1.jsonl` |

The public exporter defaults to method-based filenames. Use
`--naming-mode legacy` to reproduce the v0.1/v0.2 paths.

## Derived resources (3)

| Resource | Rows | Contents and intended use | Boundary / relationship |
|---|---:|---|---|
| `counterfactual/counterfactual_triples.jsonl` | 354 | Original/Stay/Switch graph variants for controlled edge-weight changes. | Represents 354 unique cases and 482 S09 view memberships; no model responses. |
| `training/ranking_eligibility.jsonl` | 200 | S06 cases classified by comparing old/new route costs on Original and Switch graphs. | 162 strict bidirectional, 9 Original-tie/Switch-strict, 29 without qualifying strict Switch. Ineligible rows use nulls for unavailable Switch data. Eligibility is geometric, not model likelihood. |
| `training/boundary_local.jsonl` | 162 | Local Stay/Switch edge-weight boundary constructions linked to the strict S06 subset. | Stay and Switch lie on opposite sides of a tie weight, with one-unit objective margins checked by solver. Not independent task-only path-solving results. |

## Training and evaluation boundaries

The four evaluation views contain 3,046 row memberships and 1,859 unique case
IDs. Memberships across views are repeated appearances of a task, not new
independent tasks. The four S08/S09 evaluation views have no case IDs
in S06 Train or LossVal. This establishes disjointness from these specific
training splits, not every possible external training corpus. S08 Balanced
and S08 TrainDist are disjoint and partition the 600 selected S08 cases.
Across S09 Balanced and TrainDist, 617 cases overlap; their union has 1,829
unique cases and 127 cases from the 1,956-case S09 pool are in neither view.
Cross-version overlaps are also present: S08 Balanced overlaps S09 Balanced
and TrainDist by 164 and 221; S08 TrainDist overlaps them by 110 and 265.
These are designed memberships, not evidence of leakage. Do not describe the
two S09 views as independent case-level replications.

The Hub manifest gives each resource its own configuration. S06 files use
`train` or `validation` split labels; S00, S05, S08/S09, and Derived files use
the neutral `data` label. In particular, an S08/S09 file is an evaluation
resource because of its documented role, not because the Hub labels it
`test`.

Evaluation rows intentionally separate `model_input` (topology and endpoint
pair) from `reference` (gold path and cost). The published evaluation files
expose both; withhold `reference` from the model. Derived counterfactual rows
hold topology fixed and edit only the selected edge's weight across Original,
Stay and Switch. Derived files are graph-verified
resources and do not contain generated model answers, learned preference
scores, or StepPref data.

For row-level details, see [Data Dictionary](DATA_DICTIONARY.md). For the
project's history and exact release caveats, consult the separate Dataset Card and
[Source Provenance](docs/SOURCE_PROVENANCE.md).
