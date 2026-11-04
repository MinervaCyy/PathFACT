# PathFACT Dataset Map

The fixed v0.1 public revision contains ten Core and three Derived JSONL
resources. Counts below were checked against the upload manifest and all
released records. This file is the GitHub repository guide; the Hugging Face
[Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT) is a
separate publication document with license scope and release disclosures.
Join records across stages with
`production_case_id` where present.

## Core resources (10)

| Resource | Rows | Contents and intended use | Boundary / relationship |
|---|---:|---|---|
| `tasks/g00_tasks.jsonl` | 2,200 | Graph tasks, endpoints, shortest-path reference labels, and generator provenance. | Task/reference records; contains no Student responses. |
| `paired/g05_paired.jsonl` | 244 | Selected genuine Student failures paired with Gold Teacher targets and Contrastive corrective targets. | Join to G00 using case identity. 141 pairs have additional reasoning factuality profiles; 103 have final-path checks only. |
| `training/gold_train_200_prompt_completion_v1.jsonl` | 200 | Frozen Gold SFT prompt/completion examples. | G06 Train; disjoint from LossVal. |
| `training/contrastive_train_200_prompt_completion_v1.jsonl` | 200 | Frozen Contrastive SFT examples. | Same 200 case identities as Gold Train. |
| `training/gold_loss_val_44_prompt_completion_v1.jsonl` | 44 | Gold loss-validation examples. | G06 LossVal; disjoint from Train. |
| `training/contrastive_loss_val_44_prompt_completion_v1.jsonl` | 44 | Contrastive loss-validation examples. | Same 44 case identities as Gold LossVal. |
| `evaluation/g08_balanced_300.jsonl` | 300 | Balanced G08 evaluation view, normalized to model input, reference, and provenance. | Disjoint from G08 TrainDist; overlaps G09. |
| `evaluation/g08_train_distribution_300.jsonl` | 300 | G08 evaluation sampled to the training distribution. | Disjoint from G08 Balanced; overlaps G09. |
| `evaluation/g09_balanced_846.jsonl` | 846 | Larger balanced G09 evaluation view. | Shares 617 cases with G09 TrainDist. |
| `evaluation/g09_train_distribution_1600.jsonl` | 1,600 | Larger G09 training-distribution evaluation view. | Shares 617 cases with G09 Balanced. |

G02, G04a, and G04b are validation inputs used to construct/check G05, not
additional standalone Core release files. G01, G03, and G07 are intermediate
pipeline stages. G08 Diagnostic-117 and G09 Pool-1956 are not among the ten
Core files.

## Derived resources (3)

| Resource | Rows | Contents and intended use | Boundary / relationship |
|---|---:|---|---|
| `counterfactual/counterfactual_triples.jsonl` | 354 | Original/Stay/Switch graph variants for controlled edge-weight changes. | Represents 354 unique cases and 482 G09 view memberships; no model responses. |
| `training/ranking_eligibility.jsonl` | 200 | G06 cases classified by comparing old/new route costs on Original and Switch graphs. | 162 strict bidirectional, 9 Original-tie/Switch-strict, 29 without qualifying strict Switch. Ineligible rows use nulls for unavailable Switch data. Eligibility is geometric, not model likelihood. |
| `training/boundary_local.jsonl` | 162 | Local Stay/Switch edge-weight boundary constructions linked to the strict G06 subset. | Stay and Switch lie on opposite sides of a tie weight, with one-unit objective margins checked by solver. Not independent task-only path-solving results. |

## Training and evaluation boundaries

The four evaluation views contain 3,046 row memberships and 1,859 unique case
IDs. Memberships across views are repeated appearances of a task, not new
independent tasks. The four G08/G09 evaluation views have no case IDs
in G06 Train or LossVal. This establishes disjointness from these specific
training splits, not every possible external training corpus. G08 Balanced
and G08 TrainDist are disjoint and partition the 600 selected G08 cases.
Across G09 Balanced and TrainDist, 617 cases overlap; their union has 1,829
unique cases and 127 cases from the 1,956-case G09 pool are in neither view.
Cross-version overlaps are also present: G08 Balanced overlaps G09 Balanced
and TrainDist by 164 and 221; G08 TrainDist overlaps them by 110 and 265.
These are designed memberships, not evidence of leakage. Do not describe the
two G09 views as independent case-level replications.

The Hub manifest gives each resource its own configuration. G06 files use
`train` or `validation` split labels; G00, G05, G08/G09, and Derived files use
the neutral `data` label. In particular, a G08/G09 file is an evaluation
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
