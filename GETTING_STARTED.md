# Getting Started with PathFACT

Source attribution for examples: PathFACT Dataset v0.1, MinervaCYY/PathFACT, [fixed revision `3d20d452e1efe21c80d55185f5176ca3fc5e4f9f`](https://huggingface.co/datasets/MinervaCYY/PathFACT/tree/3d20d452e1efe21c80d55185f5176ca3fc5e4f9f). Dataset excerpts are subject to the release's limited CC BY 4.0 scope and exclusions described in the [Hugging Face Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT). The repository's MIT licence applies to software, not automatically to dataset content.

## 1. What is PathFACT?

PathFACT is a research framework and dataset for studying how language models
solve weighted shortest-path problems, where they make mistakes, and how
corrections can be used for supervision. Its released resources include graph
tasks, selected Student failures paired with Teacher targets, frozen training
examples, evaluation views, and derived graph resources.

PathFACT is the overall project. ContrastiveLIMO names one historical
failure-conditioned supervised fine-tuning method; it does not describe every
PathFACT resource. The CPU software does not provide model inference, GPU
training, or model-dependent answer-likelihood scoring.

## 2. Understanding a shortest-path task

A task gives you a weighted, undirected graph and an ordered pair of endpoints.
Find a route from the first endpoint to the second whose edge weights have the
smallest possible sum. The number of edges (hops) is not the objective. There
can be more than one route with the same minimum cost.

In the released G00 example with optimal cost 746, the actual
`production_case_id` is **`PR01233`**. Its endpoints are `[0, 5]`; the reference
route is `[0, 1, 2, 3, 4, 5]`, with consecutive edge weights 160, 121, 190,
108, and 167. These sum to 746. This particular graph has a unique route; that
does not mean every PathFACT task has a unique optimum. G00 provides tasks and
reference labels, not Student responses.

## 3. From Student mistakes to Teacher corrections

G05 is the released paired resource for selected genuine Student failures.
Each record links the task to a Student trace and predicted route, a Gold
Teacher target, and a Contrastive corrective target linked to that Student
trace. A route can fail because it has invalid endpoints or edges, repeats a
node, or is valid but costs more than the minimum.

The recorded `vsp_valid` flag reports whether the extracted final route passes
the verifier's path and optimal-cost checks. It does not mean every sentence
in the Student's reasoning was independently judged false. Both Teacher
final paths pass those checks. Of the 244 pairs, 141 have additional reasoning
factuality validation profiles and 103 have solver-verified final-path checks
only. The Contrastive `wrong_rt_continuity_verified` flag checks linkage to the Student
trace; it is not a factuality judgment. See the [Data Dictionary](DATA_DICTIONARY.md)
for field definitions and validation details.

## 4. Which dataset should I use?

- **Standalone tasks or evaluation inputs:** `tasks/g00_tasks.jsonl` contains
  graph tasks and reference labels, with no Student responses.
- **Failure-conditioned supervision:** `paired/g05_paired.jsonl` links
  selected Student failures with Gold and Contrastive Teacher targets.
- **Frozen supervised fine-tuning examples:** the six `training/` G06 files
  cover three methods. Vanilla-SFT uses the historical `answer_only` filename
  prefix and the `vanilla` record condition for final-answer-only targets;
  LIMO uses `gold` for ordinary Teacher reasoning; ContrastiveLIMO uses
  `contrastive` for failure-conditioned corrective Teacher reasoning. Each
  method has Train-200 and LossVal-44 files. These are small task-specific
  resources, not a general-purpose corpus.
- **Completed evaluation views:** the four G08/G09 files in `evaluation/`
  contain model inputs and reference answers. Keep `reference` out of the model
  input. The two G09 views overlap, so they are not independent case-level
  replications.
- **Controlled graph changes:**
  `counterfactual/counterfactual_triples.jsonl` contains Original/Stay/Switch
  graph variants for edited edge weights.
- **Graph-derived eligibility and boundary cases:**
  `training/ranking_eligibility.jsonl` and `training/boundary_local.jsonl`
  describe geometric path-cost comparisons. They are not model-response
  preferences or learned preference scores.

The [Dataset Map](DATASET_MAP.md) lists all 15 v0.2 resources, counts, and
intended uses. Its overlap section is the reference for detailed split
relationships.

## 5. Read your first PathFACT record

Each JSONL line is one JSON object. In a G00 task, `topology` maps node IDs to
neighbor/weight pairs, and `node_pair` gives source then target. Download the desired JSONL from the [current PathFACT dataset](https://huggingface.co/datasets/MinervaCYY/PathFACT); the v0.1 fixed revision remains available for historical comparisons. The example assumes your working directory is the release-package root, where `tasks/g00_tasks.jsonl` is located; adjust `dataset` if you store it elsewhere.

The following small Python example loads the first task and prints its
endpoints. It uses only Python's standard library:

```python
import json
from pathlib import Path

dataset = Path("tasks/g00_tasks.jsonl")
with dataset.open(encoding="utf-8") as records:
    first_task = json.loads(next(records))

print(first_task["node_pair"])
print(first_task["topology"])
```

This example was run against the released 2,200-row G00 JSONL; it parsed the
first row and read its required fields successfully. Use the same pattern to
stream a large JSONL file without loading every row into memory.

## 6. Essential information versus advanced metadata

For solving a task, the essential input is the graph (`topology`) and endpoint
pair (`node_pair`). In an evaluation record these appear under `model_input`.
Reference labels (`ground_truth_shortest_path` and `gold_cost`, or the
evaluation record's `reference`) are answers for scoring; withhold them from
the model during evaluation.

Other fields record identity, graph-generation settings, selection, or
sampling. For example, `production_case_id` joins records across stages, while
G08/G09 view and provenance fields document where an evaluation case came from
and how it was selected. **Sampling metadata is not model input.** It should
not be appended to the model prompt as task information. The [Data Dictionary](DATA_DICTIONARY.md)
explains the released fields and their caveats.

## 7. Further documentation

- [Dataset Map](DATASET_MAP.md): released files, counts, and resource roles.
- [Data Dictionary](DATA_DICTIONARY.md): field meanings, examples, and caveats.
- [Offline Evaluation](docs/EVALUATION.md): evaluation inputs, outputs, scoring,
  and limitations.
- [Source Provenance](docs/SOURCE_PROVENANCE.md): historical sources and
  reproducibility evidence.
- [Hugging Face Dataset Card](https://huggingface.co/datasets/MinervaCYY/PathFACT):
  published dataset configurations, licensing scope, and release disclosures.
